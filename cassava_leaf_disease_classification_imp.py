import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, models

# ==========================================
# 1. Configuration & GPU Setup
# ==========================================
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32   
NUM_EPOCHS = 10    
K_FOLDS = 5
LEARNING_RATE = 1.5e-5
NUM_CLASSES = 5    
IMG_SIZE = 384     

# Paths - according to new instruction
CSV_PATH = './Data/train.csv'
IMG_DIR = './Data/train_images'

# Check for GPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"   Memory Usage: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ GPU not detected. Running on CPU (slower).")

# ==========================================
# 2. Custom Dataset for Cassava CSV+Folder Structure
# ==========================================
class CassavaDataset(Dataset):
    """
    PyTorch Dataset for Cassava: Accepts DataFrame with image names and labels,
    fetches images from the specified directory.
    """
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'image_id']
        label = int(self.df.loc[idx, 'label'])
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Make sure label is a Python int and >=0 and <NUM_CLASSES
        if (label < 0) or (label >= NUM_CLASSES):
            raise ValueError(f"Image {img_name} has out-of-range label: {label}")
        return image, label

# ==========================================
# 3. Data Preparation
# ==========================================
def get_data():
    """
    1. Reads train.csv for image names and labels.
    2. Defines Augmentation (Train) and Clean (Val/Test) transforms.
    3. Splits 10% for Test, 90% for CV, stratified.
    Returns: (df, train_val_idx, test_idx, train_transform, val_test_transform, labels)
    """
    # --- A. Define Transforms (Based on Report) ---
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\nLoading Dataset from csv...")

    df = pd.read_csv(CSV_PATH)
    if not {'image_id', 'label'}.issubset(df.columns):
        raise ValueError("train.csv must have columns: 'image_id', 'label'")
    # Fix (1): Enforce integer type for label column
    df['label'] = pd.to_numeric(df['label'], downcast='integer', errors='raise')
    # Fix (2): Check for invalid labels
    if (df['label'] < 0).any() or (df['label'] >= NUM_CLASSES).any():
        bad_rows = df[(df['label'] < 0) | (df['label'] >= NUM_CLASSES)]
        raise ValueError(f"Found label(s) out of expected range [0, {NUM_CLASSES-1}]:\n{bad_rows}")

    labels = df['label'].values

    all_indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        all_indices, 
        test_size=0.1, 
        shuffle=True, 
        stratify=labels,  # ensure split keeps original imbalance
        random_state=42
    )

    print(f"Total Images: {len(df)}")
    print(f"Training/Val Indices: {len(train_val_idx)}")
    print(f"Hold-out Test Indices: {len(test_idx)}")
    
    return df, train_val_idx, test_idx, train_transform, val_test_transform, labels
# ==========================================
# 4. Model Definition
# ==========================================
def get_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(DEVICE)

# ==========================================
# 5. Train/Val Logic
# ==========================================
def run_epoch(model, loader, criterion, optimizer=None, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()
        
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.set_grad_enabled(is_train):
        for inputs, labels in loader:
            # Fix (3): Ensure labels are of type torch.long
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            if labels.dtype != torch.long:
                labels = labels.long()
            
            if is_train:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# ==========================================
# 6. Main Execution (WITH WEIGHTED K-FOLD)
# ==========================================
def compute_class_weights(labels, indices, num_classes):
    """
    Compute class weights as 1 / frequency, normalized so min=1.
    labels: full label array
    indices: data indices to use (e.g. train split)
    Returns FloatTensor of size [num_classes]
    """
    labels_in_split = np.array(labels)[indices]
    class_sample_count = np.array(
        [np.sum(labels_in_split == t) for t in range(num_classes)]
    )
    # If any class is missing, prevent divide-by-zero
    if (class_sample_count == 0).any():
        raise ValueError(f"Class missing from given split: {class_sample_count}")
    weights = 1. / (class_sample_count + 1e-6)
    # Normalize so min weight is 1.0
    weights = weights / weights.min()
    return torch.tensor(weights, dtype=torch.float32)

def main():
    start_total = time.time()
    
    # 1. Get Data & Transforms
    df, train_val_indices, test_indices, train_tf, val_tf, all_labels = get_data()
    
    # 2. Setup History Storage
    history = {
        'train_loss': np.zeros((K_FOLDS, NUM_EPOCHS)),
        'val_loss':   np.zeros((K_FOLDS, NUM_EPOCHS)),
        'train_acc':  np.zeros((K_FOLDS, NUM_EPOCHS)),
        'val_acc':    np.zeros((K_FOLDS, NUM_EPOCHS))
    }

    # 3. Use StratifiedKFold for weighted/stratified splitting
    train_val_labels = np.array(all_labels)[train_val_indices]
    kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    best_overall_acc = 0.0
    best_model_state = None

    print(f"\nStarting {K_FOLDS}-Fold STRATIFIED (Weighted) Cross-Validation...")
    print("="*50)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_indices, train_val_labels)):
        print(f"\n FOLD {fold+1}/{K_FOLDS}")
        print("-" * 20)

        # actual split indices
        actual_train_idx = train_val_indices[train_idx]
        actual_val_idx = train_val_indices[val_idx]

        # Fix (4): Check for missing classes in training split
        split_labels = np.array(all_labels)[actual_train_idx]
        missing = set(range(NUM_CLASSES)) - set(split_labels)
        if missing:
            raise ValueError(f"Training data for fold {fold+1} is missing class(es): {missing}")

        # Compute weights for training set (for this fold)
        weights = compute_class_weights(all_labels, actual_train_idx, NUM_CLASSES)
        print("Class Weights (Fold {}): {}".format(fold+1, weights.cpu().numpy()))

        # Datasets from new structure
        train_ds = CassavaDataset(df.iloc[actual_train_idx], IMG_DIR, train_tf)
        val_ds = CassavaDataset(df.iloc[actual_val_idx], IMG_DIR, val_tf)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        # Model/criterion with WEIGHTED loss
        model = get_model(NUM_CLASSES)
        criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(NUM_EPOCHS):
            t_loss, t_acc = run_epoch(model, train_loader, criterion, optimizer, is_train=True)
            v_loss, v_acc = run_epoch(model, val_loader, criterion, is_train=False)

            history['train_loss'][fold, epoch] = t_loss
            history['val_loss'][fold, epoch]   = v_loss
            history['train_acc'][fold, epoch]  = t_acc
            history['val_acc'][fold, epoch]    = v_acc

            print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
                  f"Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | "
                  f"Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")

            if v_acc > best_overall_acc:
                best_overall_acc = v_acc
                best_model_state = model.state_dict()
                torch.save(best_model_state, f"{RESULTS_DIR}/best_model.pth")

    print("\nTraining Complete.")
    print("="*50)
    
    # ==========================================
    # 7. Aggregation & Plotting
    # ==========================================
    print("Generating Average Plots...")
    
    avg_results = {k: np.mean(v, axis=0) for k, v in history.items()}
    epochs_range = range(1, NUM_EPOCHS + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, avg_results['train_loss'], label='Mean Train Loss', marker='o')
    plt.plot(epochs_range, avg_results['val_loss'], label='Mean Val Loss', marker='o', linestyle='--')
    plt.title(f'Mean Loss over {K_FOLDS} Folds (Weighted)')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{RESULTS_DIR}/mean_loss_curve.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, avg_results['train_acc'], label='Mean Train Acc', marker='o')
    plt.plot(epochs_range, avg_results['val_acc'], label='Mean Val Acc', marker='o', linestyle='--')
    plt.title(f'Mean Accuracy over {K_FOLDS} Folds (Weighted)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{RESULTS_DIR}/mean_accuracy_curve.png")
    plt.close()

    # ==========================================
    # 8. Final Test Evaluation
    # ==========================================
    print(f"\nRestoring best model (Acc: {best_overall_acc:.4f}) for Final Testing...")
    
    final_model = get_model(NUM_CLASSES)
    final_model.load_state_dict(best_model_state)
    final_model.eval()
    
    # Prepare Test Data (Clean Transform)  
    test_ds = CassavaDataset(df.iloc[test_indices], IMG_DIR, val_tf)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    all_preds = []
    all_labels = []
    
    print("Running inference on Test Set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            # Fix (5): Ensure labels are ints and in correct range on test
            if labels.dtype != torch.long:
                labels = labels.long()
            if (labels < 0).any() or (labels >= NUM_CLASSES).any():
                raise ValueError(f"Test batch has out-of-range label: {labels}")
            outputs = final_model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Hold-out Test Set)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png")
    plt.close()
    
    report = classification_report(all_labels, all_preds)
    report_path = f"{RESULTS_DIR}/classification_report.txt"
    with open(report_path, "w") as f:
        f.write("FINAL TEST REPORT\n")
        f.write("=================\n")
        f.write(report)
    
    print(f"\n All Process Completed!")
    print(f"   - Plots saved to: {RESULTS_DIR}/")
    print(f"   - Report saved to: {report_path}")
    print(f"   - Total Time: {time.time() - start_total:.2f} seconds")
    print("\nText Report:")
    print(report)

if __name__ == "__main__":
    main()
