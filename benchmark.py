import os
import torch
import pandas as pd
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ----------- CassavaDataset Definition -----------
class CassavaDataset(Dataset):
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
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# --------------- Config ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
IMG_SIZE = 384
IMG_SIZE_VIT = 224  # ViT requires 224x224 inputs
BATCH_SIZE = 32
CSV_PATH = './Data/train.csv'
IMG_DIR = './Data/train_images'
RESULTS_CSV = './benchmark_results.csv'

EPOCHS = 10

# ------------- Transforms per model type -------------
def get_transform(model_name, is_train):
    if model_name == "ViT-B/16":
        size = IMG_SIZE_VIT
    else:
        size = IMG_SIZE
    if is_train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# ------------- Load Data -------------
def split_train_val():
    df = pd.read_csv(CSV_PATH)
    if not {'image_id', 'label'}.issubset(df.columns):
        raise ValueError("train.csv must have columns: 'image_id', 'label'")
    df['label'] = pd.to_numeric(df['label'], downcast='integer', errors='raise')
    labels = df['label'].values
    all_indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.1,
        shuffle=True,
        stratify=labels,
        random_state=42
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df

def get_data_loaders(model_name):
    train_df, test_df = split_train_val()
    train_tf = get_transform(model_name, is_train=True)
    test_tf = get_transform(model_name, is_train=False)
    train_ds = CassavaDataset(train_df, IMG_DIR, train_tf)
    test_ds = CassavaDataset(test_df, IMG_DIR, test_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    return train_loader, test_loader

# ------------- Model Loading -------------
def load_model(model_name):
    if model_name == "EfficientNet-B4":
        model = models.efficientnet_b4(weights='DEFAULT')
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    elif model_name == "ResNet-50":
        model = models.resnet50(weights='DEFAULT')
        model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif model_name == "ViT-B/16":
        model = models.vit_b_16(weights='DEFAULT')
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    elif model_name == "MobileNet-V2":
        model = models.mobilenet_v2(weights='DEFAULT')
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(DEVICE)

# ------------- Training Function -------------
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        total += images.size(0)
    avg_loss = running_loss / total
    avg_acc = running_correct / total
    return avg_loss, avg_acc

# ------------- Inference on Test Set -------------
def get_predictions(model, loader, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Test", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    n_samples = len(loader.dataset)
    avg_loss = total_loss / n_samples
    return np.array(all_labels), np.array(all_preds), avg_loss

# ------------- Benchmark and Save Results -------------
def benchmark_and_save():
    model_names = ["EfficientNet-B4", "ResNet-50", "ViT-B/16", "MobileNet-V2"]
    results = {}

    print("Training and evaluating all models on the same data split...")
    for name in model_names:
        print(f"\n=== {name} ===")
        train_loader, test_loader = get_data_loaders(name)
        model = load_model(name)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS}")
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")

        y_true, y_pred, test_loss = get_predictions(model, test_loader, criterion)
        acc = np.mean(y_true == y_pred)
        params = sum(p.numel() for p in model.parameters())
        results[name] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "Test Loss": test_loss,
            "Test Accuracy": acc,
            "# parameters": f"{params/1e6:.1f}M"
        }
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {acc:.2%}")

    # Unique correct/error computation
    correct_matrix = np.array([results[name]["y_pred"] == results[name]["y_true"] for name in model_names]).T
    table_data = []

    for i, name in enumerate(model_names):
        current_is_correct = correct_matrix[:, i]
        others_are_wrong = ~np.any(np.delete(correct_matrix, i, axis=1), axis=1)
        unique_correct = np.sum(current_is_correct & others_are_wrong)

        current_is_wrong = ~correct_matrix[:, i]
        others_are_correct = np.all(np.delete(correct_matrix, i, axis=1), axis=1)
        unique_error = np.sum(current_is_wrong & others_are_correct)

        row = {
            "Model Name": name,
            "# parameters": results[name]["# parameters"],
            "Test Loss": f"{results[name]['Test Loss']:.4f}",
            "Test Accuracy": f"{results[name]['Test Accuracy']:.2%}",
            "# unique correct samples": unique_correct,
            "# unique errors": unique_error
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)
    print("\nResults:\n")
    print(df)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved detailed results to: {RESULTS_CSV}")

if __name__ == "__main__":
    benchmark_and_save()
