
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold

# Dataset class (must be at module level for Windows multiprocessing)
class CassavaDataset(Dataset):
    def __init__(self, image_path, dataframe, transform=None):
        self.image_path = image_path
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_fp = os.path.join(self.image_path, row.iloc[0])
        image = Image.open(image_fp).convert("RGB")
        label = int(row.iloc[1])
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    # For reproducibility
    SEED = 70
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Device setup
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print("current device: ", torch.cuda.current_device())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # Paths (EDIT to your actual data locations)
    image_path = "Data/train_images"
    train_csv = "Data/train.csv"

    # Preview the CSV
    df = pd.read_csv(train_csv)
    print(df.head())

    # Transforms
    basic_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_transform = basic_transform
    val_test_transform = basic_transform

    # Split data: 90% train, 10% test (hold-out, never seen by model in k-fold)
    train_df, test_df = train_test_split(
        df, test_size=0.10, 
        stratify=df['label'] if 'label' in df.columns else df.iloc[:,1],
        random_state=SEED, shuffle=True
    )

    print(f"Train set for CV: {len(train_df)}, Hold-out test set: {len(test_df)}")

    # K-fold CV config
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    all_fold_val_acc = []
    all_fold_models = []
    num_classes = len(df['label'].unique()) if 'label' in df.columns else len(df.iloc[:,1].unique())

    # Model selection/initialization
    def get_resnet18(num_classes, feature_extract=False, use_pretrained=True):
        # To avoid deprecated warning, use weights argument if torchvision version >= 0.13
        from packaging import version
        if version.parse(torchvision.__version__) >= version.parse("0.13"):
            if use_pretrained:
                weights = torchvision.models.ResNet18_Weights.DEFAULT
            else:
                weights = None
            model = torchvision.models.resnet18(weights=weights)
        else:
            model = torchvision.models.resnet18(pretrained=use_pretrained)
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model

    def accuracy_and_confusion(model, dataloader, device, num_classes, criterion=None):
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        confmat = np.zeros((num_classes, num_classes), dtype=int)
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if criterion is not None:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                for i, l in enumerate(labels):
                    confmat[l.item(), preds[i].item()] += 1
        accuracy = 100. * correct / total if total > 0 else 0.
        avg_loss = total_loss / len(dataloader) if criterion is not None and len(dataloader) > 0 else None
        return accuracy, confmat, avg_loss

    def plot_confmat(confmat, classnames=None, save_path=None, title='Confusion Matrix'):
        plt.figure(figsize=(10, 8))
        if classnames is None:
            classnames = [str(i) for i in range(confmat.shape[0])]
        sns.heatmap(confmat, annot=True, fmt='d', cmap='Blues', yticklabels=classnames, xticklabels=classnames, 
                    cbar_kws={'label': 'Count'})
        plt.title(title, fontsize=18, pad=16)
        plt.ylabel('Actual Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        plt.show()

    # Training config
    lr = 0.0005
    batch_size = 128
    epochs = 30
    
    # Create directory for saving models and results
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    skf_labels = train_df['label'] if 'label' in train_df.columns else train_df.iloc[:,1]

    print("Starting k-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, skf_labels)):
        print(f"\nFold {fold+1}/{n_splits}...")
        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val = train_df.iloc[val_idx].reset_index(drop=True)
        
        train_ds = CassavaDataset(image_path, fold_train, train_transform)
        val_ds = CassavaDataset(image_path, fold_val, val_test_transform)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

        model = get_resnet18(num_classes=num_classes, feature_extract=False, use_pretrained=True).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', threshold=0.3
        )

        best_val_acc = 0.0
        best_model_state = None
        best_val_confmat = None

        for epoch in range(1, epochs + 1):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (preds == labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total if train_total > 0 else 0.
            
            # Validation phase
            val_acc, val_confmat, val_loss = accuracy_and_confusion(model, val_loader, device, num_classes, criterion)
            
            print(f"Fold {fold+1} | Epoch {epoch}: train_acc={train_acc:.2f}%, train_loss={train_loss:.4f}, "
                  f"val_acc={val_acc:.2f}%, val_loss={val_loss:.4f}")
            
            scheduler.step(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                best_val_confmat = val_confmat.copy()
        
        # Save fold model
        fold_model_path = f"models/best_model_fold_{fold+1}.pth"
        torch.save({
            'model_state_dict': best_model_state,
            'fold': fold + 1,
            'val_accuracy': best_val_acc,
            'num_classes': num_classes
        }, fold_model_path)
        print(f"Best model for fold {fold+1} saved to: {fold_model_path}")
        
        # Save validation confusion matrix for this fold
        plot_confmat(best_val_confmat, save_path=f"results/confusion_matrix_fold_{fold+1}.png", 
                    title=f'Confusion Matrix - Fold {fold+1} (Validation Set)')
        
        all_fold_val_acc.append(best_val_acc)
        all_fold_models.append(best_model_state)
        print(f"Best validation acc for fold {fold+1}: {best_val_acc:.2f}%")

    print("\nK-fold validation finished.")
    print(f"Average fold validation accuracy: {np.mean(all_fold_val_acc):.2f}%")

    # Best model over all folds
    best_fold_idx = int(np.argmax(all_fold_val_acc))
    print(f"Best fold: {best_fold_idx+1}, validation accuracy: {all_fold_val_acc[best_fold_idx]:.2f}%")
    best_model = get_resnet18(num_classes=num_classes, feature_extract=False, use_pretrained=True).to(device)
    best_model.load_state_dict(all_fold_models[best_fold_idx])
    
    # Save the best model
    best_model_path = "models/best_model_overall.pth"
    torch.save({
        'model_state_dict': all_fold_models[best_fold_idx],
        'best_fold': best_fold_idx + 1,
        'val_accuracy': all_fold_val_acc[best_fold_idx],
        'avg_val_accuracy': np.mean(all_fold_val_acc),
        'num_classes': num_classes,
        'all_fold_accuracies': all_fold_val_acc
    }, best_model_path)
    print(f"Best overall model saved to: {best_model_path}")

    # Evaluate on held-out test set (never seen during training)
    criterion = nn.CrossEntropyLoss()
    holdout_ds = CassavaDataset(image_path, test_df.reset_index(drop=True), val_test_transform)
    holdout_loader = DataLoader(holdout_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_acc, test_confmat, test_loss = accuracy_and_confusion(best_model, holdout_loader, device, num_classes, criterion)
    print(f"\nHold-out 10% test set - Accuracy: {test_acc:.2f}%, Loss: {test_loss:.4f}")
    
    # Save test confusion matrix
    plot_confmat(test_confmat, save_path="results/confusion_matrix_test_set.png", 
                title='Confusion Matrix - Hold-out Test Set')
    
    # Save summary results to text file
    summary_path = "results/training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("CASSAVA LEAF DISEASE CLASSIFICATION - RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Fold: {best_fold_idx + 1}\n")
        f.write(f"Best Validation Accuracy: {all_fold_val_acc[best_fold_idx]:.2f}%\n")
        f.write(f"Average Validation Accuracy: {np.mean(all_fold_val_acc):.2f}%\n")
        f.write(f"Test Set Accuracy: {test_acc:.2f}%\n")
        f.write(f"Test Set Loss: {test_loss:.4f}\n\n")
        f.write("Fold-wise Validation Accuracies:\n")
        for i, acc in enumerate(all_fold_val_acc):
            f.write(f"  Fold {i+1}: {acc:.2f}%\n")
    print(f"Training summary saved to: {summary_path}")

if __name__ == "__main__":
    main()

