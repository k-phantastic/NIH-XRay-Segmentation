import os
import glob
import csv
import time
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast # For Mixed Precision (Speed)
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CONFIGURATION
Config = {
    "CLUSTER_IMAGE_ROOT": "IMAGE_FOLDER", # CHANGE TO LOCAL DATA ROOT
    
    "CSV_FILE": "../data/Data_Entry_2017.csv",
    "SPLIT_FILE": "../data/train_val_list.txt",
    
    # Hyperparameters
    "IMG_SIZE": 224,
    "BATCH_SIZE": 16,          # High batch size for GPU
    "LEARNING_RATE": 1e-4,
    "EPOCHS": 20,
    "PATIENCE": 5,             # Early stopping
    "NUM_WORKERS": 0,          # Use 16 if GPU access is available
    
    # Outputs
    "LOG_FILE": "training_log.csv",
    "BEST_MODEL_FILE": "densenet_best.pth",
    "CHECKPOINT_FILE": "densenet_checkpoint.pth"
}

# Auto-detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available(): device = torch.device("mps")
print(f"🚀 Launching on device: {device}")


# 2. PATH MAPPING (CRITICAL FOR SPLIT FOLDERS)
def create_path_map(root_dir):
    """
    Scans images_001, images_002, etc. and creates a map:
    {'000001_000.png': '/path/to/images_001/images/000001_000.png'}
    """
    print(f"Scanning for images in {root_dir}...")
    # Look recursively for .png files in all subfolders
    image_paths = glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
    
    path_map = {os.path.basename(p): p for p in image_paths}
    print(f"✅ Found {len(path_map)} images across subfolders.")
    return path_map

# 3. DATASET CLASS
class NIHDataset(Dataset):
    def __init__(self, dataframe, path_map, transform=None, classes=None):
        self.df = dataframe
        self.path_map = path_map
        self.transform = transform
        self.classes = classes
        
        # Parse labels
        self.df['labels_list'] = self.df['Finding Labels'].apply(
            lambda x: [] if 'No Finding' in x else x.split('|')
        )
        
        # Create Multi-Hot Encoding
        from sklearn.preprocessing import MultiLabelBinarizer
        if self.classes is None:
            self.mlb = MultiLabelBinarizer()
            self.labels = self.mlb.fit_transform(self.df['labels_list'])
            self.classes = self.mlb.classes_
        else:
            self.mlb = MultiLabelBinarizer(classes=self.classes)
            self.labels = self.mlb.fit_transform(self.df['labels_list'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['Image Index']
        
        # USE THE MAP TO FIND THE FILE
        if img_name in self.path_map:
            img_path = self.path_map[img_name]
        else:
            # Fallback
            print(f"⚠️ Warning: {img_name} not found in path map.")
            return torch.zeros((3, Config["IMG_SIZE"], Config["IMG_SIZE"])), torch.tensor(self.labels[idx], dtype=torch.float32)

        # Image Load using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            return torch.zeros((3, Config["IMG_SIZE"], Config["IMG_SIZE"])), torch.tensor(self.labels[idx], dtype=torch.float32)

        # CLAHE
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(gray)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

# Transform: Letterbox Padding
class LetterboxPad:
    def __call__(self, image):
        w, h = image.size
        target = Config["IMG_SIZE"]
        scale = min(target / w, target / h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        new_img = Image.new("RGB", (target, target), (0, 0, 0))
        new_img.paste(image, ((target - new_w) // 2, (target - new_h) // 2))
        return new_img

# 4. TRAIN & VALIDATE ENGINES

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    steps = len(loader)
    
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision 
        if device.type == 'cuda':
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard float32 for CPU/MPS
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        
        if i % 100 == 0:
            print(f"\rStep [{i}/{steps}] Loss: {loss.item():.4f}", end="")
            
    return running_loss / steps

def validate(model, loader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Store probabilities
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    avg_loss = running_loss / len(loader)
    
    # Stack and Calc AUC
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    try:
        # Macro average treats all classes equally (good for imbalance)
        auc_score = roc_auc_score(all_labels, all_preds, average='macro')
    except ValueError:
        auc_score = 0.0
        print("Warning: AUC calc failed (likely missing class in batch).")

    metrics = {'Mean_AUC': auc_score}
    return avg_loss, metrics

# 5. MAIN EXECUTION
def main():
    # GENERATE PATH MAP
    path_map = create_path_map(Config["CLUSTER_IMAGE_ROOT"])
    if len(path_map) == 0:
        print("❌ Error: No images found. Check CLUSTER_IMAGE_ROOT path!")
        return

    # LOAD DATA
    full_df = pd.read_csv(Config["CSV_FILE"])
    
    # Filter using Train/Val List text file
    with open(Config["SPLIT_FILE"], 'r') as f:
        train_val_list = [line.strip() for line in f.readlines()]
    full_df = full_df[full_df['Image Index'].isin(train_val_list)]
    
    # Filter: Only keep images we actually found on disk
    full_df = full_df[full_df['Image Index'].isin(path_map.keys())]
    
    # Patient-Level Split
    patient_ids = full_df['Patient ID'].unique()
    # Fixed seed for reproducibility
    np.random.seed(42) 
    train_ids, val_ids = np.split(np.random.permutation(patient_ids), [int(.8*len(patient_ids))])
    
    train_df = full_df[full_df['Patient ID'].isin(train_ids)]
    val_df = full_df[full_df['Patient ID'].isin(val_ids)]
    
    print(f"Train Size: {len(train_df)} | Val Size: {len(val_df)}")

    # 3. SETUP DATA LOADERS
    train_tf = transforms.Compose([
        LetterboxPad(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_tf = transforms.Compose([
        LetterboxPad(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    nih_classes = sorted(['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
                          'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 
                          'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'])

    # Pass the PATH_MAP to the dataset
    train_ds = NIHDataset(train_df, path_map, transform=train_tf, classes=nih_classes)
    val_ds = NIHDataset(val_df, path_map, transform=val_tf, classes=nih_classes)

    train_loader = DataLoader(train_ds, batch_size=Config["BATCH_SIZE"], 
                              shuffle=True, num_workers=Config["NUM_WORKERS"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config["BATCH_SIZE"], 
                             shuffle=False, num_workers=Config["NUM_WORKERS"], pin_memory=True)

    # MODEL SETUP
    print("Building DenseNet-121...")
    model = models.densenet121(weights='DEFAULT')
    model.classifier = nn.Linear(model.classifier.in_features, len(nih_classes))
    model = model.to(device)

    # Weighted Loss
    pos_counts = train_ds.labels.sum(axis=0)
    # Prevent div by zero
    pos_weights = (len(train_df) - pos_counts) / (pos_counts + 1e-5)
    pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.AdamW(model.parameters(), lr=Config["LEARNING_RATE"])
    
    # Mixed Precision Scaler
    scaler = GradScaler()

    # TRAINING LOOP
    best_auc = 0.0
    patience_counter = 0

    # Initialize CSV
    if not os.path.exists(Config["LOG_FILE"]):
        with open(Config["LOG_FILE"], 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'mean_auc'])

    print("Starting Training...")
    
    try: 
        for epoch in range(Config["EPOCHS"]):
            print(f"\n[Epoch {epoch+1}/{Config['EPOCHS']}]")
            
            # Train & Validate
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
            val_loss, metrics = validate(model, val_loader, criterion, device, nih_classes)
            
            current_auc = metrics.get('Mean_AUC', 0)
            
            print(f"\nResults -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Mean AUC: {current_auc:.4f}")
            
            # Log to CSV
            with open(Config["LOG_FILE"], 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, train_loss, val_loss, current_auc])
            
            # Checkpointing (Best Model)
            if current_auc > best_auc:
                best_auc = current_auc
                patience_counter = 0
                torch.save(model.state_dict(), Config["BEST_MODEL_FILE"])
                print(f"🥇 New Best AUC! Saved to {Config['BEST_MODEL_FILE']}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{Config['PATIENCE']}")
            
            # Save Latest Backup
            torch.save(model.state_dict(), "densenet_latest_backup.pth")

            # Early Stopping
            if patience_counter >= Config["PATIENCE"]:
                print("Early Stopping Triggered.")
                break

    except KeyboardInterrupt:
        print("\n🛑 Manual Interruption (Ctrl+C) Detected!")
        print("Saving current model state before exiting...")
        torch.save(model.state_dict(), "densenet_EMERGENCY_SAVE.pth")
        print("Safe to close.")
            
    print(f"Done. Best AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()
    