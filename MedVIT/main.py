import torch
from torch import amp
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os

# Internal Imports
from data_loader import get_loaders
from model import OptimizedMedViT
from train import train_one_epoch, validate
from dataset import calculate_class_weights

def main():
    # --- 1. CONFIGURATION ---
    CSV_PATH = '../data/Data_Entry_2017.csv'
    IMAGE_ROOT = '../images'
    IMAGE_FOLDERS = [os.path.join(IMAGE_ROOT, f"images_{str(i).zfill(3)}") for i in range(1, 13)]
    
    CHECKPOINT_FILE = "checkpoint.pth"
    BEST_MODEL_FILE = "best_medvit_model.pth"
    
    BATCH_SIZE = 32 
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    SUBSET_SIZE = None
    train_p = 0.70
    val_p = 0.15
    test_p = 0.15
    num_workers = 2

    # --- 2. QUICK PATH CHECK ---
    print("Running path validation...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {os.path.abspath(CSV_PATH)}")
        return
    
    existing_folders = [f for f in IMAGE_FOLDERS if os.path.exists(f)]
    if not existing_folders:
        print(f"Error: No image folders found in {os.path.abspath(IMAGE_ROOT)}")
        return
    print(f"Found {len(existing_folders)}/12 image folders.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    # --- 3. DATA PREPARATION ---
    df = pd.read_csv(CSV_PATH)
    train_loader, val_loader, test_loader, class_names = get_loaders(
        df, 
        IMAGE_FOLDERS, 
        batch_size=BATCH_SIZE, 
        subset_size=SUBSET_SIZE,
        train_p=train_p, 
        val_p=val_p, 
        test_p=test_p,
        num_workers=num_workers
    )

    # --- 4. MODEL & OPTIMIZER INITIALIZATION ---
    print("Calculating class weights...")
    pos_weights, _ = calculate_class_weights(train_loader.dataset)
    
    model = OptimizedMedViT(num_classes=len(class_names)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Scale only if using CUDA
    scaler = amp.GradScaler('cuda') if device.type == 'cuda' else None

    # --- 5. RESUME LOGIC ---
    start_epoch = 0
    best_auc = 0.0

    if os.path.exists(CHECKPOINT_FILE):
        print(f"Found checkpoint! Resuming from {CHECKPOINT_FILE}...")
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_auc = checkpoint.get('auc', 0.0)
        print(f"Resuming at Epoch {start_epoch + 1} with Best AUC: {best_auc:.4f}")

    # --- 6. TRAINING LOOP ---
    print(f"Starting Training on {len(class_names)} classes...")
    
    for epoch in range(start_epoch, EPOCHS):
        print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, metrics = validate(model, val_loader, criterion, device, class_names)
        
        if metrics:
            current_auc = metrics['Mean_AUC']
            print(f"Summary -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Mean AUC: {current_auc:.4f}")
            
            # Save the Best Model
            if current_auc > best_auc:
                best_auc = current_auc
                torch.save({
                    'state_dict': model.state_dict(),
                    'class_names': class_names,
                    'auc': best_auc,
                }, BEST_MODEL_FILE)
                print(f"New Best Model Saved (AUC: {best_auc:.4f})")

        # ALWAYS save a checkpoint to resume later
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'auc': best_auc,
        }, CHECKPOINT_FILE)
        print(f"Progress saved to {CHECKPOINT_FILE}")

    print("\nTraining Complete. Best Mean AUC reached:", round(best_auc, 4))

if __name__ == "__main__":
    main()