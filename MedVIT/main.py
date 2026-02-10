import torch
from torch import amp
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import datetime
import csv # Added for logging

# Internal Imports
from data_loader import get_loaders
from model import OptimizedMedViT
from train import train_one_epoch, validate
from dataset import calculate_class_weights

def main():
    # --- 0. EXPERIMENT TRACKING ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"run_{timestamp}_full_data"
    
    # Create the directory
    exp_dir = f"experiments/{run_name}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Dynamic paths
    CHECKPOINT_FILE = os.path.join(exp_dir, "checkpoint.pth")
    BEST_MODEL_FILE = os.path.join(exp_dir, "best_medvit_model.pth")
    LOG_FILE = os.path.join(exp_dir, "results.csv")

    # --- 1. CONFIGURATION ---
    CSV_PATH = '../data/Data_Entry_2017.csv'
    IMAGE_ROOT = '../images'
    IMAGE_FOLDERS = [os.path.join(IMAGE_ROOT, f"images_{str(i).zfill(3)}") for i in range(1, 13)]
    
    # (Removed the hardcoded path overrides that were here)
    
    BATCH_SIZE = 32 
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    SUBSET_SIZE = None
    train_p = 0.70
    val_p = 0.15
    test_p = 0.15
    num_workers = 2

    PATIENCE = 3        
    patience_counter = 0 

    # --- 2. QUICK PATH CHECK ---
    print(f"Starting Experiment: {run_name}")
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {os.path.abspath(CSV_PATH)}")
        return
    
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
    pos_weights = calculate_class_weights(train_loader.dataset)
    
    model = OptimizedMedViT(num_classes=len(class_names)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    # In main.py
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5}, # Slow updates for pre-trained parts
        {'params': model.meta_encoder.parameters(), 'lr': 1e-4}, # Fast updates for new parts
        {'params': model.gating_unit.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-5)
    scaler = amp.GradScaler('cuda') if device.type == 'cuda' else None

    # --- 5. RESUME LOGIC ---
    start_epoch = 0
    best_auc = 0.0

    if os.path.exists(CHECKPOINT_FILE):
        print(f"Found checkpoint! Resuming...")
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_auc = checkpoint.get('auc', 0.0)

    # --- 6. TRAINING LOOP ---
    # Initialize CSV Log
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'mean_auc'])

    print(f"Starting Training on {len(class_names)} classes...")
    
    for epoch in range(start_epoch, EPOCHS):
        print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, metrics = validate(model, val_loader, criterion, device, class_names)
        
        if metrics:
            current_auc = metrics['Mean_AUC']
            print(f"Summary -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Mean AUC: {current_auc:.4f}")
            
            # Log results to CSV
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, train_loss, val_loss, current_auc])

            # Improvement logic
            if current_auc > best_auc:
                best_auc = current_auc
                patience_counter = 0  
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'class_names': class_names,
                    'auc': best_auc,
                    'config': {
                        'lr': LEARNING_RATE,
                        'batch_size': BATCH_SIZE,
                        'subset': SUBSET_SIZE,
                        'split': (train_p, val_p, test_p)
                    }
                }, BEST_MODEL_FILE)
                print(f"🥇 New Best Model Saved to {BEST_MODEL_FILE}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print(f"\n[EARLY STOPPING] Triggered at epoch {epoch+1}")
                break

        # Save resume checkpoint
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'auc': best_auc,
        }, CHECKPOINT_FILE)

    print("\nTraining Complete. Best Mean AUC reached:", round(best_auc, 4))

if __name__ == "__main__":
    main()