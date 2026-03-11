import torch
from torch import amp
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import datetime
import csv
from torch.optim.lr_scheduler import OneCycleLR

# Internal Imports
from data_loader import get_loaders
from model import OptimizedMedViT
from train import train_one_epoch, validate
from dataset import calculate_class_weights

def main():
    CACHE_FILE = "path_cache.json"
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print("🚀 Fresh start: path_cache.json deleted. Re-scanning image directories...")

    # --- 0. EXPERIMENT TRACKING ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"run_{timestamp}_MedViT_SOTA"
    exp_dir = f"experiments/{run_name}"
    os.makedirs(exp_dir, exist_ok=True)
    
    BEST_MODEL_FILE = os.path.join(exp_dir, "best_medvit_model.pth")
    LOG_FILE = os.path.join(exp_dir, "results.csv")

    # --- 1. HYPERPARAMETERS & CONFIG ---
    CSV_PATH = '../data/Data_Entry_2017.csv'
    IMAGE_ROOT = '../images'
    # Added the extra "images" at the end to match your example path
    IMAGE_FOLDERS = [os.path.join(IMAGE_ROOT, f"images_{str(i).zfill(3)}", "images") for i in range(1, 13)]
    
    # Progressive Resizing Schedule
    # Phase 1: Standard (224), Phase 2: High-Res (320)
    PHASES = [
        {'res': 224, 'epochs': 10, 'bs': 32, 'acc_steps': 1},
        {'res': 448, 'epochs': 10,  'bs': 4,  'acc_steps': 8} # Acc steps keeps effective batch at 32
    ]
    
    MAX_LR = 1e-4
    WEIGHT_DECAY = 1e-4 # Increased for better regularization
    PATIENCE = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. DATA LOAD & PREP ---
    df = pd.read_csv(CSV_PATH)
    
    # Initial Loader for weights calculation
    temp_loader, _, _, class_names = get_loaders(df, IMAGE_FOLDERS, batch_size=32)
    print(f"Classes found: {class_names}")
    pos_weights = calculate_class_weights(temp_loader.dataset).to(device)
    
    # --- 3. MODEL INITIALIZATION ---
    model = OptimizedMedViT(num_classes=len(class_names)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    # Differential Learning Rates (Backbone learns slower to preserve ImageNet features)
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': MAX_LR / 10}, 
        {'params': model.meta_encoder.parameters(), 'lr': MAX_LR},
        {'params': model.gating_unit.parameters(), 'lr': MAX_LR},
        {'params': model.classifier.parameters(), 'lr': MAX_LR}
    ], weight_decay=WEIGHT_DECAY)

    scaler = amp.GradScaler('cuda') if device.type == 'cuda' else None

    # --- 4. TRAINING LOGIC ---
    best_auc = 0.0
    patience_counter = 0
    global_epoch = 0

    # Initialize CSV Log
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'res', 'train_loss', 'val_loss', 'mean_auc'])

    for phase_idx, phase in enumerate(PHASES):
        res = phase['res']
        current_phase_epochs = phase['epochs']
        batch_size = phase['bs']
        acc_steps = phase['acc_steps']

        print(f"\n{'='*20}\nENTERING PHASE {phase_idx+1}: {res}x{res}\n{'='*20}")

        # Update loaders for new resolution
        train_loader, val_loader, _, _ = get_loaders(
            df, IMAGE_FOLDERS, 
            batch_size=batch_size, 
            target_size=(res, res) # Ensure get_loaders accepts target_size
        )

        # Scheduler: Warmup for 10% of total phase steps, then Cosine Decay
        total_steps = current_phase_epochs * len(train_loader)
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=[MAX_LR/10, MAX_LR, MAX_LR, MAX_LR], # Matches param_groups
            total_steps=total_steps,
            pct_start=0.1, 
            anneal_strategy='cos'
        )

        for epoch in range(current_phase_epochs):
            global_epoch += 1
            print(f"\n[Epoch {global_epoch}] Res: {res} | BS: {batch_size} | AccSteps: {acc_steps}")
            
            # Pass acc_steps to train_one_epoch for gradient accumulation
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler, 
                scheduler=scheduler, acc_steps=acc_steps
            )
            
            val_loss, metrics = validate(model, val_loader, criterion, device, class_names)
            current_auc = metrics['Mean_AUC']
            
            print(f"Result -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | AUC: {current_auc:.4f}")

            # Save Results
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([global_epoch, res, train_loss, val_loss, current_auc])

            # Save Best Model
            if current_auc > best_auc:
                best_auc = current_auc
                patience_counter = 0
                # In main.py, change the save line to:
                torch.save({
                    'state_dict': model.state_dict(),
                    'class_names': class_names,
                    'auc': best_auc,
                    'epoch': global_epoch
                }, BEST_MODEL_FILE)
                print(f"🥇 New Best AUC! Model saved.")
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break
        
        if patience_counter >= PATIENCE: break

    print(f"\nTraining Finished. Best AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()