"""
train_model.py - Main training loop for NIH Chest X-Ray classification using ResNet50.

Includes entire training process, including:
- Loading the dataset and creating DataLoaders
- Initializing the ResNet50 model and optimizer
- Running the training loop with validation and checkpointing
- Computing and printing metrics after each epoch
The configuration for paths, hyperparameters, and toggles is centralized in config.py.
"""
import cv2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.amp import autocast, GradScaler

from tqdm import tqdm
from PIL import Image

# Local imports
from data_loader import NIH8Dataset, train_data_transforms, val_data_transforms
from utils import (
    calculate_class_weights,
    get_train_val_test_split, 
    get_device, 
    save_checkpoint,
    load_checkpoint,
    calculate_metrics,
    print_metrics, 
    EarlyStopping
)
from config import *

######################################################################
# Training and Validation
######################################################################

def train_one_epoch(epoch, model, loader, criterion, optimizer, scaler, device):
    """
    Runs one full training epoch and returns the mean loss.

    Arguments:
    - epoch:     0-indexed epoch number (for the progress bar label).
    - model:     ResNet50 in train mode.
    - loader:    Training DataLoader.
    - criterion: BCEWithLogitsLoss with pos_weight.
    - optimizer: AdamW optimizer
    - scaler:    GradScaler instance (or None if AMP disabled).
    - device:    Compute device.

    Returns:
        Mean BCE loss over all images in the epoch.
    """
    model.train()
    running_loss = 0.0
    n_samples    = 0
    use_cuda     = (device.type == "cuda")
    use_amp      = bool(USE_AMP and use_cuda and scaler is not None)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)

    for batch in pbar:
        if batch is None:
            continue  # Dropped by safe_collate (missing image file)

        # non_blocking=True overlaps H2D transfer with CPU preprocessing.
        x = batch["image"].to(device,  non_blocking=use_cuda)
        y = batch["labels"].to(device, non_blocking=use_cuda)

        # Early NaN guard — catches corrupted images before they poison weights.
        if not torch.isfinite(x).all():
            raise ValueError("NaN / Inf detected in input batch.")
        if not torch.isfinite(y).all():
            raise ValueError("NaN / Inf detected in label batch.")

        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss   = criterion(logits, y)

            if not torch.isfinite(loss):
                raise ValueError(f"NaN/Inf loss under AMP: {loss.item():.4f}")

            scaler.scale(loss).backward()

            if GRAD_CLIP:
                # Unscale before clipping so clip threshold is in real units.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(x)
            loss   = criterion(logits, y)

            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss: {loss.item():.4f}")
            
            loss.backward()

            if GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

        # Update running loss and sample count for mean loss calculation.
        batch_size    = x.size(0)
        running_loss += loss.item() * batch_size
        n_samples    += batch_size

        pbar.set_postfix(loss=f"{running_loss / max(n_samples, 1):.4f}")

    return running_loss / max(n_samples, 1)


@torch.no_grad()
def validate(epoch, model, loader, criterion, device, num_classes):
    """
    Runs inference on the validation set and returns loss + metrics.

    Collects all predictions in memory before computing metrics so that AUC
    is computed globally (not batch-by-batch, which would be inaccurate).

    Arguments:
    - epoch:       0-indexed epoch number.
    - model:       ResNet50 in eval mode.
    - loader:      Validation DataLoader.
    - criterion:   BCEWithLogitsLoss (same as training, for loss comparison).
    - device:      Compute device.
    - num_classes: Number of output classes (15 for NIH).

    Returns:
        (val_loss, metrics): Mean validation loss and the metrics dict from
                             calculate_metrics().
    """
    model.eval()
    running_loss = 0.0
    n_samples    = 0
    all_labels   = []
    all_probs    = []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val  ]", leave=False)

    for batch in pbar:
        if batch is None:
            continue

        x = batch["image"].to(device,  non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)

        logits = model(x)
        loss   = criterion(logits, y)
        probs  = torch.sigmoid(logits)  # Convert logits → probabilities
        
        batch_size    = x.size(0)
        running_loss += loss.item() * batch_size
        n_samples    += batch_size

        all_labels.append(y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    # Stack batches into single arrays for global metric computation.
    all_labels = (np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0, num_classes)))
    all_probs  = (np.concatenate(all_probs,  axis=0) if all_probs else np.zeros((0, num_classes)))

    metrics = calculate_metrics(all_labels, all_probs, threshold=0.5)
    return running_loss / max(n_samples, 1), metrics

######################################################################
# Main Setup 
######################################################################

def main():
    ######################################################################
    # 1. Device Setup and Configuration Confirmation
    ######################################################################
    start_time = time.time()

    print("\n" + "="*60)
    print(" 1. DEVICE AND PATH CONFIRMATION")
    print("="*60)

    device = get_device()
    print(f"  Device:        {device}")
    print(f"  CSV Path:      {CSV_PATH}")
    print(f"  Image Folders: {IMAGE_FOLDERS}")

    ######################################################################
    # 2. Dataset/DataLoader Setup
    ######################################################################
    print("\n" + "="*60)
    print(" 2. DATASET SETUP AND SPLIT")
    print("="*60)
    
    df = pd.read_csv(CSV_PATH)
    print(f"    Total records in CSV: {len(df)}")

    # Get train/val/test split (by patient)
    train_df, val_df, test_df = get_train_val_test_split(df, TRAIN_SPLIT, VAL_SPLIT)

    # Image Folders for dataset
    folders = [str(IMAGE_FOLDERS[0] / f"images_{str(i).zfill(3)}") for i in range(1, 13)]

    print("\n  Building datasets...")
    train_ds = NIH8Dataset(train_df, folders, transform=train_data_transforms, cache_path=CACHE_PATH)
    val_ds   = NIH8Dataset(val_df,   folders, transform=val_data_transforms, cache_path=CACHE_PATH)
    test_ds  = NIH8Dataset(test_df,  folders, transform=val_data_transforms, cache_path=CACHE_PATH)

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=PERSISTENT_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=PERSISTENT_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=PERSISTENT_WORKERS, pin_memory=True)

    num_classes = len(train_ds.classes)

    print(f"\n  Dataset Initialization Complete!")
    print(f"        Classes: {num_classes}")
    print(f"        Train batches: {len(train_loader)}")
    print(f"        Val batches: {len(val_loader)}")
    print(f"        Test batches: {len(test_loader)}")

    ######################################################################
    # 3. Model Setup (Includes Weights, Loss, and Optimizer Setup)
    ######################################################################
    print("\n" + "="*60)
    print(" 3. MODEL, LOSS, AND OPTIMIZER")
    print("="*60)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Quick sanity check before training starts.
    with torch.no_grad():
        dummy = torch.randn(2, 3, 256, 256).to(device)
        out   = model(dummy)
        assert out.shape == torch.Size([2, num_classes]), \
            f"  Unexpected output shape: {out.shape}"
    print(f"    Sanity check passed: (2, 3, 256, 256) → {out.shape}")

    weights, weight_dict = calculate_class_weights(train_ds, max_weight=MAX_CLASS_WEIGHT)
    weights = weights.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=weights) # Use pos_weight to handle class imbalance in multi-label classification
    optimizer = torch.optim.AdamW(
        [
            {'params': model.fc.parameters(), 'lr': LEARNING_RATE},
            {'params': model.layer4.parameters(), 'lr': LR_LAYER4}, # Fine-tuning only the last block and the fully connected layer
        ], 
        weight_decay=WEIGHT_DECAY # For preventing overfitting 
    ) 

    scaler = GradScaler("cuda") if USE_AMP and device.type == "cuda" else None
    if scaler:
        print("\n   Automatic Mixed Precision (AMP) enabled")

    ######################################################################
    # 4. Training Loop
    ######################################################################
    print("\n" + "="*60)
    print(f" 4. TRAINING  ({EPOCHS} epochs | patience={PATIENCE} | delta={PATIENCE_DELTA})")
    print("="*60)

    # Initialize history dict to store metrics for plotting
    history = {
        "train_loss": [],
        "val_loss": [],
        "mean_auc": []
    }

    best_auc = -1.0

    # Early stopping / Patience 
    early_stopping = EarlyStopping(patience = PATIENCE, delta = PATIENCE_DELTA, verbose=True)

    # Main loop
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(epoch, model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_metrics = validate(epoch, model, val_loader, criterion, device, num_classes)
        
        mean_auc = val_metrics['auc_macro'] if val_metrics['auc_macro'] is not None else 0.0
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["mean_auc"].append(mean_auc)

        # Print on first, and then every 5 epochs, and also on the last epoch.
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            print(f"\nEpoch {epoch+1}/{EPOCHS} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Mean AUC:   {mean_auc:.4f}")
            print(f"  Valid Classes: {val_metrics['num_valid_classes']}/{num_classes}")
            print("-" * 50)

    
        # Checkpointing based on validation AUC
        if mean_auc > best_auc:
            best_auc = mean_auc
            save_checkpoint({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "best_auc": best_auc,
                "history": history
            }, filename=MODEL_SAVE_PATH)
            print(f"    New best model saved with AUC: {best_auc:.4f}")

        # Early stopping check
        early_stopping.check_early_stop(val_loss)
        if early_stopping.stop_training:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break


    print("\n" + "="*60)
    print(f"  Training complete.  Best val AUC: {best_auc:.4f}")
    print(f"  Checkpoint: {MODEL_SAVE_PATH}")
    print("="*60)

    ######################################################################
    # 5. Model Testing / Evaluation 
    ######################################################################
    print("\n" + "="*60)
    print(" 5. FINAL TEST EVALUATION")
    print("="*60)
    # Load the best model checkpoint before testing
    checkpoint = torch.load(MODEL_SAVE_PATH, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"    Loaded epoch {checkpoint['epoch']+1} "
          f"    (val AUC: {checkpoint['best_auc']:.4f})")
    
    test_loss, test_metrics = validate(
        epoch=0,
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes
    )

    print(f"\n  Test Loss: {test_loss:.4f}")
    print(f"  Test AUC:  {test_metrics['auc_macro']:.4f}")
    print_metrics(test_metrics, train_ds.classes)

    # Final runtime calculation
    elapsed = time.time() - start_time
    hours   = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"\n  Total runtime: {hours}h {minutes}m {seconds}s")



if __name__ == "__main__":
    main()