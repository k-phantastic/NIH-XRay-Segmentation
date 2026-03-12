"""
tuning.py - Hyperparameter tuning for NIH Chest X-Ray ResNet50

Uses Optuna to find the best hyperparameters

After tuning, the best parameters are printed and saved to best_params.json.
You can then copy them into config.py for your full training run.

Typical workflow:
    1. Run tuning.py  
    2. Check best_params.json
    3. Copy best values into config.py
    4. Run train_model.py for the full 30-epoch run
"""
import cv2

import pandas as pd
import numpy as np
import json

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.amp import GradScaler
from torchvision import models

from tqdm import tqdm

import optuna
from optuna.samplers import TPESampler

from data_loader import NIH8Dataset, train_data_transforms, val_data_transforms
from utils import (
    calculate_class_weights,
    get_train_val_split,
    get_device,
    calculate_metrics,
)
from config import *

######################################################################
# Tuning Settings
######################################################################

N_TRIALS       = 20   # Number of hyperparameter combinations to try

PROXY_EPOCHS   = 5    # Epochs per trial

STUDY_NAME     = "nih_resnet50_tuning"
RESULTS_PATH   = "best_params.json"


######################################################################
# Search Space
# These are the hyperparameters Optuna will search over.
# Adjust the ranges based on what you want to explore.
######################################################################

def sample_hyperparams(trial):
    """
    Dictionary of the hyperparameter search space for Optuna
    """
    return {
        # Learning rate for the classifier head — log scale so it
        # samples evenly across orders of magnitude (1e-5 to 1e-3).
        "lr_head": trial.suggest_float("lr_head", 1e-5, 1e-3, log=True),

        # Learning rate for layer4 — always kept lower than head lr
        # since it's a pretrained block we want to adjust gently.
        "lr_layer4": trial.suggest_float("lr_layer4", 1e-6, 1e-4, log=True),

        # Weight Decay - higher values reduce overfitting but can hurt convergence if too large.
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),

        # Cap on class weights — prevents rare classes from dominating.
        "max_class_weight": trial.suggest_float("max_class_weight", 3.0, 20.0),
    }


######################################################################
# Single trial
######################################################################

def run_trial(trial, train_ds, val_ds, device):
    """
    Trains the model for PROXY_EPOCHS with the sampled hyperparameters
    and returns the best val AUC achieved

    Arguments:
    - trial:    Optuna trial object (provides sampled hyperparameters)
    - train_ds: Training dataset
    - val_ds:   Validation dataset
    - device:   Compute device (GPU)

    Returns:
        Best val AUC (float) across the proxy epochs.
    """
    params = sample_hyperparams(trial)
    num_classes = len(train_ds.classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=True,
    )

    # Model 
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Optimizer param groups 
    param_groups = [
        {"params": model.fc.parameters(),    "lr": params["lr_head"]},
        {"params": model.layer4.parameters(), "lr": params["lr_layer4"]},
    ]

    # Loss and optimizer
    weights, _ = calculate_class_weights(train_ds, max_weight=params["max_class_weight"])
    weights    = weights.to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=params["weight_decay"])

    # AMP 
    scaler  = GradScaler("cuda") if USE_AMP and device.type == "cuda" else None
    use_amp = bool(USE_AMP and device.type == "cuda" and scaler is not None)

    # Proxy training loop --
    best_auc = -1.0

    for epoch in range(PROXY_EPOCHS):

        # Training step
        model.train()
        for batch in tqdm(train_loader, desc=f"Trial {trial.number+1} Epoch {epoch+1}/{PROXY_EPOCHS} [Train]", leave=False):
            if batch is None:
                continue

            x = batch["image"].to(device, non_blocking=(device.type == "cuda"))
            y = batch["labels"].to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss   = criterion(logits, y)
                scaler.scale(loss).backward()
                if GRAD_CLIP:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss   = criterion(logits, y)
                loss.backward()
                if GRAD_CLIP:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

        # Validation step
        model.eval()
        all_labels   = []
        all_probs    = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Trial {trial.number+1} Epoch {epoch+1}/{PROXY_EPOCHS} [Val]", leave=False):
                if batch is None:
                    continue
                x = batch["image"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)

                probs = torch.sigmoid(model(x))
                all_labels.append(y.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        all_labels = np.concatenate(all_labels, axis=0)
        all_probs  = np.concatenate(all_probs,  axis=0)
        metrics    = calculate_metrics(all_labels, all_probs, threshold=0.5)
        val_auc    = metrics["auc_macro"]

        print(f"  Trial {trial.number+1} | Epoch {epoch+1}/{PROXY_EPOCHS} | AUC: {val_auc:.4f}")

        best_auc = max(best_auc, val_auc)

        # Optuna pruning - stops unpromising trials early based on intermediate results, saving even more compute.
        trial.report(val_auc, epoch)
        if trial.should_prune():
            print(f"  Trial {trial.number+1} pruned at epoch {epoch+1}.")
            raise optuna.exceptions.TrialPruned()

    return best_auc


######################################################################
# Main
######################################################################

def main():
    device = get_device()
    print(f"\n[Optuna] Device: {device}")
    print(f"[Optuna] Trials: {N_TRIALS} | Proxy epochs per trial: {PROXY_EPOCHS}")

    # Load data and build datasets once — reused across all trials
    # to avoid re-indexing the image cache every trial.
    print("\n[Optuna] Loading data...")
    df = pd.read_csv(CSV_PATH)
    train_df, val_df = get_train_val_split(df, TRAIN_SPLIT)

    # Image Folders for dataset
    folders = [str(IMAGE_FOLDERS[0] / f"images_{str(i).zfill(3)}") for i in range(1, 13)]

    train_ds = NIH8Dataset(train_df, folders, transform=train_data_transforms, cache_path=CACHE_PATH)
    val_ds   = NIH8Dataset(val_df,   folders, transform=val_data_transforms,   cache_path=CACHE_PATH)

    # Create Optuna study — direction="maximize" because we want higher AUC.
    # MedianPruner stops a trial early if its intermediate AUC is consistently
    # below the median of completed trials at the same epoch.
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=TPESampler(seed=42),        # TPE: learns from past trials
        pruner=optuna.pruners.MedianPruner( # Kills bad trials early
            n_startup_trials=5,             # Don't prune until 5 trials complete
            n_warmup_steps=2,               # Don't prune before epoch 2
        ),
    )

    def objective(trial):
        return run_trial(trial, train_ds, val_ds, device)

    print(f"\n[Optuna] Starting {N_TRIALS} trials...")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # ----------------------------------------------------------------
    # Results
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("OPTUNA TUNING COMPLETE")
    print("="*60)
    print(f"  Best trial:  #{study.best_trial.number + 1}")
    print(f"  Best AUC:    {study.best_value:.4f}")
    print(f"\n  Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Save best params to JSON so you can refer back to them
    with open(RESULTS_PATH, "w") as f:
        json.dump({"best_auc": study.best_value, "best_params": study.best_params}, f, indent=2)
    print(f"\n  Saved to {RESULTS_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()