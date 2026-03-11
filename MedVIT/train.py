import torch
from torch import amp
from tqdm import tqdm
import numpy as np
from torchmetrics.classification import MultilabelAUROC

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, scheduler=None, acc_steps=1):
    model.train()
    total_loss = 0
    
    # set_to_none=True is slightly faster than zero_grad()
    optimizer.zero_grad(set_to_none=True)

    # Use tqdm for a nice progress bar
    pbar = tqdm(loader, desc="Training")
    
    for i, batch in enumerate(pbar):
        # Move data to GPU
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        meta = batch['meta'].to(device, non_blocking=True)

        # 1. Forward pass with Mixed Precision
        with amp.autocast(device_type='cuda'):
            outputs = model(images, meta)
            # We divide by acc_steps so the gradients are averaged correctly
            loss = criterion(outputs, labels) / acc_steps 

        # 2. Backward pass (scaled for mixed precision)
        scaler.scale(loss).backward()

        # 3. Update weights only every 'acc_steps' batches
        if (i + 1) % acc_steps == 0:
            # Unscales gradients and calls optimizer.step()
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()
            # Clear gradients for the next accumulation cycle
            optimizer.zero_grad(set_to_none=True)
        
        # 4. Scheduler step
        # IMPORTANT: Most schedulers (like OneCycleLR) expect a step per BATCH
        if scheduler is not None:
            scheduler.step()

        # Logging: multiply back by acc_steps to show the actual loss value
        total_loss += loss.item() * acc_steps
        
        # Update progress bar with current loss
        if i % 10 == 0:
            pbar.set_postfix({'loss': f"{loss.item() * acc_steps:.4f}"})
        
    return total_loss / len(loader)

def validate(model, loader, criterion, device, class_names):
    model.eval()
    val_loss = 0
    
    # VECTORIZED METRIC: Lives on the GPU
    metric = MultilabelAUROC(num_labels=len(class_names), average="macro").to(device)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            meta = batch['meta'].to(device, non_blocking=True)
            
            with amp.autocast(device_type='cuda'):
                outputs = model(images, meta)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            
            # Update the metric on GPU (vectorized)
            # We use torch.sigmoid because AUROC expects probabilities or logits
            metric.update(outputs, labels.long())
            
    # Compute the final Mean AUC score
    mean_auc = metric.compute().item()
    
    # Return in a format main.py expects
    return val_loss / len(loader), {'Mean_AUC': mean_auc}