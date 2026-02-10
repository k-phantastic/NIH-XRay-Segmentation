import torch
from torch import amp
from tqdm import tqdm
import numpy as np
from dataset import calculate_metrics 

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    """
    Trains for one epoch using Automatic Mixed Precision (AMP).
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        # Unpack the dictionary from NIH8Dataset
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        meta = batch['meta'].to(device, non_blocking=True)
        
        # Zero gradients efficiently
        optimizer.zero_grad(set_to_none=True)
        
        # Enable Mixed Precision
        with amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(images, meta)
            loss = criterion(outputs, labels)

        # Scaler handles FP16 precision to save VRAM and increase speed on 3070
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def validate(model, loader, criterion, device, class_names):
    """
    Evaluates the model and returns loss + per-class AUC metrics.
    """
    model.eval()
    val_loss = 0
    all_labels, all_preds = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            meta = batch['meta'].to(device, non_blocking=True)
            
            with amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(images, meta)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            
            # Store labels and predictions for AUC calculation
            all_labels.append(labels.cpu().numpy())
            # Use Sigmoid here because the loss function (BCEWithLogits) uses raw logits
            all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
            
    # Combine all batches
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, class_names)
    
    return val_loss / len(loader), metrics