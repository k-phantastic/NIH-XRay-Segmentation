import torch
from torch import amp
from tqdm import tqdm
import numpy as np
from dataset import calculate_metrics 

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        meta = batch['meta'].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type=device.type):
            outputs = model(images, meta)
            loss = criterion(outputs, labels)

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
    model.eval()
    val_loss = 0
    all_labels, all_preds = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            meta = batch['meta'].to(device, non_blocking=True)
            
            with amp.autocast(device_type=device.type):
                outputs = model(images, meta)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            all_labels.append(labels.cpu().numpy())
            all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
            
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    
    metrics = calculate_metrics(all_labels, all_preds, class_names)
    if metrics is None:
        metrics = {'Mean_AUC': 0.5}
        
    return val_loss / len(loader), metrics