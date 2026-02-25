import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import torch.nn.functional as F
import cv2


def calculate_class_weights(dataset, max_weight=10.0):
    """
    Calculates weights to handle class imbalance.
    Standard formula for Multi-label: ratio of negatives to positives.
    """
    # Get the binary matrix (N images x 15 classes)
    all_labels = dataset.label_matrix 
    
    # Count positive occurrences (ones) for each class
    pos_counts = np.sum(all_labels, axis=0)
    total_samples = all_labels.shape[0]
    
    # Count negative occurrences (zeros)
    neg_counts = total_samples - pos_counts
    
    # Calculate pos_weight (Negatives / Positives)
    # This is the standard input for torch.nn.BCEWithLogitsLoss
    # We add 1e-6 to avoid division by zero
    class_weights = neg_counts / (pos_counts + 1e-6)
    
    # Cap weights to prevent explosion
    class_weights = np.minimum(class_weights, max_weight)
    
    # Map to names for the printout
    weight_dict = {dataset.classes[i]: round(class_weights[i], 4) for i in range(len(dataset.classes))}
    
    return torch.tensor(class_weights, dtype=torch.float32), weight_dict

def get_train_val_split(df, train_size=0.8, random_state=42):
    """
    Splits the dataframe based on Patient ID to prevent data leakage.
    """
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    
    # This ensures that all images of the same patient stay together
    train_idx, val_idx = next(gss.split(df, groups=df['Patient ID']))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    print(f"Split complete: {len(train_df)} train images, {len(val_df)} val images.")
    return train_df, val_df

def visualize_results(batch_dict, classes, n=4):
    """
    Quickly plots a batch of images with their multi-hot labels.
    """
    images = batch_dict['image']
    labels = batch_dict['labels']
    
    plt.figure(figsize=(16, 4))
    for i in range(min(n, len(images))):
        plt.subplot(1, n, i+1)
        
        # Un-normalize for viewing
        img = images[i].squeeze().cpu().numpy()
        plt.imshow(img, cmap='gray')
        
        # Get names of active labels
        active_labels = [classes[j] for j, val in enumerate(labels[i]) if val == 1]
        plt.title("\n".join(active_labels), fontsize=8)
        plt.axis('off')
    plt.show()

def get_device():
    """Returns the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Saves the current training state."""
    print(f"=> Saving checkpoint: {filename} (Epoch {state['epoch']})")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer):
    """Resumes training from a saved checkpoint."""
    print(f"=> Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

def calculate_metrics(y_true, y_pred_probs, threshold=0.5, class_names=None):
    """
    Compute comprehensive metrics for multi-label classification.
    
    Args:
        y_true: Ground truth labels (N, num_classes)
        y_pred_probs: Predicted probabilities (N, num_classes)
        threshold: Classification threshold (default: 0.5)
        class_names: Optional list of class names
    
    Returns:
        dict: Dictionary of metrics
    """
    # Check for empty inputs
    if len(y_true) == 0 or len(y_pred_probs) == 0:
        return {
            'auc_macro': 0.0,
            'auc_per_class': None,
            'precision_macro': 0.0,
            'precision_per_class': None,
            'recall_macro': 0.0,
            'recall_per_class': None,
            'f1_macro': 0.0,
            'f1_per_class': None,
            'accuracy': 0.0,
            'num_valid_classes': 0
        }
    
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    # Initialize metrics
    metrics = {
        'auc_macro': 0.0,
        'auc_per_class': None,
        'precision_macro': 0.0,
        'precision_per_class': None,
        'recall_macro': 0.0,
        'recall_per_class': None,
        'f1_macro': 0.0,
        'f1_per_class': None,
        'accuracy': 0.0,
        'num_valid_classes': 0
    }
    
    # AUC Computation
    try:
        auc_per_class = []
        valid_classes = []
        
        for i in range(y_true.shape[1]):
            try:
                # Check if class has both positive and negative samples
                unique_labels = np.unique(y_true[:, i])
                has_both_classes = len(unique_labels) > 1
                
                if has_both_classes:  # ✓ Now it's a boolean, not array
                    auc = roc_auc_score(y_true[:, i], y_pred_probs[:, i])
                    auc_per_class.append(auc)
                    valid_classes.append(i)
                else:
                    auc_per_class.append(np.nan)
                    
            except (ValueError, IndexError) as e:
                auc_per_class.append(np.nan)
        
        metrics['auc_per_class'] = np.array(auc_per_class)
        
        # Compute macro AUC
        if len(valid_classes) > 0:  # Use len(), not just if valid_classes
            valid_aucs = [auc_per_class[i] for i in valid_classes]
            metrics['auc_macro'] = float(np.mean(valid_aucs))
            metrics['num_valid_classes'] = len(valid_classes)
        else:
            metrics['auc_macro'] = 0.0
            metrics['num_valid_classes'] = 0
            
    except Exception as e:
        print(f"AUC computation failed: {e}")
        import traceback
        traceback.print_exc()
        metrics['auc_per_class'] = np.full(y_true.shape[1], np.nan)
        metrics['auc_macro'] = 0.0
    
    # Other Metrics
    try:
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    except Exception as e:
        metrics['precision_per_class'] = np.zeros(y_true.shape[1])
    
    try:
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    except Exception as e:
        metrics['recall_per_class'] = np.zeros(y_true.shape[1])
    
    try:
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
    except Exception as e:
        metrics['f1_per_class'] = np.zeros(y_true.shape[1])
    
    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
    except Exception as e:
        metrics['accuracy'] = 0.0
    
    return metrics

# Setup adapted from Kaggle implementation
def print_metrics(metrics, class_names, show_invalid=True):
    """
    Print comprehensive metrics report.
    
    Args:
        metrics: Dictionary returned by compute_metrics()
        class_names: List of class names
        show_invalid: Whether to show classes with invalid AUC (default: True)
    """
    print("=" * 90)
    print(f"{'Class':<25} {'AUC':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
    print("-" * 90)
    
    # Track statistics
    valid_count = 0
    invalid_classes = []
    
    for i, name in enumerate(class_names):
        # Get AUC (handle NaN)
        if metrics['auc_per_class'] is not None:
            auc = metrics['auc_per_class'][i]
            if np.isnan(auc): 
                auc_str = "N/A" # Mark as N/A for invalid classes
                invalid_classes.append(name)
            else:
                auc_str = f"{auc:.4f}"
                valid_count += 1
        else:
            auc_str = "N/A" # If AUC computation failed entirely, mark all as N/A
            invalid_classes.append(name)
        
        # Get other metrics
        precision = metrics['precision_per_class'][i] if metrics['precision_per_class'] is not None else 0
        recall = metrics['recall_per_class'][i] if metrics['recall_per_class'] is not None else 0
        f1 = metrics['f1_per_class'][i] if metrics['f1_per_class'] is not None else 0
        
        
        # Format output
        name_truncated = name[:24]  # Truncate long names
        print(f"{name_truncated:<25} {auc_str:>10} {precision:>12.4f} "
              f"{recall:>10.4f} {f1:>10.4f}")
    
    # Macro averages
    print("-" * 90)
    auc_macro_str = f"{metrics['auc_macro']:.4f}" if metrics['auc_macro'] is not None else "N/A"
    print(f"{'MACRO AVG':<25} {auc_macro_str:>10} {metrics['precision_macro']:>12.4f} "
          f"{metrics['recall_macro']:>10.4f} {metrics['f1_macro']:>10.4f} {'':>10}")
    
    # Overall accuracy
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    
    # Summary statistics
    if metrics['num_valid_classes'] > 0:
        print(f"Valid Classes (for AUC): {metrics['num_valid_classes']}/{len(class_names)}")
    
    if invalid_classes and show_invalid:
        print(f"\nClasses with invalid AUC ({len(invalid_classes)}):")
        for name in invalid_classes[:5]:  # Show first 5
            print(f"   - {name}")
        if len(invalid_classes) > 5:
            print(f"   ... and {len(invalid_classes) - 5} more")
    
    print("=" * 90)

# --- 4. EXPLAINABILITY (Grad-CAM) ---

class GradCAM:
    """Generates heatmaps to visualize where the model is 'looking'."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks to capture gradients and activations
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.eval()
        output = self.model(input_tensor)
        
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()
        
        heatmap = F.relu(heatmap)
        heatmap /= (torch.max(heatmap) + 1e-8)
        return heatmap.detach().cpu().numpy()

def overlay_heatmap(heatmap, original_img_np):
    """Overlays the heatmap on a grayscale image (BGR format for CV2)."""
    heatmap_resized = cv2.resize(heatmap, (original_img_np.shape[1], original_img_np.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Convert grayscale original to BGR for blending
    if len(original_img_np.shape) == 2:
        original_img_np = cv2.cvtColor(original_img_np, cv2.COLOR_GRAY2BGR)
        
    superimposed = cv2.addWeighted(original_img_np, 0.6, heatmap_color, 0.4, 0)
    return superimposed
