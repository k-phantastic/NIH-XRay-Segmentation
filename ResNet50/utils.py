"""
utils.py - Helper functions for training, evaluation, and explainability/visualization. 
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import torch.nn.functional as F
import cv2
import seaborn as sns

def calculate_class_weights(dataset, max_weight=10.0):
    """
    Calculates weights to handle class imbalance. (We use with BCEWithLogitsLoss)
    Standard formula for Multi-label: ratio of negatives to positives.

    Arguments: 
    - dataset: The dataset object which has a label_matrix attribute (N x num_classes)
    - max_weight: Cap for weights to prevent explosion (default: 10.0)
    """
    
    all_labels = dataset.label_matrix                   # Get the binary matrix (N images x 15 classes)
    pos_counts = np.sum(all_labels, axis=0)         # Count positive occurrences (ones) for each class
    total_samples = all_labels.shape[0]
    neg_counts = total_samples - pos_counts         # Count negative occurrences (zeros)
    
    # Calculate pos_weight (Negatives / Positives), standard input for torch.nn.BCEWithLogitsLoss
    class_weights = neg_counts / (pos_counts + 1e-6)# We add 1e-6 to avoid division by zero
    
    # Cap weights to prevent explosion
    class_weights = np.minimum(class_weights, max_weight)
    
    # Map to names for the printout
    weight_dict = {dataset.classes[i]: round(class_weights[i], 4) for i in range(len(dataset.classes))}
    
    print(f"    [Utils] Class weights calculated for {len(weight_dict)} classes with range [{class_weights.min():.2f}, {class_weights.max():.2f}]")
    print("     [Utils] Identified Class Weights:")
    for i, (class_name, weight) in enumerate(weight_dict.items()):
        print(f"  {class_name}: {weight:.2f}")
    return torch.tensor(class_weights, dtype=torch.float32), weight_dict

######################################################################
# Data Splitting
######################################################################

def get_train_val_split(df, train_size=0.8, random_state=42):
    """
    Splits the dataframe based on Patient ID to prevent data leakage.
    --> Creates train and vailidation sets
    """
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    
    # This ensures that all images of the same patient stay together
    train_idx, val_idx = next(gss.split(df, groups=df['Patient ID']))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    print(f"    [Utils] Train-Val Split Complete: ")
    print(f"    Train   ---> Images: {len(train_df)} | Unique Patients: {train_df['Patient ID'].nunique()} | Num. Classes: {train_df['Finding Labels'].nunique()}")
    print(f"    Val     ---> Images: {len(val_df)} | Unique Patients: {val_df['Patient ID'].nunique()} | Num. Classes: {val_df['Finding Labels'].nunique()}")

    return train_df, val_df

def get_train_val_test_split(df, train_size=0.7, val_size=0.15, random_state=42):
    """
    Splits the dataframe based on Patient ID to prevent data leakage.
    --> Creates train, vailidation sets, and test set 
    """
    # Catch edge case where train_size + val_size >= 1.0
    if train_size + val_size >= 1.0:
        raise ValueError("train_size + val_size must be less than 1.0 to leave room for the test set.")
    test_size = 1.0 - train_size - val_size
    print(f"[Utils] Performing Train/Val/Test split with proportions: Train={train_size}, Val={val_size}, Test={test_size}")
    # 1. Catch test set first
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df['Patient ID']))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)  

    # 2. Now split the remaining train_val_df into train and val
    relative_val_size = val_size / (train_size + val_size)  # e.g. 0.15/0.85 ≈ 0.176

    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df['Patient ID']))
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df   = train_val_df.iloc[val_idx].reset_index(drop=True)
    
    print(f"    [Utils] Train-Val-Test Split Complete: ")
    print(f"    Train   ---> Images: {len(train_df)} | Unique Patients: {train_df['Patient ID'].nunique()} | Num. Classes: {train_df['Finding Labels'].nunique()}")
    print(f"    Val     ---> Images: {len(val_df)} | Unique Patients: {val_df['Patient ID'].nunique()} | Num. Classes: {val_df['Finding Labels'].nunique()}")
    print(f"    Test    ---> Images: {len(test_df)} | Unique Patients: {test_df['Patient ID'].nunique()} | Num. Classes: {test_df['Finding Labels'].nunique()}")
    
    return train_df, val_df, test_df


######################################################################
# Device Selection, Checkpointing
######################################################################
def get_device():
    """
    Returns the best available device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """
    Saves the current training state.
    """
    print(f"    [Utils] => Saving checkpoint: {filename} (Epoch {state['epoch']})")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Resumes training from a saved checkpoint.
    """
    print(f"    [Utils] => Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

######################################################################
# Metrics
######################################################################

def calculate_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Compute comprehensive metrics for multi-label classification.
    
    Arguments:
    - y_true: Ground truth labels (N, num_classes)
    - y_pred_probs: Predicted probabilities (N, num_classes)
    - threshold: Classification threshold (default: 0.5)
    
    Returns:
    - dict: Dictionary of metrics
    """

    ######################################################################
    # Check for empty inputs, initialize metrics
    ######################################################################
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
    # Convert probabilities to binary predictions based on threshold
    y_pred = (y_pred_probs >= threshold).astype(int) 
    
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

    ######################################################################
    # AUC Computation
    ######################################################################
    try:
        auc_per_class = []
        valid_classes = []
        
        for i in range(y_true.shape[1]):
            try:
                # Check if class has both positive and negative samples
                unique_labels = np.unique(y_true[:, i])
                has_both_classes = len(unique_labels) > 1
                
                if has_both_classes:  # Boolean, not array
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
        print(f"    [Utils] AUC computation failed: {e}")
        metrics['auc_per_class'] = np.full(y_true.shape[1], np.nan)
        metrics['auc_macro'] = 0.0

    ######################################################################
    # Precision, Recall, f1-score, accuracy
    ######################################################################
    for name, fn in [
        ("precision", precision_score),
        ("recall", recall_score),
        ("f1", f1_score)
    ]:
        try:
            metrics[f'{name}_macro'] = fn(y_true, y_pred, average='macro', zero_division=0)
            metrics[f'{name}_per_class'] = fn(y_true, y_pred, average=None, zero_division=0)
        except Exception as e:
            print(f"    [Utils] {name} computation failed: {e}")
            metrics[f'{name}_per_class'] = np.zeros(y_true.shape[1])
            metrics[f'{name}_macro'] = 0.0
    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred) # Exact match accuracy 
    except Exception as e:
        metrics['accuracy'] = 0.0

    return metrics

# Setup adapted from Kaggle implementation
def print_metrics(metrics, class_names, show_invalid=True):
    """
    Print comprehensive metrics report.
    
    Arguments:
    - metrics: Dictionary returned by compute_metrics()
    - class_names: List of class names
    - show_invalid: Whether to show classes with invalid AUC (default: True)
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
        recall =    metrics['recall_per_class'][i] if metrics['recall_per_class'] is not None else 0
        f1 =        metrics['f1_per_class'][i] if metrics['f1_per_class'] is not None else 0
        
        
        # Format output
        print(f"{name[:24]:<25} {auc_str:>10} {precision:>12.4f} "
              f"{recall:>10.4f} {f1:>10.4f}")
    
    # Macro averages
    print("-" * 90)
    auc_macro_str = f"{metrics['auc_macro']:.4f}" if metrics['auc_macro'] is not None else "N/A"
    print(f"{'MACRO AVG':<25} {auc_macro_str:>10} {metrics['precision_macro']:>12.4f} "
          f"{metrics['recall_macro']:>10.4f} {metrics['f1_macro']:>10.4f} {'':>10}")
    
    # Overall accuracy
    print(f"\nOverall (Exact Match) Accuracy: {metrics['accuracy']:.4f}")
    
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


######################################################################
# Explainability and Visualization
# Grad-CAM, heatmap, visualization functions
######################################################################

def plot_training_history(checkpoint):
    """
    Plots training and validation loss curves over epochs.
    """
    history = checkpoint['history']
    best_auc = checkpoint['best_auc']
    checkpoint_epoch = checkpoint['epoch']

    total_epochs  = range(1, len(history['train_loss']) + 1)
    sns.set_theme(style="whitegrid") # match the densenet notebook
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(total_epochs, history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(total_epochs, history['val_loss'],   label='Val Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss')
    ax1.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xticks(total_epochs)
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(total_epochs, history['mean_auc'], label='Val AUC', color='green', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean AUC')
    ax2.set_title('Model Performance (Mean AUC)', fontsize=14, fontweight='bold')
    ax2.set_xticks(total_epochs)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle(f"Training History - Best AUC: {best_auc:.4f} (Epoch {checkpoint_epoch+1})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def visualize_results(batch_dict, classes, n=4):
    """
    Quickly plots a batch of images with their multi-hot labels.

    Arguments:
    - batch_dict: Dict returned by the DataLoader (keys: 'image', 'labels').
    - classes:    List of class name strings from dataset.classes.
    - n:          Number of images to display (default 4).
    """
    images = batch_dict['image']
    labels = batch_dict['labels']
    
    plt.figure(figsize=(16, 4))
    for i in range(min(n, len(images))):
        plt.subplot(1, n, i+1)
        
        # Un-normalize for viewing
        img = images[i][0].squeeze().cpu().numpy()
        plt.imshow(img, cmap='gray')
        
        # Get names of active labels
        active_labels = [classes[j] for j, val in enumerate(labels[i]) if val == 1]
        plt.title("\n".join(active_labels), fontsize=8)
        plt.axis('off')
    plt.show()

class GradCAM:
    """
    Generates heatmaps to visualize where the model is 'looking'.
    """
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
    """
    Overlays the heatmap on a grayscale image (BGR format for CV2).
    """
    heatmap_resized = cv2.resize(heatmap, (original_img_np.shape[1], original_img_np.shape[0]))
    heatmap_resized = cv2.GaussianBlur(heatmap_resized, (15, 15), 0)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Convert grayscale original to BGR for blending
    if len(original_img_np.shape) == 2:
        original_img_np = cv2.cvtColor(original_img_np, cv2.COLOR_GRAY2BGR)
        
    superimposed = cv2.addWeighted(original_img_np, 0.6, heatmap_color, 0.4, 0)

    return superimposed

def visualize_gradcam(model, val_ds, train_ds, target_layer, device, n_samples=5):
    """
    Generates Grad-CAM heatmap visualizations for one image per pathology.

    For each sample, displays four panels:
    1. Original grayscale X-ray with ground truth labels
    2. Raw Grad-CAM heatmap for the top predicted class
    3. Heatmap overlaid on the original image
    4. Top 3 predicted classes with probabilities and ground truth

    Sampling logic:
    - Tries to show one unique pathology per row (skips No Finding)
    - Falls back to any remaining unseen pathology if fewer than n_samples found

    Arguments:
    - model:     ResNet50 with best weights already loaded (call load_checkpoint first).
    - val_ds:    Validation NIH8Dataset to sample images from.
    - train_ds:  Training dataset - used for class name lookup.
    - device:    Compute device.
    - n_samples: Number of images to visualize (default 5).
    """
    model.eval()
    cam = GradCAM(model, target_layer)

    # Sample one image per unique pathology from the validation set.
    shown_classes = set()
    samples       = []

    # First pass - one unique pathology per sample, no 'No Finding'
    for idx in range(len(val_ds)): 
        sample = val_ds[idx]
        if sample is None:
            continue

        active  = [val_ds.classes[j] for j, v in enumerate(sample["labels"]) if v == 1]
        if not active:
            continue

        primary = active[0]
        if primary not in shown_classes and primary != "No Finding":
            shown_classes.add(primary)
            samples.append(sample)

        if len(samples) >= n_samples:
            break

    num_samples = len(samples)

    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))

    # Ensure axes is always 2D even for a single sample
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for i, sample in enumerate(samples):
        input_tensor = sample["image"].unsqueeze(0).to(device)
        labels       = sample["labels"]

        # Forward pass - get class probabilities
        with torch.no_grad():
            logits = model(input_tensor)
            probs  = torch.sigmoid(logits)[0].cpu().numpy()

        # Top predicted class - used as the Grad-CAM target
        pred_idx  = int(np.argmax(probs))
        pred_name = train_ds.classes[pred_idx]
        pred_prob = probs[pred_idx]

        # Ground truth class names for display
        gt_labels = [train_ds.classes[j] for j, v in enumerate(labels) if v == 1]

        # Generate Grad-CAM heatmap for the top predicted class
        heatmap = cam.generate_heatmap(input_tensor, pred_idx)

        # Un-normalize the image for display.
        img_np  = input_tensor[0, 0].cpu().numpy()
        img_np  = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        img_np  = (img_np * 255).astype(np.uint8)

        overlay = overlay_heatmap(heatmap, img_np)

        # Top 3 predictions for the text panel
        top3_idx = np.argsort(probs)[::-1][:3]
        top3_str = "\n".join([
            f"{train_ds.classes[j]}: {probs[j]:.2f}" for j in top3_idx
        ])

        # Plot 4 panels per row (Original | Heatmap | Overlay | Top 3 Predictions)

        # Panel 1 - original X-ray with ground truth
        axes[i, 0].imshow(img_np, cmap="gray")
        axes[i, 0].set_title(f"Ground Truth: {', '.join(gt_labels)}", fontsize=14)
        axes[i, 0].axis("off") # Remove axes for cleaner look

        # Panel 2 - raw heatmap for the top predicted class
        axes[i, 1].imshow(heatmap, cmap="jet")
        axes[i, 1].set_title(f"Grad-CAM Heatmap (Top Prediction: {pred_name})", fontsize=14)
        axes[i, 1].axis("off")

        # Panel 3 - heatmap overlaid on original image
        axes[i, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[i, 2].set_title(f"Overlay", fontsize=14)
        axes[i, 2].axis("off")

        # Panel 4 - text: top 3 predictions + ground truth
        axes[i, 3].axis("off")
        axes[i, 3].text(0.1, 0.75, "Top 3 Predictions:",
                        fontsize=12, fontweight="bold",
                        transform=axes[i, 3].transAxes)
        axes[i, 3].text(0.1, 0.45, top3_str,
                        fontsize=12, family="monospace",
                        transform=axes[i, 3].transAxes)
        axes[i, 3].text(0.1, 0.15, f"Ground Truth:\n{', '.join(gt_labels)}",
                        fontsize=12, color="green",
                        transform=axes[i, 3].transAxes)

    plt.suptitle(
        f"Grad-CAM Explainability ({num_samples} shown)",
        fontsize=20, y=1.01
    )
    plt.tight_layout()
    plt.show()

    print(f"  Pathologies shown: {', '.join(sorted(shown_classes))}")

def plot_roc_curves(all_labels, all_probs, classes):
    """
    Plots per-class ROC curves in a 3x5 grid for 15 NIH classes.

    Arguments:
    - all_labels: (N, 15) ground truth binary matrix.
    - all_probs:  (N, 15) predicted probabilities.
    - classes:    List of class name strings from dataset.classes
    """
    from sklearn.metrics import roc_curve, auc

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes      = axes.flatten()

    for i, name in enumerate(classes):
        ax = axes[i]

        fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
        ax.fill_between(fpr, tpr, alpha=0.15) # Shade under the curve

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.legend(loc="lower right", fontsize=9) # Add legend for AUC
        ax.set_ylabel("TPR") if i % 5 == 0 else None
        ax.set_xlabel("FPR") if i >= 10 else None

    plt.suptitle("Per-Class ROC Curves", fontsize=14)
    plt.tight_layout()
    plt.show()

######################################################################
# Patience Counter
# https://medium.com/biased-algorithms/a-practical-guide-to-implementing-early-stopping-in-pytorch-for-model-training-99a7cbd46e9d
######################################################################

class EarlyStopping:
    """
    Stops training when validation AUC has not improved for specified epochs.

    Arguments:
    - patience: Number of epochs to wait for improvement before stopping.
    - delta: Minimum change in validation AUC to qualify as an improvement.
    - verbose: Whether to print a message when stopping.
    """
    def __init__(self, patience=5, delta=0.001, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("    [Utils] Stopping early as no improvement has been observed.")

