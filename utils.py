import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
import os
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import cv2


def calculate_class_weights(dataset):
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
    print(f"=> Saving checkpoint: {filename}")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer):
    """Resumes training from a saved checkpoint."""
    print(f"=> Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

def calculate_metrics(all_labels, all_preds, class_names):
    """Calculates Mean AUC-ROC and Per-Class AUC-ROC."""
    metrics = {}
    try:
        # Macro average for overall performance
        metrics['Mean_AUC'] = roc_auc_score(all_labels, all_preds, average='macro')
        
        for i, name in enumerate(class_names):
            try:
                metrics[f'AUC_{name}'] = roc_auc_score(all_labels[:, i], all_preds[:, i])
            except ValueError:
                metrics[f'AUC_{name}'] = 0.0 # Handles cases with no positive samples in subset
    except Exception as e:
        print(f"Metric calculation error: {e}")
        return None
    return metrics

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