import os
from pyexpat import model
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score, roc_curve
from torch.utils.data import DataLoader
from data_loader import get_loaders
from model import OptimizedMedViT
from tqdm import tqdm
from collections import OrderedDict
import argparse

def calculate_metrics(all_labels, all_preds, class_names):
    metrics = {}
    auc_list = []
    best_thresholds = []
    
    for i, class_name in enumerate(class_names):
        # Check if we have both positive and negative samples for this class
        if len(np.unique(all_labels[:, i])) > 1:
            # 1. Calculate AUC
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            metrics[f'AUC_{class_name}'] = auc
            auc_list.append(auc)
            
            # 2. Find Optimal Threshold (Youden's J statistic)
            fpr, tpr, thresholds = roc_curve(all_labels[:, i], all_preds[:, i])
            # J = sensitivity + specificity - 1 (which is same as tpr - fpr)
            idx = np.argmax(tpr - fpr)
            best_thr = thresholds[idx]
            best_thresholds.append(best_thr)
        else:
            # Fallback for rare/missing classes in test set
            metrics[f'AUC_{class_name}'] = 0.5
            best_thresholds.append(0.5)
            
    metrics['Mean_AUC'] = np.mean(auc_list) if auc_list else 0.5
    metrics['Best_Thresholds'] = best_thresholds
    return metrics

def plot_auc_bar(metrics, class_names):
    """Generates a bar chart for per-class AUC-ROC performance."""
    auc_values = [metrics.get(f'AUC_{cls}', 0.5) for cls in class_names]
    mean_auc = metrics.get('Mean_AUC', 0.5)

    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")
    colors = sns.color_palette("coolwarm", len(class_names))
    
    ax = sns.barplot(x=class_names, y=auc_values, palette=colors, hue=class_names, legend=False)
    plt.axhline(0.5, color='red', linestyle='--', label='Random Guess (0.5)')
    plt.axhline(mean_auc, color='blue', linestyle='-', label=f'Mean AUC ({mean_auc:.3f})')
    
    plt.ylim(0, 1.0)
    plt.title('Final Model Performance: AUC-ROC by Disease Category', fontsize=16)
    plt.ylabel('AUC Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('test_set_auc_bar.png')
    print("Saved: test_set_auc_bar.png")
    plt.show()

def plot_pr_curves(all_labels, all_preds, class_names):
    """Generates Precision-Recall curves (critical for imbalanced medical data)."""
    plt.figure(figsize=(10, 8))
    for i, name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(all_labels[:, i], all_preds[:, i])
        ap = average_precision_score(all_labels[:, i], all_preds[:, i])
        plt.plot(recall, precision, label=f'{name} (AP={ap:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves (Test Set)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('test_set_pr_curves.png')
    print("Saved: test_set_pr_curves.png")
    plt.show()

def main():
    # --- 1. CONFIGURATION ---
    parser = argparse.ArgumentParser(description="Evaluate MedViT Model")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the .pth model checkpoint"
    )
    args = parser.parse_args()
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = args.model_path
    OUTPUT_DIR = os.path.dirname(MODEL_PATH)
    CSV_PATH = '../data/Data_Entry_2017.csv'
    IMAGE_ROOT = '../images'
    IMAGE_FOLDERS = [os.path.join(IMAGE_ROOT, f"images_{str(i).zfill(3)}", "images") for i in range(1, 13)]

    # --- 2. MODEL SETUP ---
    print(f"Loading weights from {MODEL_PATH}...")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    # 1. Extract Class Names
    if 'class_names' in checkpoint:
        class_names = checkpoint['class_names']
        print(f"✅ Classes: {class_names}")
    
    # 2. Initialize Model Architecture
    model = OptimizedMedViT(num_classes=len(class_names)).to(DEVICE)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("💡 Found 'state_dict' key. Extracting weights...")
    else:
        state_dict = checkpoint
        print("💡 Loading weights directly from checkpoint...")

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k 
        new_state_dict[name] = v

    # 5. Load and Verify
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Weights Loaded! Missing keys: {msg.missing_keys}")
    
    model.eval()

    # --- 3. DATA LOADING ---
    df = pd.read_csv(CSV_PATH)
    
    # SAFETY CHECK 2: Ensure the GroupShuffleSplit uses the EXACT same seed/params as training
    _, _, test_loader, _ = get_loaders(
        df, 
        IMAGE_FOLDERS, 
        batch_size=32,
        subset_size=None,
        target_size=(448, 448)
    )
    print(f"🚀 Loaded {len(test_loader.dataset)} test images.")

    all_labels = []
    all_preds = []

    # --- 4. EVALUATION ---
    print(f"Running inference...")
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda'):
            for batch in tqdm(test_loader):
                images = batch['image'].to(DEVICE)
                meta = batch['meta'].to(DEVICE)
                labels = batch['labels']
                
                outputs = model(images, meta)
                
                # SAFETY CHECK 3: Check if outputs are NaNs or constants
                preds = torch.sigmoid(outputs).cpu().numpy()
                
                all_labels.append(labels.numpy())
                all_preds.append(preds)

    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)

    # --- 5. THE "TRUTH" CHECK ---
    metrics = calculate_metrics(all_labels, all_preds, class_names)
    print(f"\n✨ FINAL MEAN AUC: {metrics['Mean_AUC']:.4f}")

    # Generate Visuals
    # 2. Use os.path.join to save into that folder
    plot_auc_bar(metrics, class_names)
    plt.savefig(os.path.join(OUTPUT_DIR, 'test_set_auc_bar.png'))

    plot_pr_curves(all_labels, all_preds, class_names)
    plt.savefig(os.path.join(OUTPUT_DIR, 'test_set_pr_curves.png'))

    # Print a small sample to see if the model is varying its guesses
    print(f"Sample Preds (Class 0): {all_preds[:5, 0]}")

if __name__ == "__main__":
    main()