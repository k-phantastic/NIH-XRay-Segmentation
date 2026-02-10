import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score
from torch.utils.data import DataLoader
from data_loader import get_loaders
from model import OptimizedMedViT
from train import calculate_metrics

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
    # --- CONFIGURATION ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "best_medvit_model.pth"
    CSV_PATH = '../data/Data_Entry_2017.csv'
    IMAGE_ROOT = '../images'
    IMAGE_FOLDERS = [os.path.join(IMAGE_ROOT, f"images_{str(i).zfill(3)}") for i in range(1, 13)]
    SUBSET_SIZE = 120000
    
    # 1. Load Model
    print(f"Loading best model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    class_names = checkpoint['class_names']
    
    model = OptimizedMedViT(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 2. Load Data (Using the new 4-value unpack)
    print("Initializing Data Loaders...")
    df = pd.read_csv(CSV_PATH)
    _, _, test_loader, _ = get_loaders(
        df, 
        IMAGE_FOLDERS, 
        batch_size=32, 
        subset_size=SUBSET_SIZE
    )

    all_labels = []
    all_preds = []

    # 3. Evaluation on Test Set
    print(f"Evaluating on {len(test_loader.dataset)} unseen images...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(DEVICE)
            meta = batch['meta'].to(DEVICE)
            labels = batch['labels']
            
            outputs = model(images, meta)
            preds = torch.sigmoid(outputs).cpu().numpy()
            
            all_labels.append(labels.numpy())
            all_preds.append(preds)

    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)

    # 4. Compute Metrics
    metrics = calculate_metrics(all_labels, all_preds, class_names)
    
    # Binary predictions for classification report (threshold = 0.5)
    bin_preds = (all_preds > 0.5).astype(int)

    # 5. Output Results
    print("\n" + "="*30)
    print("FINAL TEST SET REPORT")
    print("="*30)
    print(f"Mean AUC: {metrics['Mean_AUC']:.4f}")
    print("\nDetailed Per-Class Report:")
    print(classification_report(all_labels, bin_preds, target_names=class_names, zero_division=0))

    # 6. Generate Visuals
    plot_auc_bar(metrics, class_names)
    plot_pr_curves(all_labels, all_preds, class_names)

if __name__ == "__main__":
    main()