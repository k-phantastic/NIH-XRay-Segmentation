import os
import pandas as pd
import numpy as np
import cv2
import torch
import json
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score

class LetterboxPad:
    """Pads to square to keep heart/lung proportions accurate (NIH images are 1024x1024)."""
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        # Calculate padding for left, top, right, bottom
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return F.pad(image, padding, 0, 'constant')

class ApplyCLAHE:
    """Enhances local contrast for 8-bit NIH grayscale images."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = None  # Don't create it here!

    def __call__(self, img):
        img_np = np.array(img)
        
        # Create the CLAHE object only when the worker actually runs
        if self.clahe is None:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit, 
                tileGridSize=self.tile_grid_size
            )
            
        if img_np.dtype != np.uint8:
            img_np = (img_np / (img_np.max() / 255)).astype(np.uint8)
            
        return Image.fromarray(self.clahe.apply(img_np))

class NIH8Dataset(Dataset):
    def __init__(self, csv_input, root_folders, subset_size=None, transform=None, cache_path="path_cache.json"):
        # 1. Load Dataframe
        if isinstance(csv_input, str):
            self.df = pd.read_csv(csv_input)
        else:
            self.df = csv_input.copy()
            
        # 2. Label Processing
        self.df['Finding Labels'] = self.df['Finding Labels'].fillna('No Finding')
        self.df['Label_List'] = self.df['Finding Labels'].str.split('|')
        self.mlb = MultiLabelBinarizer()
        self.label_matrix = self.mlb.fit_transform(self.df['Label_List'])
        self.classes = self.mlb.classes_ 
        
        # 3. Metadata Encoding
        self.df['View_Encoded'] = self.df['View Position'].map({'AP': 1, 'PA': 0}).fillna(0)
        self.transform = transform
        
        # 4. Global Path Indexing with Caching
        if os.path.exists(cache_path):
            print(f"Loading paths from cache: {cache_path}")
            with open(cache_path, 'r') as f:
                self.path_map = json.load(f)
        else:
            print(f"Indexing NIH folders (this may take a few minutes)...")
            self.path_map = {}
            for folder in root_folders:
                for root, _, files in os.walk(folder):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg')):
                            self.path_map[f] = os.path.join(root, f)
            with open(cache_path, 'w') as f:
                json.dump(self.path_map, f)

        # Ensure we only keep rows where the image actually exists on disk
        self.df = self.df[self.df['Image Index'].isin(self.path_map.keys())].reset_index(drop=True)
        self.label_matrix = self.mlb.transform(self.df['Label_List'])

        # 5. Patient-Level Subsetting
        if subset_size and subset_size < len(self.df):
            unique_pats = self.df['Patient ID'].unique()
            pats_to_keep = np.random.choice(unique_pats, size=min(len(unique_pats), subset_size // 2), replace=False)
            self.df = self.df[self.df['Patient ID'].isin(pats_to_keep)].reset_index(drop=True)
            self.label_matrix = self.mlb.transform(self.df['Label_List'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image Index']
        full_path = self.path_map.get(img_name)
        
        if not full_path:
            return None

        # Load as grayscale (preserves bit-depth from NIH PNGs)
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        # Metadata tensor: [View, Age (Normalized), Gender]
        meta_tensor = torch.tensor([
            row['View_Encoded'], 
            row['Patient Age'] / 100.0, 
            1 if row['Patient Gender'] == 'M' else 0
        ], dtype=torch.float32)

        return {
            'image': image,
            'labels': torch.tensor(self.label_matrix[idx], dtype=torch.float32),
            'meta': meta_tensor
        }

# --- UTILITIES ---

def calculate_class_weights(dataset):
    """Calculates weights to handle class imbalance (Negatives / Positives)."""
    all_labels = dataset.label_matrix 
    pos_counts = np.sum(all_labels, axis=0)
    total_samples = all_labels.shape[0]
    neg_counts = total_samples - pos_counts
    class_weights = neg_counts / (pos_counts + 1e-6)
    
    weight_dict = {dataset.classes[i]: round(class_weights[i], 4) for i in range(len(dataset.classes))}
    return torch.tensor(class_weights, dtype=torch.float32), weight_dict

def calculate_metrics(all_labels, all_preds, class_names):
    """Calculates Mean AUC-ROC and Per-Class AUC-ROC."""
    metrics = {}
    aucs = []
    
    for i, name in enumerate(class_names):
        try:
            # Check if we have at least one positive and one negative sample
            if len(np.unique(all_labels[:, i])) > 1:
                score = roc_auc_score(all_labels[:, i], all_preds[:, i])
                metrics[f'AUC_{name}'] = score
                aucs.append(score)
            else:
                metrics[f'AUC_{name}'] = 0.5
        except Exception:
            metrics[f'AUC_{name}'] = 0.5
            
    # Ensure we ALWAYS return a dictionary with Mean_AUC
    metrics['Mean_AUC'] = np.mean(aucs) if len(aucs) > 0 else 0.5
    return metrics