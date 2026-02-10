import os
import pandas as pd
import numpy as np
import cv2
import torch
import json
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score

# --- CUSTOM TRANSFORMS ---

class LetterboxPad:
    """Pads to square to keep heart/lung proportions accurate."""
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return F.pad(image, padding, 0, 'constant')

class ApplyCLAHE:
    """Enhances local contrast for grayscale images."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        # REMOVE self.clahe from here to avoid pickling errors
        
    def __call__(self, img):
        img_np = np.array(img)
        
        # Create the CLAHE object ONLY when the worker actually runs the function
        # This prevents the 'cannot pickle cv2.CLAHE' error on Windows
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        
        if img_np.dtype != np.uint8:
            img_np = (img_np / (img_np.max() / 255)).astype(np.uint8)
            
        return Image.fromarray(clahe.apply(img_np))

# --- DATASET CLASS ---

class NIH8Dataset(Dataset):
    def __init__(self, csv_input, root_folders, subset_size=None, transform=None, cache_path="path_cache.json"):
        # 1. Load Dataframe
        self.df = csv_input.copy() if not isinstance(csv_input, str) else pd.read_csv(csv_input)
            
        # 2. Label Processing
        self.df['Finding Labels'] = self.df['Finding Labels'].fillna('No Finding')
        self.df['Label_List'] = self.df['Finding Labels'].str.split('|')
        self.mlb = MultiLabelBinarizer()
        self.label_matrix = self.mlb.fit_transform(self.df['Label_List'])
        self.classes = self.mlb.classes_ 
        
        # 3. Metadata Clean/Encode
        self.df['View_Encoded'] = self.df['View Position'].map({'AP': 1, 'PA': 0}).fillna(0)
        self.df['Age_Cleaned'] = np.clip(self.df['Patient Age'], 0, 100) / 100.0
        self.df['Gender_Encoded'] = self.df['Patient Gender'].map({'M': 1, 'F': 0}).fillna(0)
        
        self.transform = transform
        
        # 4. Path Indexing
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self.path_map = json.load(f)
        else:
            print("Indexing NIH folders...")
            self.path_map = {f: os.path.join(r, f) for folder in root_folders 
                             for r, _, files in os.walk(folder) for f in files if f.lower().endswith(('.png', '.jpg'))}
            with open(cache_path, 'w') as f:
                json.dump(self.path_map, f)

        self.df = self.df[self.df['Image Index'].isin(self.path_map.keys())].reset_index(drop=True)
        self.label_matrix = self.mlb.transform(self.df['Label_List'])

        # 5. Subsetting
        if subset_size and subset_size < len(self.df):
            unique_pats = self.df['Patient ID'].unique()
            pats_to_keep = np.random.choice(unique_pats, size=min(len(unique_pats), subset_size // 2), replace=False)
            self.df = self.df[self.df['Patient ID'].isin(pats_to_keep)].reset_index(drop=True)
            self.label_matrix = self.mlb.transform(self.df['Label_List'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        full_path = self.path_map.get(row['Image Index'])
        
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        meta_tensor = torch.tensor([
            row['View_Encoded'], 
            row['Age_Cleaned'], 
            row['Gender_Encoded']
        ], dtype=torch.float32)

        return {
            'image': image,
            'labels': torch.tensor(self.label_matrix[idx], dtype=torch.float32),
            'meta': meta_tensor
        }

# --- UTILS ---

def calculate_class_weights(dataset):
    all_labels = dataset.label_matrix 
    pos_counts = np.sum(all_labels, axis=0)
    total_samples = all_labels.shape[0]
    # Sqrt scaling for stability
    class_weights = np.sqrt((total_samples - pos_counts) / (pos_counts + 1e-6))
    return torch.tensor(class_weights, dtype=torch.float32)

def calculate_metrics(all_labels, all_preds, class_names):
    metrics = {}
    aucs = []
    for i, name in enumerate(class_names):
        try:
            if len(np.unique(all_labels[:, i])) > 1:
                score = roc_auc_score(all_labels[:, i], all_preds[:, i])
                metrics[f'AUC_{name}'] = score
                aucs.append(score)
            else:
                metrics[f'AUC_{name}'] = 0.5
        except:
            metrics[f'AUC_{name}'] = 0.5
    metrics['Mean_AUC'] = np.mean(aucs) if aucs else 0.5
    return metrics