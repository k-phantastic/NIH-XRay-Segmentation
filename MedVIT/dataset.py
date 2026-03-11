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

# --- CUSTOM TRANSFORMS ---

class LetterboxPad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp, vp = (max_wh - w) // 2, (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return F.pad(image, padding, 0, 'constant')

class ApplyCLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)
        # Create locally to avoid Windows Pickling/Multiprocessing errors
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        if img_np.dtype != np.uint8:
            img_np = (img_np / (img_np.max() / 255)).astype(np.uint8)
        return Image.fromarray(clahe.apply(img_np))

# --- DATASET CLASS ---

class NIH8Dataset(Dataset):
    def __init__(self, csv_input, root_folders, subset_size=None, transform=None, cache_path="path_cache.json", force_refresh=False):
        # 0. Optional: Automate cache deletion
        if force_refresh and os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"Old cache deleted. Re-scanning {len(root_folders)} folders...")

        self.df = csv_input.copy() if not isinstance(csv_input, str) else pd.read_csv(csv_input)
        self.transform = transform

        # 1. Path Indexing
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self.path_map = json.load(f)
        else:
            self.path_map = {f: os.path.join(r, f) for folder in root_folders 
                             for r, _, files in os.walk(folder) for f in files if f.lower().endswith(('.png', '.jpg'))}
            with open(cache_path, 'w') as f:
                json.dump(self.path_map, f)

        if subset_size is not None:
            self.df = self.df.head(subset_size)

        # 2. Sync and Label Processing
        self.df['Finding Labels'] = self.df['Finding Labels'].fillna('No Finding')
        self.df['Label_List'] = self.df['Finding Labels'].str.split('|')
        
        # FIXED CLASSES: Ensures column 0 is always Atelectasis, column 1 is Cardiomegaly, etc.
        self.classes = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
            'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding',
            'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
        ]
        self.mlb = MultiLabelBinarizer(classes=self.classes)
        
        # Only keep rows we actually have files for
        self.df = self.df[self.df['Image Index'].isin(self.path_map.keys())].reset_index(drop=True)
        
        # Fit the binarizer to our fixed classes
        self.label_matrix = self.mlb.fit_transform(self.df['Label_List'])

        # 3. VECTORIZATION STEP
        self.view_arr = self.df['View Position'].map({'AP': 1, 'PA': 0}).fillna(0).values.astype(np.float32)
        self.age_arr = (np.clip(self.df['Patient Age'].values, 0, 100) / 100.0).astype(np.float32)
        self.gender_arr = self.df['Patient Gender'].map({'M': 1, 'F': 0}).fillna(0).values.astype(np.float32)
        self.image_names = self.df['Image Index'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        full_path = self.path_map[img_name]
        
        # Image Loading
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((1024, 1024), dtype=np.uint8)
            
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
            
        meta_tensor = torch.tensor([
            self.view_arr[idx], 
            self.age_arr[idx], 
            self.gender_arr[idx]
        ], dtype=torch.float32)

        return {
            'image': image,
            'labels': torch.from_numpy(self.label_matrix[idx]).float(),
            'meta': meta_tensor
        }

def calculate_class_weights(dataset):
    # Use the pre-calculated label_matrix instead of trying to index the dataframe
    labels = dataset.label_matrix 
    
    # Sum the columns to get positive counts for each class
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    
    # Formula: pos_weight = negative_counts / positive_counts
    # High weight for rare diseases, low weight for common ones
    weights = neg_counts / (pos_counts + 1e-6)
    
    return torch.tensor(weights, dtype=torch.float32)