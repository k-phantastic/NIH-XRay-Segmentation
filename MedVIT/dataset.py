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
    def __init__(self, csv_input, root_folders, subset_size=None, transform=None, cache_path="path_cache.json"):
        self.df = csv_input.copy() if not isinstance(csv_input, str) else pd.read_csv(csv_input)
        self.transform = transform

        # 1. Path Indexing (Keep this as is, it's already optimized)
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self.path_map = json.load(f)
        else:
            self.path_map = {f: os.path.join(r, f) for folder in root_folders 
                             for r, _, files in os.walk(folder) for f in files if f.lower().endswith(('.png', '.jpg'))}
            with open(cache_path, 'w') as f:
                json.dump(self.path_map, f)

        # Sync and Label Processing
        self.df['Finding Labels'] = self.df['Finding Labels'].fillna('No Finding')
        self.df['Label_List'] = self.df['Finding Labels'].str.split('|')
        self.mlb = MultiLabelBinarizer()
        
        # Only keep rows we actually have files for
        self.df = self.df[self.df['Image Index'].isin(self.path_map.keys())].reset_index(drop=True)
        self.label_matrix = self.mlb.fit_transform(self.df['Label_List'])
        self.classes = self.mlb.classes_

        # 2. VECTORIZATION STEP: Pre-calculate all metadata and labels into NumPy
        # Doing this here saves massive amounts of time during the actual training loop
        self.view_arr = self.df['View Position'].map({'AP': 1, 'PA': 0}).fillna(0).values.astype(np.float32)
        self.age_arr = (np.clip(self.df['Patient Age'].values, 0, 100) / 100.0).astype(np.float32)
        self.gender_arr = self.df['Patient Gender'].map({'M': 1, 'F': 0}).fillna(0).values.astype(np.float32)
        self.image_names = self.df['Image Index'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # FAST INDEXING: No more .iloc here
        img_name = self.image_names[idx]
        full_path = self.path_map[img_name]
        
        # Image Loading
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
            
        # Fast Metadata assembly from pre-calculated arrays
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
    pos_counts = np.sum(dataset.label_matrix, axis=0)
    total_samples = len(dataset)
    class_weights = np.sqrt((total_samples - pos_counts) / (pos_counts + 1e-6))
    return torch.tensor(class_weights, dtype=torch.float32)