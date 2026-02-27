"""
data_loader.py - Dataset class and transformations for NIH Chest X-Ray classification 
Dataset Source: https://www.kaggle.com/datasets/nih-chest-xrays/data
"""

import os
import json
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer

class NIH8Dataset(Dataset):
    """
    Dataset established with PyTorch for NIH Chest X-Rays from Kaggle. 
    
    Arguments: 
    - csv_input: Path to csv (Data_Entry_2017.csv) containing image metadata and labels.
    - root_folders: List of directories to search for provided images (e.g., ["images/"]).
    - subset_size: Optional integer to limit dataset size by sampling unique patients (critical for avoiding data leakage in medical imaging).
    - transform: Torchvision transforms to apply to each image.
    - cache_path: Path to JSON file-paths (NIH has 100k+ images).
    """
    def __init__(self, csv_input, root_folders, subset_size=None, transform=None, cache_path="path_cache.json"):

        if isinstance(csv_input, str):
            self.df = pd.read_csv(csv_input)
        else:
            self.df = csv_input.copy() # If already loaded
            
        ######################################################################
        # Handling missing labels (also counts and reports missing)
        # Current approach: Fill missing labels with 'No Finding' to prevent split errors.
        # Alternative approach: Drop rows with missing labels to avoid potential bias. 
        ######################################################################
        n_missing = self.df['Finding Labels'].isna().sum()
        if n_missing > 0:
            print(f"[Dataset] Found {n_missing} images with missing labels- Filling with 'No Finding'")
            self.df['Finding Labels'] = self.df['Finding Labels'].fillna('No Finding')
        self.transform = transform

        ######################################################################
        # Multi-Hot Encoding for 'Finding Labels' (e.g., 'Effusion|Infiltration')
        # Converts to binary vector for multi-label classification (e.g., [0,1,1,0,...] for all classes)
        ######################################################################
        self.df['Label_List'] = self.df['Finding Labels'].str.split('|')
        self.mlb = MultiLabelBinarizer()
        self.label_matrix = self.mlb.fit_transform(self.df['Label_List'])
        self.classes = self.mlb.classes_ # The names of the diseases + No Finding
        print(f"[Dataset] Found {len(self.classes)} unique pathologies: {list(self.classes)}")

        ######################################################################
        # Metadata Encoding + Missing data handling: 
        ######################################################################
        # View Position Encoding (AP=1, PA=0)
        self.df['View_Encoded'] = self.df['View Position'].map({'AP': 1, 'PA': 0}).fillna(0)
        # Patient Age - Fill missing values with median age 
        self.df['Patient Age'] = self.df['Patient Age'].fillna(self.df['Patient Age'].median())
        # Patient Gender - Map 'M' to 1, 'F' to 0, and fill missing values with 1 (can consider dropping)
        self.df['Patient Gender'] = self.df['Patient Gender'].map({'M':1, 'F':0}).fillna(1) 

        ######################################################################
        # Optional Patient-Level Subsetting 
        # Sample patients instead of images to avoid data leakage 
        # (critical in medical imaging where multiple images can come from the same patient).
        ######################################################################
        if subset_size:
            unique_pats = self.df['Patient ID'].unique()
            # We sample enough patients to roughly equal the image count desired
            # Assuming each patient has ~3 images on average
            pats_to_keep = np.random.choice(unique_pats, size=subset_size // 3, replace=False) 
            self.df = self.df[self.df['Patient ID'].isin(pats_to_keep)].reset_index(drop=True)
            # Re-sync the label matrix with the new filtered dataframe
            self.label_matrix = self.mlb.transform(self.df['Label_List'])

        ######################################################################
        # Path Indexing w/caching to speed up loading (NIH has 100k+ images)
        ######################################################################
        if os.path.exists(cache_path):
            print(f"[Dataset] Loading image paths from cache: {cache_path}")
            with open(cache_path, 'r') as f:
                self.path_map = json.load(f)
        else:
            print(f"[Dataset] Indexing NIH folders (first run may take ~30s)...")
            self.path_map = {}
            total_images = 0
            
            for folder in root_folders:
                if not os.path.exists(folder):
                    print(f"    Warning: Folder not found: {folder}")
                    continue
                
                for root, _, files in os.walk(folder):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.path_map[f] = os.path.join(root, f)
                            total_images += 1
            
            with open(cache_path, 'w') as f:
                json.dump(self.path_map, f) # Save for future runs
            print(f"    Indexed {total_images:,} images, saved to {cache_path}")

        ######################################################################
        # Dataset Summary
        ######################################################################
        print(f"\n{'='*60}")
        print(f"DATASET SUMMARY")
        print(f"{'='*60}")
        print(f"Total images:     {len(self.df):,}")
        print(f"Unique patients:  {self.df['Patient ID'].nunique():,}")
        print(f"Pathologies:      {len(self.classes)}")
        print(f"\n{'='*60}")

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
        if image is None:
            raise RuntimeError(f"Failed to load image: {full_path}")
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'labels': torch.tensor(self.label_matrix[idx], dtype=torch.float32),
            'meta': {
                'view':         torch.tensor(row['View_Encoded'], dtype=torch.float32),
                'age':          row['Patient Age'],
                'gender':       row['Patient Gender'],
                'image_name':   img_name
            }
        }

def safe_collate(batch):
    """
    Filter out None samples
    Not currently using this in train_model.py but can be helpful if you encounter missing images or loading errors.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None
    return default_collate(batch)

######################################################################
# Transform classes for input images 
######################################################################
class LetterboxPad:
    """
    Pads to square to keep heart/lung proportions accurate (NIH images are 1024x1024). 
    Adds black borders to make image is square w/o distortion and with consistent aspect ratios.
    """
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        padding = [
            (max_wh - w) // 2,          # Left
            (max_wh - h) // 2,          # Top
            (max_wh - w + 1) // 2,      # Right (add 1 pixel if odd)
            (max_wh - h + 1) // 2]      # Bottom (add 1 pixel if odd)
        return F.pad(image, padding, 0, 'constant')
    
class ApplyCLAHE:
    """
    Enhances local contrast for 8-bit NIH grayscale images.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )
        
        img_np = np.array(img)
        if img_np.dtype != np.uint8:
            img_np = (img_np / (img_np.max() / 255)).astype(np.uint8)
        
        return Image.fromarray(clahe.apply(img_np))
    
class RepeatChannels:
    """
    Repeats single grayscale channel to create 3 identical channels for ResNet input.
    This allows us to leverage pretrained weights on ImageNet without modifying the first conv layer.
    (1, H, W) -> (3, H, W)
    """
    def __call__(self, img):
        return img.repeat(3, 1, 1)  # Convert 1-channel to 3-channel by repeating


######################################################################
# Transform pipelines for training and validation.
######################################################################

train_data_transforms = transforms.Compose([
    LetterboxPad(),                         # Standardization
    ApplyCLAHE(),                           # Contrast Fix
    transforms.Resize((256, 256)),          # Common Capstone Grid size
    transforms.RandomRotation(5),           # Data Augmentation (Overfitting fix)
#transforms.RandomHorizontalFlip(),         # Doubling data variety ##### Might be dangerous for xrays 
    transforms.ToTensor(),                  # Normalizing to [0,1]
    RepeatChannels(),                       # Convert 1-channel to 3-channel for ResNet
    transforms.Normalize(                   # NIH-specific mean/std based on ImageNet stats (since we're using pretrained weights)
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ) 
])

# Validation transforms are the same as training but without augmentation
val_data_transforms = transforms.Compose([
    LetterboxPad(),
    ApplyCLAHE(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    RepeatChannels(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])