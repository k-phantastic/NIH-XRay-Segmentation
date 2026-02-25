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

class LetterboxPad:
    """Pads to square to keep heart/lung proportions accurate (NIH images are 1024x1024)."""
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        padding = [(max_wh - w) // 2, (max_wh - h) // 2, 
                   (max_wh - w + 1) // 2, (max_wh - h + 1) // 2]
        return F.pad(image, padding, 0, 'constant')
    
class ApplyCLAHE:
    """Enhances local contrast for 8-bit NIH grayscale images."""
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
    def __call__(self, img):
        return img.repeat(3, 1, 1)  # Convert 1-channel to 3-channel by repeating

class NIH8Dataset(Dataset):
    def __init__(self, csv_input, root_folders, subset_size=None, transform=None, cache_path="path_cache.json"):
        if isinstance(csv_input, str):
            self.df = pd.read_csv(csv_input)
        else:
            self.df = csv_input.copy()
            
        # SAFETY: Fill empty labels with 'No Finding' to prevent split errors, not sure if this is the correct approach, could bias data. Instead, also drop these rows.
        # Count missing before handling
        n_missing = self.df['Finding Labels'].isna().sum()
        if n_missing > 0:
            print(f"Found {n_missing} images with missing labels")
            print(f"   Filling with 'No Finding' (alternative: drop these rows)")
            self.df['Finding Labels'] = self.df['Finding Labels'].fillna('No Finding')
        self.transform = transform
        
        # Multi-Hot Encoding for 'Finding Labels' (e.g., 'Effusion|Infiltration')
        self.df['Label_List'] = self.df['Finding Labels'].str.split('|')
        self.mlb = MultiLabelBinarizer()
        self.label_matrix = self.mlb.fit_transform(self.df['Label_List'])
        self.classes = self.mlb.classes_ # The names of the diseases + No Finding
        print(f"Found {len(self.classes)} unique pathologies: {list(self.classes)}")

        # Metadta Encoding: 

        # View Position Encoding (AP=1, PA=0)
        self.df['View_Encoded'] = self.df['View Position'].map({'AP': 1, 'PA': 0}).fillna(0)
        # Age and Gender - Fill missing values with median age and encode gender as binary (M=1, F=0)
        self.df['Patient Age'] = self.df['Patient Age'].fillna(self.df['Patient Age'].median())
        self.df['Patient Gender'] = self.df['Patient Gender'].map({'M':1, 'F':0}).fillna(1) # Assuming missing gender is male, could be biased, consider dropping these rows instead.

        # 3. Patient-Level Subsetting (CRITICAL for Capstone)
        if subset_size:
            unique_pats = self.df['Patient ID'].unique()
            # We sample enough patients to roughly equal the image count you want
            pats_to_keep = np.random.choice(unique_pats, size=subset_size // 3, replace=False)
            self.df = self.df[self.df['Patient ID'].isin(pats_to_keep)].reset_index(drop=True)
            # Re-sync the label matrix with the new filtered dataframe
            self.label_matrix = self.mlb.transform(self.df['Label_List'])

        # Global Path Indexing w/caching to speed up loading (NIH has 100k+ images)
        if os.path.exists(cache_path):
            print(f"Loading image paths from cache: {cache_path}")
            with open(cache_path, 'r') as f:
                self.path_map = json.load(f)
        else:
            print(f"Indexing NIH folders (this may take ~30s)...")
            self.path_map = {}
            total_images = 0
            
            for folder in root_folders:
                if not os.path.exists(folder):
                    print(f"Warning: Folder not found: {folder}")
                    continue
                
                for root, _, files in os.walk(folder):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.path_map[f] = os.path.join(root, f)
                            total_images += 1
            
            # Save cache for next time
            with open(cache_path, 'w') as f:
                json.dump(self.path_map, f)
            
            print(f"Indexed {total_images:,} images, saved to {cache_path}")
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
        
        if not full_path: return None

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
                'view': torch.tensor(row['View_Encoded'], dtype=torch.float32),
                'age': row['Patient Age'],
                'gender': row['Patient Gender'],
                'image_name': img_name
            }
        }

def safe_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None
    return default_collate(batch)

# Modfified from data_loader.py 
train_data_transforms = transforms.Compose([
    LetterboxPad(),               # Standardization
    ApplyCLAHE(),                 # Contrast Fix
    transforms.Resize((256, 256)), # Common Capstone Grid size
    transforms.RandomRotation(5), # Data Augmentation (Overfitting fix)
    #transforms.RandomHorizontalFlip(), # Doubling data variety ##### Might be dangerous for xrays 
    transforms.ToTensor(),        # Normalizing to [0,1]
    RepeatChannels(),             # Convert 1-channel to 3-channel for ResNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # NIH-specific mean/std
])

# Validation transforms are the same as training but without augmentation
val_data_transforms = transforms.Compose([
    LetterboxPad(),
    ApplyCLAHE(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    RepeatChannels(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])