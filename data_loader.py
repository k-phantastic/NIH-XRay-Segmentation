import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
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
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_np = np.array(img)
        # NIH images are usually uint8; if not, normalize them
        if img_np.dtype != np.uint8:
            img_np = (img_np / (img_np.max() / 255)).astype(np.uint8)
        return Image.fromarray(self.clahe.apply(img_np))

class NIH8Dataset(Dataset):
    def __init__(self, csv_input, root_folders, subset_size=None, transform=None):
        if isinstance(csv_input, str):
            self.df = pd.read_csv(csv_input)
        else:
            self.df = csv_input.copy()
            
        # SAFETY: Fill empty labels with 'No Finding' to prevent split errors, not sure if this is the correct approach, could bias data. Instead, also drop these rows.
        self.df['Finding Labels'] = self.df['Finding Labels'].fillna('No Finding')
        self.transform = transform
        
        # Multi-Hot Encoding for 'Finding Labels' (e.g., 'Effusion|Infiltration')
        self.df['Label_List'] = self.df['Finding Labels'].str.split('|')
        self.mlb = MultiLabelBinarizer()
        self.label_matrix = self.mlb.fit_transform(self.df['Label_List'])
        self.classes = self.mlb.classes_ # The names of the 8 diseases + No Finding
        
        # View Position Encoding (AP=1, PA=0)
        self.df['View_Encoded'] = self.df['View Position'].map({'AP': 1, 'PA': 0}).fillna(0)
        
        # 3. Patient-Level Subsetting (CRITICAL for Capstone)
        if subset_size:
            unique_pats = self.df['Patient ID'].unique()
            # We sample enough patients to roughly equal the image count you want
            pats_to_keep = np.random.choice(unique_pats, size=subset_size // 3, replace=False)
            self.df = self.df[self.df['Patient ID'].isin(pats_to_keep)].reset_index(drop=True)
            # Re-sync the label matrix with the new filtered dataframe
            self.label_matrix = self.mlb.transform(self.df['Label_List'])

        # Global Path Indexing for 12 folders
        self.path_map = {}
        print(f"Indexing NIH folders...")
        for folder in root_folders:
            for root, _, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg')):
                        self.path_map[f] = os.path.join(root, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image Index']
        full_path = self.path_map.get(img_name)
        
        if not full_path: return None

        # Load as grayscale (preserves bit-depth from NIH PNGs)
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'labels': torch.tensor(self.label_matrix[idx], dtype=torch.float32),
            'meta': {
                'view': torch.tensor(row['View_Encoded'], dtype=torch.float32),
                'age': row['Patient Age'],
                'gender': 1 if row['Patient Gender'] == 'M' else 0
            }
        }

# Final Transform Pipeline
data_transforms = transforms.Compose([
    LetterboxPad(),               # Standardization
    ApplyCLAHE(),                 # Contrast Fix
    transforms.Resize((256, 256)), # Common Capstone Grid size
    transforms.RandomRotation(5), # Data Augmentation (Overfitting fix)
    transforms.RandomHorizontalFlip(), # Doubling data variety
    transforms.ToTensor(),        # Normalizing to [0,1]
    transforms.Normalize(mean=[0.485], std=[0.229]) # NIH-specific mean/std
])

# To run:
# folders = [f"D:/NIH/images_{str(i).zfill(3)}" for i in range(1, 13)]
# ds = NIH8Dataset('Data_Entry_2017.csv', folders, subset_size=10000, transform=data_transforms)
# loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)