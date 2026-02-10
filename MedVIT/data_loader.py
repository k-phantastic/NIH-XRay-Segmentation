import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit
import os

from dataset import NIH8Dataset, LetterboxPad, ApplyCLAHE

# Standard ImageNet normalization + NIH specific tweaks
data_transforms = transforms.Compose([
    LetterboxPad(),                
    ApplyCLAHE(),                  
    transforms.Resize((224, 224)), 
    transforms.RandomRotation(5),  
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(),         
    transforms.Normalize(mean=[0.485], std=[0.229]) 
])

def get_train_val_test_split(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Splits by Patient ID into three sets based on provided proportions.
    """
    # Ensure proportions add up to 1.0
    total = train_size + val_size + test_size
    train_size, val_size, test_size = train_size/total, val_size/total, test_size/total

    # 1. Separate the Test set first
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df['Patient ID']))
    
    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    
    # 2. Separate Train and Val from the remaining data
    # We calculate the relative size of val compared to (train + val)
    relative_val_size = val_size / (train_size + val_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val['Patient ID']))
    
    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)
    
    return df_train, df_val, df_test

def get_loaders(df, root_folders, batch_size=32, subset_size=None, num_workers=2, 
                train_p=0.7, val_p=0.15, test_p=0.15):
    """
    Now accepts split percentages as parameters.
    """
    # 1. Get the three splits using the parameters
    train_df, val_df, test_df = get_train_val_test_split(
        df, train_size=train_p, val_size=val_p, test_size=test_p
    )
    
    # 2. Calculate proportional subsets dynamically
    # We treat 'subset_size' as the total budget for the experiment
    val_subset = None
    test_subset = None
    
    if subset_size:
        # If user provides a subset_size, we split it according to the requested percentages
        # e.g., if subset_size=5000 and val_p=0.15, val_subset=750
        val_subset = int(subset_size * (val_p / train_p)) 
        test_subset = int(subset_size * (test_p / train_p))
        
        # Note: In this logic, subset_size is used specifically for the TRAIN set, 
        # and Val/Test are scaled relative to it to maintain the ratio.
    
    print(f"Data Setup | Ratio: {train_p}/{val_p}/{test_p}")
    print(f"Subsets    | Train: {subset_size if subset_size else 'Full'} "
          f"| Val: {val_subset if val_subset else 'Full'} "
          f"| Test: {test_subset if test_subset else 'Full'}")

    # 3. Initialize Datasets
    train_ds = NIH8Dataset(train_df, root_folders, subset_size=subset_size, transform=data_transforms)
    val_ds = NIH8Dataset(val_df, root_folders, subset_size=val_subset, transform=data_transforms) 
    test_ds = NIH8Dataset(test_df, root_folders, subset_size=test_subset, transform=data_transforms)

    # 4. Create Loaders
    loader_args = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True}
    
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)
    
    return train_loader, val_loader, test_loader, train_ds.classes