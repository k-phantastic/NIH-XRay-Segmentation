import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit
import os
from dataset import NIH8Dataset, LetterboxPad, ApplyCLAHE

def get_train_val_test_split(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    total = train_size + val_size + test_size
    train_size, val_size, test_size = train_size/total, val_size/total, test_size/total

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df['Patient ID']))
    
    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    
    relative_val_size = val_size / (train_size + val_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val['Patient ID']))
    
    return df_train_val.iloc[train_idx].reset_index(drop=True), \
           df_train_val.iloc[val_idx].reset_index(drop=True), \
           df_test

def get_loaders(df, root_folders, batch_size=32, subset_size=None, num_workers=6, 
                train_p=0.7, val_p=0.15, test_p=0.15, target_size=(448, 448)):
    
    # 1. Split logic (Stayed the same)
    train_df, val_df, test_df = get_train_val_test_split(df, train_p, val_p, test_p)
    
    # 2. Dynamic Transforms (PHASE AWARE)
    train_transforms = transforms.Compose([
        LetterboxPad(),
        ApplyCLAHE(),
        transforms.Resize(target_size),
        transforms.RandomRotation(3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    val_transforms = transforms.Compose([
        LetterboxPad(),
        ApplyCLAHE(),
        transforms.Resize(target_size),
        # No Rotation or Flip here - we want clean evaluation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    # 3. Subsetting Logic (Stayed the same)
    val_subset = int(subset_size * (val_p / train_p)) if subset_size else None
    test_subset = int(subset_size * (test_p / train_p)) if subset_size else None

    # 4. Initialize Datasets with Phase-specific transforms
    train_ds = NIH8Dataset(train_df, root_folders, subset_size=subset_size, transform=train_transforms)
    val_ds = NIH8Dataset(val_df, root_folders, subset_size=val_subset, transform=val_transforms) 
    test_ds = NIH8Dataset(test_df, root_folders, subset_size=test_subset, transform=val_transforms)

    # 5. Create Loaders
    loader_args = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True}
    
    return (
        DataLoader(train_ds, shuffle=True, **loader_args),
        DataLoader(val_ds, shuffle=False, **loader_args),
        DataLoader(test_ds, shuffle=False, **loader_args),
        train_ds.classes
    )