"""
config.py
Configuration file for NIH Chest X-Ray training (Using ResNet50).
"""

from pathlib import Path
import os

######################################################################
# File Path Setup
# 
# Root
# ├── data/
# │   └── Data_Entry_2017.csv
# ├── images/
# │   ├── images_...
######################################################################
ROOT_DIR = Path.cwd()
CSV_PATH = ROOT_DIR.parent / "data" / "Data_Entry_2017.csv"
IMAGE_FOLDERS = [ROOT_DIR.parent / "images"] # List of folders to search 

CACHE_PATH = "path_cache.json" # Path to save the image path cache

MODEL_SAVE_PATH = "best_resnet50_nih_TEST3.pt" # Path to save the best model checkpoint


######################################################################
# Dataset Subsetting / Splitting
######################################################################
TRAIN_SUBSET_SIZE = None
VAL_SUBSET_SIZE = None

TRAIN_SPLIT = 0.8 
VAL_SPLIT = 0.2 # Change if working with Train/Val/Test split


######################################################################
# Data Loading
######################################################################
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 128

NUM_WORKERS = 6 
PERSISTENT_WORKERS = True # Set to True for faster data loading after the first epoch


######################################################################
# Model Training Parameters
######################################################################
EPOCHS = 30
LEARNING_RATE = 5e-5 # 1e-4
WEIGHT_DECAY = 1e-5


######################################################################
# Training Optimizations
######################################################################
USE_AMP = True            # Use Automatic Mixed Precision for faster training
GRAD_CLIP = None          # Set 0.1 for gradient clipping
MAX_CLASS_WEIGHT = 10.0   # Cap class weights to prevent loss explosion

