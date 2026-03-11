# ResNet-50 for NIH Chest X-ray Classification
**Author:** Khanh Phan  
**Model Info:** ResNet-50 (torchvision, ImageNet pretrained)  

---

## Overview

This directory contains the setup for tuning a pre-trained ResNet50 model from torchvision's ``models.resnet50(weights=models.ResNet50_Weights.DEFAULT)``. It is applied to the [NIH Chest X-ray](https://www.kaggle.com/datasets/nih-chest-xrays/data) dataset for multi-label classification of 14 pathologies (+ 1 for patients with no finding). Some key design decisions include: 

- **Patient-level train/val/test split (70/15/15)** to prevent data leakage across the repeated patients
- **CLAHE preprocessing** to enhance low-contrast findings
- **Class weighting** with a tuned cap to handle extreme label imbalance (~200 hernia cases)
- **Optuna hyperparameter tuning** to select learning rates, weight decay, and class weight cap
- **512x512** training for enhanced detailing on "smaller" pathologies
- **Grad-CAM explainability** to visualize which image regions drive each prediction        

---

## File Structure
```
ResNet50/
├── config.py           # All hyperparameters, file paths, and training toggles
├── data_loader.py      # NIH8Dataset class for the dataset, preprocessing transforms
├── train_model.py      # Full training loop with validation, checkpointing, and test evaluation
├── tuning.py           # Optuna hyperparameter search (ran before train_model.py)
├── utils.py            # Metrics, data splitting, GradCAM, visualization, patience
├── resnet.ipynb        # Notebook for post-training analysis (ROC curves, Grad-CAM, metrics)
├── path_cache.json     # Image path index (created on first run)
└── best_params*.json   # Optuna output (created after tuning)
```

---

## Setup

### Requirements

Note, `torch` version ` 2.10.0+cu128` was used as model was trained locally using a 5070 Ti. 

```bash
pip install torch torchvision opencv-python pillow scikit-learn pandas numpy matplotlib tqdm optuna
```

### Data Layout

Download the dataset from Kaggle, sample arrangement as setup by the author is as follows. 

```
Root
├── data/
│   └── Data_Entry_2017.csv         (included upon cloning repo)
├── images/                         (sample arrangement)
│   ├── images_001/
│   │   └── images/
│   │       └── *.png
│   ├── images_002/
│   ...
│   └── images_012/
└── ResNet50/                       (current folder)             
```

Additional confirmation can be established in `config.py` to match your layout:

```python
...
ROOT_DIR = Path.cwd()
CSV_PATH = ROOT_DIR.parent / "data" / "Data_Entry_2017.csv"
IMAGE_FOLDERS = [ROOT_DIR.parent / "images"] # List of folders to search 
...
```

---

## Execution Workflow
### Hyperparameter Tuning

```bash
python tuning.py
```

Hyperparameter tuning is performed using Optuna. Default settings were 20 trials at 5 proxy epochs per trial. Settings are located at the top of `tuning.py` to change this as well as the output file (default `best_params.json` outputted at current folder) for the best parameters. Completed parameters can be copied into `config.py`. Additional parameters can be studied using this framework, but with our current scope we found the following: 

```python 
LEARNING_RATE = 5.6115164153345e-05
LR_LAYER4 = 7.969454818643937e-05
WEIGHT_DECAY = 0.000157029708840554
MAX_CLASS_WEIGHT = 13.177194231349622
```

### Full Dataset Loading and Training

---

#### Preprocessing

As referenced in `data_loader.py`, all training images follow the following: 
| Step | Class / Transform | Purpose |
|---|---|---|
| 1 | `LetterboxPad` | Pads to square with black borders to ensure same image shape and proportion as input
| 2 | `ApplyCLAHE` | "Contrast-Limited Adaptive Histogram Equalization" (clip=2.0, tile=8×8) to enhance local contrast and reveal subtle details |
| 3 | `Resize(512, 512)` | Higher than standard recommendations, more expensive to train, but retains more detail (especially for smaller pathologies) |
| 4 | `RandomRotation(5)` | simulates slight patient positioning variation |
| 5 | `ToTensor` | normalizes to [0, 1] |
| 6 | `RepeatChannels` | Copies single grayscale channel to 3 identical channels for ResNet's expected input |
| 7 | `Normalize` | ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) — compatible with pretrained weights |

---

#### Full Execution
```bash
python train_model.py
```

The data is fully loaded with the necessary functions for the model's training loop. We ensure a patient stratified split leveraging `GroupShuffleSplit`. As mentioned previously, the training is performed on torchvision's `models.resnet50(weights=models.ResNet50_Weights.DEFAULT)`. All necessary settings for `train_model.py` can be established through `config.py`.

---

#### Additional Archetecture
We take advantage of the `save_checkpoint()` function from `utils.py` set up previously by Danny to save the best epoch to also allow the model to train continuously with restart point in case of any failure. 

Training proceeds for up to 30 epochs with early stopping (patience=5, delta=0.001), with progress summarized every 5 epochs upon running. 

We use `AdamW()` for our optimizer and `BCEWithLogitsLoss()` as our criterion loss function. 

#### Result Aggregation
Upon completion of training and obtaining a checkpoint file with the best results, `resnet.ipynb` can be ran to view final statistics and relevant visualizations for evaluation.

Included in `utils.py` is a setup for Grad-CAM (Gradient Class Activation Mapping) to help see regions of the X-ray input that the model is most influenced by.

---

## Results

| Class | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Atelectasis | 0.7882 | 0.2311 | 0.6807 | 0.3451 |
| Cardiomegaly | 0.8887 | 0.1690 | 0.6009 | 0.2638 |
| Consolidation | 0.7988 | 0.1261 | 0.5882 | 0.2077 |
| Edema | 0.8917 | 0.1567 | 0.5345 | 0.2423 |
| Effusion | 0.8675 | 0.3344 | 0.7917 | 0.4702 |
| Emphysema | 0.9306 | 0.3263 | 0.6643 | 0.4376 |
| Fibrosis | 0.7972 | 0.1253 | 0.1708 | 0.1446 |
| Hernia | 0.9118 | 0.4000 | 0.0465 | 0.0833 |
| Infiltration | 0.6879 | 0.2703 | 0.6039 | 0.3734 |
| Mass | 0.7995 | 0.2051 | 0.4946 | 0.2899 |
| No Finding | 0.7623 | 0.7219 | 0.7367 | 0.7292 |
| Nodule | 0.7341 | 0.1410 | 0.4806 | 0.2180 |
| Pleural Thickening | 0.7787 | 0.1382 | 0.3179 | 0.1927 |
| Pneumonia | 0.7248 | 0.0642 | 0.0636 | 0.0639 |
| Pneumothorax | 0.8815 | 0.2656 | 0.6691 | 0.3802 |
| **Macro Avg** | **0.8162** | **0.2450** | **0.4963** | **0.2961** |
Overall (Exact Match) Accuracy: 0.2752

### Limitations
**Class Imbalance** The model still suffers from the class imbalance issue despite the loss function and weights set up to remedy. 

**Class Recall** Currently the threshold is set up at 0.5, to which the scope of sucess in limiting false negative is limited for many classes, it would be good to look to tune this (or tune per class) alongside providing proper disclaimers in downstream usage