# DenseNet-121 for NIH Chest X-ray Classification
**Author:** Layth Marabeh  
**Model Info:** DenseNet-121 (torchvision, ImageNet pretrained)  

---

## Overview

This directory contains the setup for fine-tuning a pre-trained DenseNet-121 model from torchvision's `models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)`. It is applied to the [NIH Chest X-ray](https://www.kaggle.com/datasets/nih-chest-xrays/data) dataset for multi-label classification of 14 thoracic pathologies. Some key design decisions include: 

- **Patient-level train/val split (80/20)** via `GroupShuffleSplit` to prevent data leakage across repeated patients
- **CLAHE preprocessing** to mathematically enhance local contrast and reveal subtle low-opacity findings
- **Dynamically Weighted BCE Loss** calculated precisely from the training distribution to aggressively penalize minority class misses
- **Automatic Mixed Precision (AMP)** to drastically reduce memory overhead and accelerate gradient computations
- **Optimal Clinical Thresholding** using Youden's J statistic to independently tune each class for maximum Recall, prioritizing patient safety
- **Grad-CAM explainability** with customized hooks to bypass in-place memory restrictions, allowing visual interpretability of feature extraction

---

## File Structure
```text
DenseNet121/              (current folder)  
├── densenet-121.ipynb    # Core unified notebook for data pipeline, training, and evaluation
├── densenet_best.pth     # Saved weights of the highest performing validation epoch
├── Data_Entry_2017.csv   # NIH dataset labels and metadata
└── train_val_list.txt    # Standardized list of images for training/validation
```

---

## Setup

### Requirements

This model is configured to dynamically route to CUDA (NVIDIA), MPS (Apple Silicon Mac), or CPU depending on the local hardware environment.

```bash
pip install torch torchvision opencv-python pillow scikit-learn pandas numpy matplotlib tqdm
```

### Data Layout

Download the dataset from Kaggle. Ensure the image root path in the notebook's configuration block points correctly to your local archive.

```text
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
└── DenseNet121/                          (current folder)                            
```

Additional confirmation can be established in the first configuration cell of the notebook:

```python
# 1. Config & Paths 
IMAGE_ROOT = "/Users/laythmarabeh/Documents/UCSD/288R/Data/archive/images"
CSV_FILE = "../data/Data_Entry_2017.csv"
SPLIT_FILE = "../data/train_val_list.txt"
```

---

## Execution Workflow
### Hyperparameter Tuning

Unlike Optuna search grids, hyperparameters for this baseline were established empirically for stable convergence on a 224x224 input space.

```python 
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2 # (via AdamW)
BATCH_SIZE = 16
IMG_SIZE = 224
```

### Full Dataset Loading and Training

---

#### Preprocessing

As established in the custom `NIHDataset` class, all training images follow this specific pipeline to preserve anatomical accuracy:
| Step | Class / Transform | Purpose |
|---|---|---|
| 1 | `LetterboxPad` | Pads to a square with black borders to prevent stretching and distortion of the cardiac silhouette |
| 2 | `Grayscale Conversion` | Forces raw 3-channel input to an explicit 1-channel matrix |
| 3 | `ApplyCLAHE` | "Contrast-Limited Adaptive Histogram Equalization" (clip=2.0, tile=8×8) applied to the single channel to enhance localized textures |
| 4 | `RepeatChannels` | Copies the perfectly enhanced grayscale channel 3 times to satisfy ResNet/DenseNet expected RGB input |
| 5 | `RandomRotation(3) & Flip` | Simulates slight patient positioning variation (applied to training set only) |
| 6 | `ToTensor` | Converts image to PyTorch tensor format |
| 7 | `Normalize` | ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) — compatible with pretrained weights |

---

#### Full Execution

All code is contained sequentially within `densenet-121.ipynb`. Execution simply requires running the cells top-to-bottom. The pipeline automatically maps the image paths, instantiates the data loaders, and triggers the PyTorch training loop. 

---

#### Additional Architecture
Training proceeds for up to 20 epochs with dynamic **Early Stopping** (patience=5), monitoring the validation set's Macro AUC. The model's state dictionary is automatically preserved as `densenet_best.pth` whenever a new peak AUC is achieved. 

We utilize **AdamW** for our optimizer to explicitly decouple weight decay from the gradient updates, and a custom **Weighted BCEWithLogitsLoss()** to handle the severe dataset imbalance.

#### Result Aggregation
Upon completion of training, the notebook generates independent optimal decision thresholds for each disease using Youden's J statistic. 

The final cell blocks execute the customized **Grad-CAM (Gradient Class Activation Mapping)**. The standard PyTorch hook architecture was rewritten dynamically (`custom_forward`) to bypass DenseNet's hardcoded in-place ReLU operations, allowing flawless heatmap generation without autograd graph destruction.

---

## Results

*Note: Metrics below reflect performance using class-specific optimized thresholds to prioritize patient triage safety.*

| Class | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Atelectasis | 0.8088 | 0.2152 | 0.7217 | 0.3315 |
| Cardiomegaly | 0.8965 | 0.0814 | 0.7738 | 0.1474 |
| Consolidation | 0.7886 | 0.0831 | 0.6982 | 0.1485 |
| Edema | 0.8938 | 0.0662 | 0.8045 | 0.1223 |
| Effusion | 0.8836 | 0.3457 | 0.7851 | 0.4800 |
| Emphysema | 0.8839 | 0.1558 | 0.5827 | 0.2458 |
| Fibrosis | 0.7997 | 0.0592 | 0.5294 | 0.1066 |
| Hernia | 0.9513 | 0.0842 | 0.8421 | 0.1531 |
| Infiltration | 0.7202 | 0.2522 | 0.6692 | 0.3663 |
| Mass | 0.8223 | 0.1151 | 0.7447 | 0.1993 |
| Nodule | 0.7367 | 0.1201 | 0.5880 | 0.1994 |
| Pleural_Thickening | 0.7850 | 0.0934 | 0.5289 | 0.1587 |
| Pneumonia | 0.7340 | 0.0204 | 0.5870 | 0.0394 |
| Pneumothorax | 0.8338 | 0.1545 | 0.6025 | 0.2459 |
| **MACRO AVG** | **0.8242** | **0.1319** | **0.6756** | **0.2103** |
| Overall (Exact Match) Accuracy: | 0.2457 | | | |

### Limitations

**Threshold Tradeoffs:** By aggressively tuning thresholds to maximize Recall (minimizing dangerous false negatives), the model generates a high volume of false positives. While appropriate for a triage screening tool, the resulting Macro Precision (13.19%) is too low for standalone, unmonitored diagnostic use. 

**Shortcut Learning (Spurious Correlations):** Visual interpretability analysis via Grad-CAM revealed that for specific pathologies (such as 'Mass'), the DenseNet architecture occasionally anchors its predictions on peripheral background artifacts and patient positioning rather than true pulmonary structures. 

**Patient Metadata:** With the current iteration, patient metadata (Age, Gender, View Position) is vectorized but not concatenated into the DenseNet classifier block. It is likely that integrating these demographic and technical priors would significantly boost predictive accuracy.