# Multi-Label Thoracic Pathology Classification: ResNet, DenseNet, & MedViT

This repository contains the implementation and comparative study of three deep learning paradigms evaluated on the NIH Chest X-ray14 dataset. This project investigates the efficacy of traditional CNNs against hybrid Vision Transformers for clinical triage.

## 1. Environment Setup

We recommend using **Conda** for environment management to ensure dependency stability.

### **Create and Activate Environment**
```bash
conda create -n capst_env python=3.10 -y
conda activate capst_env
Install Dependencies
pip install torch torchvision timm torchmetrics pandas numpy scikit-learn opencv-python Pillow matplotlib seaborn plotly tqdm ipython
```

## 2. Repository Structure

```text
MedVIT/
├── data/
│   ├── images/              # 112,120 PNG X-ray files (1024x1024)
│   └── Data_Entry_2017.csv  # NIH Metadata and NLP-mined labels
├── experiments/             # Checkpoint storage for .pth files
├── dataset.py               # CLAHE, Letterbox Padding, and Dataset logic
├── data_loader.py           # GroupShuffleSplit (Patient-level)
├── model.py                 # MedViT + Metadata Gating Architecture
├── train.py                 # Training loop (OneCycleLR, AMP, AdamW)
└── visualize_results.py     # Metrics (AUC/PR), Grad-CAM, and Attention Rollout
```

## 3. Data Preparation

- Images: Place all NIH .png files in the /data/images/ directory.

- Leakage Prevention: Our pipeline uses a GroupShuffleSplit keyed to Patient ID. This ensures that no images from a single patient exist in both the training and test sets simultaneously.

- Preprocessing: The code automatically applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance local contrast for subtle pathologies.

## 4. Running the Code
### Training
To train the MedViT model with Metadata Gating:

```bash
python train.py --batch_size 16 --lr 1e-4 --epochs 30 --model medvit
```

### Evaluation & Visualization
To generate performance metrics and PR curves, run the visualization script against your best saved checkpoint:

```bash
python visualize_results.py --model_path "experiments/YOUR_RUN_FOLDER/best_medvit_model.pth"
```

## 5. Technical Highlights

- Hybrid Architecture: MedViT utilizes Shifted Window Attention to capture global anatomical context.

- Metadata Gating: Dynamically fuses image features with Patient Age, Sex, and View Position.

- Imbalance Handling: Implements Weighted Binary Cross-Entropy (BCE) loss to penalize false negatives in rare classes.

- Efficiency: Accelerated via Automatic Mixed Precision (AMP) to optimize GPU memory.

## 6. Troubleshooting

- Windows Users: If your file paths contain spaces (e.g., OneDrive), wrap your --model_path in double quotes: "C:\Users\Name\Path To Model\best.pth".

- CUDA Errors: Verify GPU availability with python -c "import torch; print(torch.cuda.is_available())".

- Memory (OOM): If you encounter Out of Memory errors, decrease the --batch_size to 8 or 4 in train.py.