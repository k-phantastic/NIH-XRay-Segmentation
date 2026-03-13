General Instructions for all Models:
Please adhere 100% to the repository structure for each model.
This will prevent errors in running the code and is absolutely imperative to achieve replication.

Upon cloning repository, install any remaining dependencies using our requirements.txt
pip install -r requirements.txt
Notably, an appropriate PyTorch version should be chosen relative to the system the training and inference is running on.

Download the dataset from Kaggle and organize the image folders under images/ as recommended in the Repository Structure section.
https://www.kaggle.com/datasets/nih-chest-xrays/data

DenseNet-121 Instructions:
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
'''  
- All activities related to the this model is included in densenet-121.ipynb. Run the cells in order.
- This will start training and produce a best_densenet_model.pth file in the Densenet folder.
- Once this file is obtained move on to densenet_metrics.ipynb to evaluate the model and produce densenet_metrics.
- These cells should also be run in order. All metrics related to the model will be printed in the output of the cells.
- Should training fail (this should not be the case), i have included the best path that was produced during training 
  in my personal machine inside of the densenet folder

Resnet Instructions:
ResNet50/               (current folder)   
├── config.py           # All hyperparameters, file paths, and training toggles
├── data_loader.py      # NIH8Dataset class for the dataset, preprocessing transforms
├── train_model.py      # Full training loop with validation, checkpointing, and test evaluation
├── tuning.py           # Optuna hyperparameter search (ran before train_model.py)
├── utils.py            # Metrics, data splitting, GradCAM, visualization, patience
├── resnet.ipynb        # Notebook for post-training analysis (ROC curves, Grad-CAM, metrics)
'''
- (prerequisite being installing necessary in our requirements.txt)
- File path, hyperparameters, and additional model settings can be found and set up in config.py (current state of file is equivalent to that used in creating results)
- Run python tuning.py for hyperparameter search if desired
- Run python train_model.py for complete dataset loading and model training.
- Results and visualizations can be seen in resnet.ipynb

MedVit Insstructions:
Repository Structure

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

## Data Preparation

- Images: Place all NIH .png files in the /data/images/ directory.

- Leakage Prevention: Our pipeline uses a GroupShuffleSplit keyed to Patient ID. This ensures that no images from a single patient exist in both the training and test sets simultaneously.

- Preprocessing: The code automatically applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance local contrast for subtle pathologies.

## Running the Code
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

## Technical Highlights

- Hybrid Architecture: MedViT utilizes Shifted Window Attention to capture global anatomical context.

- Metadata Gating: Dynamically fuses image features with Patient Age, Sex, and View Position.

- Imbalance Handling: Implements Weighted Binary Cross-Entropy (BCE) loss to penalize false negatives in rare classes.

- Efficiency: Accelerated via Automatic Mixed Precision (AMP) to optimize GPU memory.

## Troubleshooting

- Windows Users: If your file paths contain spaces (e.g., OneDrive), wrap your --model_path in double quotes: "C:\Users\Name\Path To Model\best.pth".

- CUDA Errors: Verify GPU availability with python -c "import torch; print(torch.cuda.is_available())".

- Memory (OOM): If you encounter Out of Memory errors, decrease the --batch_size to 8 or 4 in train.py.