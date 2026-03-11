# NIH Chest X-Ray Classification 
**Final Report:** link to final report here     
**Authors:** Layth Marabeh, Khanh Phan, Danny Xia       
**Reach Project Demo:** https://khanhphan.com/NIH-XRay-Segmentation/        
**HuggingFace Space:** https://huggingface.co/spaces/k-phantastic/TritoNapse

---

## Overview 

This repository contains our capstone project in building a multi-label chest X-ray pathology classification system that studies the setup for three different deep learning architectures on the [NIH Chest X-ray](https://www.kaggle.com/datasets/nih-chest-xrays/data) dataset from Kaggle. 

As part of our reach goal in building an tangible UI for seeing different model inferences, we created a draft of an application that utilizes the completed trained weights to predict pathologies:

**Input:** A frontal-view chest X-ray image (PNG or JPG)  
**Output:** Predicted probabilities for up to 15 thoracic pathologies (14 conditions + No Finding)

### Models

| Model | Author | Architecture | Resolution 
|---|---|---|---
| DenseNet-121 (baseline)| Layth Marabeh | CNN, dense connections | 224×224 
| [ResNet-50](ResNet50/README.md) | Khanh Phan | CNN, residual skip connections | 512×512 
| MaxViT (modified) | Danny Xia | Vision transformer + metadata fusion | 224×224 

Each model was independently developed and trained; top to bottom being our anticipated worst to best in performance. All share similar image preprocessing and patient stratified validation/test set methodologies. 

---

## Repository Structure

```
Root
│
├── ResNet50/                     # ResNet-50 training pipeline (Khanh)

```

---

## Setup

---

## Dataset

---

## Exploratory Data Analysis

---

## Workflow

---

## Results

--- 

## References
