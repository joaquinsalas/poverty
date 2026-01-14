---
title: Sentinel-2 → Multidimensional poverty model regression with Vision Transformers (ViT)
description: Mesh-level poverty regression using Sentinel-2 imagery and standard Vision Transformer baselines.
---

# Sentinel-2 → CONEVAL regression (ViT)

This folder contains scripts to train and evaluate **Vision Transformer (ViT) regression models** for predicting **mesh-level poverty indicators** from **Sentinel-2A image patches** at 469 m resolution.  
The models are supervised, end-to-end ViT baselines without foundation-model pretraining beyond standard ImageNet initialization.

Targets are derived from CPV-2020 ensemble predictions aggregated per mesh cell.

---

## Scripts

### `sentinel2coneval_ViT_12b.py`
Trains a **ViT-Base (patch16, 224)** regressor using **12 Sentinel-2 spectral bands**.

- Sentinel-2 reflectance scaling applied (`× 5e-5`)
- Last band duplicated to match ViT input channel requirements (13 channels)
- Images resized to 224×224
- Target defined as the mean of `prediction_01 … prediction_30`
- Train/validation/test split: **50 / 20 / 30**
- Early stopping based on validation R²
- Saves best checkpoint and training history

---

### `sentinel2coneval_ViT_12b_inference.py`
Runs **inference and evaluation** using a trained 12-band ViT model.

- Loads saved checkpoint
- Computes test-set R²
- Writes test R² to CSV for reproducibility

---

### `sentinel2coneval_ViT_22b.py`
Trains a **ViT-Base regressor** using **12 Sentinel-2 bands plus spectral indices** (22 input channels).

Spectral indices include:
- NDVI, EVI, NDWI, NDBI
- SAVI, NBR, EVI2, MSAVI
- NMDI, NDI45, SI

The training pipeline mirrors the 12-band setup, enabling direct comparison between raw-band and index-augmented inputs.

---

### `sentinel2coneval_ViT_22b_inference.py`
Inference-only evaluation for the 22-band ViT model.

- Loads trained checkpoint
- Computes test-set R²
- Saves evaluation results to CSV

---

## Data assumptions

- Sentinel-2 image patches stored as GeoTIFFs:  
  `{IMAGE_DIR}/{codigo}.tif`
- Each patch contains **12 spectral bands**
- Label CSV includes:
  - `codigo` (image identifier)
  - `prediction_01 … prediction_30` (ensemble predictions, averaged internally)

---

## Outputs

Typical outputs include:

- `models/model_sentinel2coneval_vit_*.pth` — trained ViT checkpoints  
- `models/history_sentinel2coneval_vit_*.pth` — training/validation loss history  
- `data/sentinel2coneval_vit_*_r2_test_*.csv` — test-set R² metrics  

---




