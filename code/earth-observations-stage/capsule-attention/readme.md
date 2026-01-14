# Sentinel-2 → CONEVAL regression (CAN)

This folder contains scripts to train and evaluate **Capsule Attention Network (CAN)** regressors for predicting **mesh-level poverty indicators** from **Sentinel-2A image patches** at 469 m resolution.  
The models are fully supervised and operate directly on multispectral inputs, with optional spectral-index augmentation.

Targets are defined as the mean of CPV-2020 ensemble predictions (`prediction_01 … prediction_30`) per mesh cell.

---

## Scripts

### `sentinel2coneval_CAN_12b.py`
Capsule Attention baseline using **Sentinel-2 raw bands only**.

- Input: 12 Sentinel-2 bands, scaled by `5e-5`
- Last band duplicated → **13 channels**
- Robust per-band percentile clipping and rescaling to `[0,1]`
- Architecture:
  - CNN stem
  - Primary Capsules (convolutional)
  - Multi-head self-attention over capsules
  - Capsule pooling + MLP regression head
- Train/validation/test split: **50 / 20 / 30**
- OneCycleLR scheduling and early stopping on validation R²
- Saves best model, training history, and test-set R²

:contentReference[oaicite:0]{index=0}

---

### `sentinel2coneval_CAN_22b.py`
Capsule Attention model using **raw Sentinel-2 bands plus spectral indices**.

- Input construction:
  - Original Sentinel-2 bands
  - Optional band duplication
  - **10 spectral indices** appended:
    - NDVI, EVI, NDWI, NDBI
    - SAVI, NBR, EVI2, MSAVI
    - NDI45, SI
- Robust per-band clipping and scaling applied after feature construction
- Input channel count inferred automatically from data
- Same CAN architecture and training protocol as the 12-band variant
- Enables direct comparison between raw-band and index-augmented inputs

:contentReference[oaicite:1]{index=1}

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

- `models/best_model_CAN_*.pth` — trained CAN checkpoints  
- `models/training_history_CAN_*.pth` — training/validation loss history  
- `data/test_results_CAN_*_r2.csv` — test-set R² metrics  

---


