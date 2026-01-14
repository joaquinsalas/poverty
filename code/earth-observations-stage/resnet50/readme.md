# Sentinel-2 → CONEVAL regression (ResNet-50)

This folder contains scripts to train, evaluate, and interpret ResNet-50 convolutional neural networks for predicting mesh-level poverty indicators from Sentinel-2A image patches at 469 m resolution.  
The models are trained in a supervised manner and operate on multispectral inputs, with optional spectral-index augmentation.

Targets are defined as the mean of CPV-2020 ensemble predictions (`prediction_01 … prediction_30`) per mesh cell.

---

## Scripts

### `sentinel2coneval_resnet50_12b.py`
Trains a **ResNet-50 regressor** using **12 Sentinel-2 spectral bands**.

- Radiometric scaling (`× 5e-5`)
- Last band duplicated → **13 channels** (to match ResNet stem)
- Train/validation/test split: **50 / 20 / 30** (reproducible)
- OneCycleLR scheduling with early stopping on validation R²
- Saves best model checkpoint, training history, and test metrics (R², RMSE, MAE)

---

### `sentinel2coneval_resnet50_12b_inference.py`
Inference-only evaluation for the **12-band ResNet-50**.

- Loads trained checkpoint
- Computes test-set R²
- Measures **per-item CO₂ emissions** during inference using CodeCarbon
- Saves accuracy and emissions statistics to CSV

---

### `sentinel2coneval_resnet50_12b_shap.py`
SHAP-based interpretability for the **12-band ResNet-50**.

- Gradient SHAP explanations on multispectral inputs
- Per-sample spatial SHAP maps (Σ|SHAP|)
- Global **channel importance** via mean absolute SHAP
- RGB visualizations aligned with SHAP heatmaps
- Saves SHAP arrays, CSV summaries, and figures

---

### `sentinel2coneval_resnet50_22b.py`
Trains a **ResNet-50 regressor** using **raw Sentinel-2 bands plus spectral indices**.

- Input channels:
  - 12 Sentinel-2 bands (+ optional duplication)
  - **10–11 spectral indices**, including NDVI, EVI, NDWI, NDBI, SAVI, NBR, EVI2, MSAVI, NDI45, SI (and variants)
- Same training protocol as 12-band model
- Enables direct comparison between raw-band and index-augmented representations

---

### `sentinel2coneval_resnet50_22b_inference.py`
Inference and emissions evaluation for the **22-band ResNet-50**.

- Loads trained checkpoint
- Computes test-set R²
- Measures inference-time CO₂ emissions per sample
- Writes consolidated accuracy + emissions CSV

---

### `sentinel2coneval_resnet50_22b_shap.py`
SHAP analysis for the **22-band ResNet-50**.

- Gradient SHAP with background and explain subsets
- Channel-wise importance across raw bands and indices
- Per-sample spatial explanations
- Saves SHAP tensors, channel-importance CSVs, and publication-ready figures

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

- `models/best_model_sentinel2coneval_resnet50_*.pth` — trained checkpoints  
- `models/training_history_resnet50_*.pth` — training/validation loss history  
- `data/test_results_resnet50_*_metrics.csv` — R² / RMSE / MAE  
- `data/emissions/*.csv` — per-item CO₂ emissions during inference  
- `data/shap_resnet50_*/` — SHAP values, indices, and metadata  
- `figures/*.png` — SHAP visualizations and channel-importance plots  


