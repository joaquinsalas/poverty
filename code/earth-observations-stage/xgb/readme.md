# Sentinel-2 → CONEVAL regression (XGBoost / XGB)

This folder contains scripts to train and evaluate **XGBoost regressors** for predicting mesh-level targets from **Sentinel-2A image patches**.  
Each sample loads a multispectral GeoTIFF, applies a fixed crop/downsample, **flattens the patch into a feature vector**, standardizes features with `StandardScaler`, then fits an `XGBRegressor` using GPU acceleration.

The target is defined as the mean of CPV-2020 ensemble predictions (`prediction_01 … prediction_30`) per mesh cell.

---

## Scripts

### `sentinel2coneval_xgb_12b.py`
Trains an **XGBRegressor** using **raw Sentinel-2 bands only**.

- Loads `{IMG_DIR}/{codigo}.tif`, scales reflectance by `5e-5`
- Applies a fixed crop/downsample: `img[:, 24:72:2, 24:72:2]`
- Flattens to 1D features (`feat.ravel()`)
- Train/test split: **50 / 50** with `random_state=42`
- Standardizes with `StandardScaler` (fit on train only)
- Hyperparameter search via `RandomizedSearchCV` (GPU `tree_method="gpu_hist"`)
- Saves model, scaler, and a one-row CSV summary with test R² + best params


---

### `sentinel2coneval_xgb_12b_inference.py`
Inference-only evaluation + **CO₂ per-item emissions** for the trained 12-band XGB model.

- Rebuilds features exactly as in training (crop/downsample + flatten)
- Recreates a deterministic **test split** (`test_size=0.5`, `random_state=42`)
- Loads saved `xgb_target_12b.pkl` and `scaler_target_12b.pkl`
- Computes test R² (inference-only)
- Measures inference emissions using CodeCarbon `OfflineEmissionsTracker`
  (repeated full-test inference; reports mean/median grams CO₂e per item)
- Writes emissions CSV into `../data/emissions/`


---

### `sentinel2coneval_xgb_22b.py`
Trains an **XGBRegressor** using **raw Sentinel-2 bands plus spectral indices** (index-augmented feature stack).

- Same reflectance scaling and crop/downsample as the 12-band version
- Builds an `idx_stack` (computed from the first 6 bands) and concatenates it to the band stack
- Flattens the combined stack to 1D features
- Same 50/50 split, standardization, and GPU RandomizedSearchCV workflow
- Saves model/scaler and a CSV summary


---

### `sentinel2coneval_xgb_22b_inference.py`
Inference-only evaluation + CO₂ per-item emissions for the trained index-augmented XGB model.

- Rebuilds features to **match the training feature definition** (bands + indices)
- Deterministic test split (`random_state=42`)
- Loads trained model and scaler
- Computes test R² (inference-only)
- Measures grams CO₂e per item using CodeCarbon (offline), saved under `../data/emissions/`


---

## Data assumptions

- Sentinel-2 image patches stored as GeoTIFFs:  
  `{IMG_DIR}/{codigo}.tif`
- Metadata CSV includes:
  - `codigo` (image identifier)
  - `prediction_01 … prediction_30` (ensemble predictions; averaged into `target`)

---

## Outputs

Typical outputs include:

- `models/xgb_target_12b.pkl`, `models/scaler_target_12b.pkl`
- `models/xgb_target_23.pkl`, `models/scaler_target_23.pkl` (index-augmented variant naming in script)
- `data/xgb_12b_target_summary.csv`
- `data/xgb_23b_target_summary.csv`
- `data/emissions/xgb_12_CO2_emissions_per_item_grams_gpu_infer.csv`
- `data/emissions/xgb_23_CO2_emissions_per_item_grams_gpu_infer_rev.csv`

---

