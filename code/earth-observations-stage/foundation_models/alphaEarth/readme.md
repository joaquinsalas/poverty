# AlphaEarth (64-band embeddings) → CONEVAL regression (S5)

This folder contains scripts that work with **AlphaEarth / Google Satellite Embedding V1** patches exported as **64-band GeoTIFFs**.  
Each GeoTIFF is read as `(C,H,W)` with `C=64` and reshaped into a **token sequence** `(L,64)` where `L=H×W`. An **S5** backbone (`S5Block`) is used to regress a scalar target defined as the mean of `prediction_01..prediction_30` from the CPV-2020 ensemble table.

---

## Scripts

### `download_GEE_embeddings.py`
Downloads **annual embedding tiles** from Google Earth Engine and saves them as GeoTIFFs.

- Uses EE collection: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- Builds a 960×960 m ROI around each mesh centroid (`pixel_size_m=10`, `half_size_pixels=48`)
- Merges the grid shapefile with the reference CSV to ensure `CODIGO`/`codigo` alignment
- Exports one file per mesh cell to: `/mnt/data-r1/data/alphaEarth/{CODIGO}.tif`

:contentReference[oaicite:0]{index=0}

---

### `sentinel2coneval_s5_AE_64b.py`
Trains an **S5-based regressor** over variable-length 64-dim token sequences.

- Loads `{IMG_DIR}/{codigo}.tif` and reshapes to `(H*W, 64)`
- Filters to existing readable files with exactly **64 bands**
- Deterministic dataset split: train=0.50, val=0.20, test=rest (`SEED=42`)
- Variable-length batching via padding + `lengths` mask (`collate_varlen`)
- Model: Linear encoder (64→512) + `N_LAYERS=3` S5 blocks + masked mean pooling + linear head
- Training: AdamW + cosine LR schedule + early stopping on **validation R²**
- Saves checkpoint to `../models/s5_alphaEarth_regressor.pth`
- Appends test R² to `../data/s5_alphaEarth_test_r2.csv`

:contentReference[oaicite:1]{index=1}

---

### `sentinel2coneval_s5_AE_64b_inference.py`
Inference-only evaluation + **per-item CO₂ emissions** for the trained S5 model.

- Rebuilds the same deterministic **test split** as training (`SEED=42`)
- Loads checkpoint from `../models/s5_alphaEarth_regressor.pth`
- Computes test R² (inference-only) and logs it to `../data/s5_alphaEarth_test_r2.csv`
- Measures inference emissions with CodeCarbon `OfflineEmissionsTracker`:
  - repeats full-test inference `REPETITIONS=3`
  - reports **median/mean grams CO₂e per item**
- Writes emissions CSV to:
  `../data/emissions/s5_alphaEarth_CO2_emissions_per_item_grams_gpu_infer.csv`

:contentReference[oaicite:2]{index=2}

---

### `sentinel2coneval_s5_AE_64b_shap.py`
Computes **SHAP attributions** for the S5 regressor and aggregates to **channel importance**.

- Loads the trained checkpoint `../models/s5_alphaEarth_regressor.pth`
- Builds a small background set (`N_BACKGROUND=8`) and explain set (`N_EXPLAIN=5`)
- Uses `shap.GradientExplainer` over padded sequences
- Saves:
  - `../data/shap_s5_alphaEarth/shap_values_explain.npy`  (N, L, 64)
  - `../data/shap_s5_alphaEarth/explain_metadata.csv`
  - `../data/shap_s5_alphaEarth/channel_importance.csv` (mean |SHAP| per channel)
  - `../data/shap_s5_alphaEarth/figures/channel_importance.png`
- Optional compute cap: `MAX_TOKENS=512` (subsamples/pads to a fixed token budget per image)

:contentReference[oaicite:3]{index=3}

---

## Repo layout (expected)

repo/
data/
ensemble_inferences_calidad_vivienda_2020.csv
data/emissions/ (created)
data/shap_s5_alphaEarth/ (created)
models/ (created)
INEGI_CPV2020_n9/
INEGI_CPV2020_n9_.csv
INEGI_CPV2020_n9_.shp
src/
download_GEE_embeddings.py
sentinel2coneval_s5_AE_64b.py
sentinel2coneval_s5_AE_64b_inference.py
sentinel2coneval_s5_AE_64b_shap.py

---

## Quick run

From `repo/src/`:

```bash
python download_GEE_embeddings.py
python sentinel2coneval_s5_AE_64b.py
python sentinel2coneval_s5_AE_64b_inference.py
python sentinel2coneval_s5_AE_64b_shap.py
