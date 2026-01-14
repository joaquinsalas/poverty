## Sentinel-2 → housing quality poverty indicator (mesh-level) regression with S5 + DINOv3

This folder contains four scripts that train and/or evaluate a hybrid model for predicting **housing-quality** targets over the 469 m meshgrid using **Sentinel-2A** image patches. The pipeline uses a **DINOv3 Vision Transformer** backbone (frozen) to extract patch-level tokens, followed by a lightweight **S5 (state-space)** sequence head and a linear regressor.

### Description 

- **Inputs**
  - A CSV with a `codigo` column that maps each sample to a GeoTIFF: `${IMG_DIR}/{codigo}.tif`.
  - Each GeoTIFF is expected to have **12 bands** (Sentinel-2 reflectances) and is scaled by `S2_SCALE` (default `5e-5`).
  - The CSV contains:
    - `target`, or
    - `prediction_01 ... prediction_30` columns (the scripts average them to define `target`).

- **Preprocessing**
  - Converts the 12-band patch into a **3-channel image** for DINOv3:
    - either by selecting `RGB_BANDS = (3,2,1)`, or
    - by applying per-image **PCA (12→3)** if `RGB_BANDS=None`.
  - Optional per-band percentile clipping (`PERCENTILE_CLIP`) to suppress outliers.
  - Resizes to a fixed `IMG_SIZE` (multiple of 16 for ViT/16) and applies normalization using `SAT_MEAN` / `SAT_STD`.

- **Model**
  - `AutoModel.from_pretrained(DINO_PRETRAINED)` provides token embeddings.
  - Register + class tokens are removed; only patch tokens are passed to:
    - a linear projection to `D_MODEL`,
    - `N_LAYERS` of `S5Block`,
    - mean pooling,
    - final linear regression head.

- **Splits**
  - Deterministic split with seed `SEED`:
    - train: `TRAIN_SPLIT` (default 0.50)
    - val: `VAL_SPLIT` (default 0.20)
    - test: remainder

### Scripts

- `sentinel2coneval_s5_DINOv3.py`  
  Trains **S5 + DINOv3-ViT-L/16** (default) with early stopping on validation R², resumes if a checkpoint exists, and writes test R² to CSV. :contentReference[oaicite:0]{index=0}

- `sentinel2coneval_s5_DINOv3_inference.py`  
  Inference-only evaluation for the **ViT-L/16** checkpoint and measures **CO₂e per processed item** using CodeCarbon (offline tracker). :contentReference[oaicite:1]{index=1}

- `sentinel2coneval_s5_DINOv3_ViT_7B.py`  
  Same training flow as above but configured for **DINOv3-ViT-7B/16** (very large). Produces a separate checkpoint and test R² CSV. :contentReference[oaicite:2]{index=2}

- `sentinel2coneval_s5_DINOv3_ViT_7B_inference.py`  
  Inference + emissions measurement for the **ViT-7B** checkpoint. This version keeps the 7B backbone on **CPU** and runs only the small S5 head on **GPU** to reduce VRAM pressure. :contentReference[oaicite:3]{index=3}

### Outputs

- Checkpoints (examples)
  - `../models/s5_dinov3_regressor_dinov3.pth`
  - `../models/s5_dinov3_regressor_ViT_7B.pth`

- Metrics
  - `../data/s5_dinov3_test_r2.csv`
  - `../data/s5_dinov3_test_r2_ViT_7B.csv`

- Emissions (inference scripts)
  - `../data/emissions/s5_dinov3_CO2_emissions_per_item_grams_gpu_infer.csv`
  - `../data/emissions/s5_dinov3_ViT_7B_CO2_emissions_per_item_grams_gpu_infer.csv`

### Notes / knobs you can edit quickly

- `IMG_DIR`, `LABEL_CSV`, `MODEL_PATH`, `IMG_SIZE`, `RGB_BANDS` vs PCA, `S2_SCALE`
- DINOv3 model selection via `DINO_PRETRAINED`
- Compute/accuracy trade-offs: `BATCH_SIZE`, `N_LAYERS`, `D_MODEL`, `PATIENCE`

