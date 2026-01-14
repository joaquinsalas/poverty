# effb3_alphaearth_infer_emissions.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"            # pick your GPU
os.environ["CODECARBON_SAVE_TO_API"] = "false"      # no dashboard posts
os.environ["CODECARBON_SAVE_TO_FILE"] = "false"     # avoid CC auto-CSV

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.backends import cudnn
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
import rasterio
import csv

# ---- CodeCarbon (OFFLINE tracker) ----
try:
    from codecarbon import OfflineEmissionsTracker
except ImportError:
    OfflineEmissionsTracker = None
    print("WARNING: codecarbon not installed. `pip install codecarbon` to record emissions.")

# torchvision EfficientNet-B3
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# ----------------------------
# Config
# ----------------------------
IMG_DIR   = "/mnt/data-r1/data/alphaEarth"
LABEL_CSV = "../data/ensemble_inferences_calidad_vivienda_2020.csv"
MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "effb3_alphaEarth_regressor.pth")  # checkpoint path
R2_CSV    = "../data/effb3_alphaEarth_test_r2.csv"
EMISSIONS_CSV = "../data/emissions/effb3_alphaEarth_CO2_emissions_per_item_grams_gpu_infer.csv"

SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 4

# Only evaluate; emissions settings
REPETITIONS = 3
COUNTRY_ISO = "MEX"
MEASURE_POWER_SECS = 1

# Splits (solo para construir el test set)
TRAIN_SPLIT = 0.50
VAL_SPLIT   = 0.20

D_INPUT  = 64
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cudnn.benchmark = True

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_test_r2_to_csv(path, r2_value):
    write_header = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["test_r2"])
        w.writerow([r2_value])

# ----------------------------
# Dataset
# ----------------------------
class AlphaEarthDataset(Dataset):
    """
    Needs:
      - 'codigo' : filename stem ({IMG_DIR}/{codigo}.tif)
      - 'target' or prediction_01..prediction_30 (averaged)
    Each .tif is (C,H,W) with C==64; we resize to (64, 224, 224).
    """
    def __init__(self, img_dir, df):
        self.img_dir = img_dir
        self.df = df.reset_index(drop=True)
        if "target" not in self.df.columns:
            pred_cols = [f"prediction_{i:02d}" for i in range(1, 31)]
            self.df["target"] = self.df[pred_cols].mean(axis=1)

    def __len__(self):
        return len(self.df)

    def _load_and_resize(self, codigo):
        path = os.path.join(self.img_dir, f"{codigo}.tif")
        with rasterio.open(path) as src:
            arr = src.read()  # (C,H,W)
        if arr.shape[0] != D_INPUT:
            raise ValueError(f"{path} has {arr.shape[0]} bands; expected {D_INPUT}.")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        x = torch.from_numpy(arr).unsqueeze(0)  # (1,C,H,W)
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        return x.squeeze(0)  # (C,224,224)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        codigo = str(row["codigo"])
        x = self._load_and_resize(codigo)
        y = torch.tensor(float(row["target"]), dtype=torch.float32)
        return x, y

# ---------- filter to existing 64-band tifs
def filter_existing_samples(img_dir, df):
    keep_rows = []
    missing, wrong_bands, unreadable = 0, 0, 0
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Scanning .tif files"):
        codigo = str(getattr(row, "codigo"))
        path = os.path.join(img_dir, f"{codigo}.tif")
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            with rasterio.open(path) as src:
                if src.count != D_INPUT:
                    wrong_bands += 1
                    continue
        except Exception:
            unreadable += 1
            continue
        keep_rows.append(row)
    fdf = pd.DataFrame(keep_rows, columns=df.columns)
    stats = {"kept": len(fdf), "missing": missing, "wrong_bands": wrong_bands, "unreadable": unreadable}
    return fdf, stats

# ----------------------------
# Model: EfficientNet-B3 adapted for 64ch regression
# ----------------------------
class EffB3Regressor(nn.Module):
    def __init__(self, in_ch=D_INPUT):
        super().__init__()
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        # Adapt first conv 3->64
        first_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            w = first_conv.weight  # (out,3,kh,kw)
            w_mean = w.mean(dim=1, keepdim=True)  # (out,1,kh,kw)
            new_conv.weight.copy_(w_mean.repeat(1, in_ch, 1, 1))
        self.backbone.features[0][0] = new_conv

        # Replace classifier head for regression (1536 -> 1)
        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_feats, 1)

    def forward(self, x):
        out = self.backbone(x)  # (B,1)
        return out.squeeze(1)

# ----------------------------
# Eval helpers
# ----------------------------
@torch.no_grad()
def evaluate_r2(model, loader, device, phase="Testing"):
    model.eval()
    preds_all, y_all = [], []
    for xb, yb in tqdm(loader, desc=phase, leave=False):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        preds = model(xb).view(-1)
        preds_all.append(preds.detach().cpu().numpy())
        y_all.append(yb.detach().cpu().numpy())
    preds_all = np.concatenate(preds_all)
    y_all = np.concatenate(y_all)
    return float(r2_score(y_all, preds_all))

@torch.no_grad()
def measure_emissions(model, loader, device, n_items):
    """
    Repeat full test passes to measure CO2e with CodeCarbon.
    Returns median/mean grams CO2e per item.
    """
    grams_samples = []
    if OfflineEmissionsTracker is None:
        grams_samples = [np.nan] * REPETITIONS
    else:
        for _ in range(REPETITIONS):
            tracker = OfflineEmissionsTracker(
                country_iso_code=COUNTRY_ISO,
                measure_power_secs=MEASURE_POWER_SECS,
                save_to_file=False,
                log_level="critical",
                tracking_mode="process"
            )
            tracker.start()
            for xb, _ in tqdm(loader, desc="Measuring emissions", leave=False):
                xb = xb.to(device, non_blocking=True)
                _ = model(xb)
            emissions_kg = tracker.stop()  # kg CO2e
            grams_per_item = (np.nan if emissions_kg is None
                              else float(emissions_kg) * 1000.0 / max(1, n_items))
            grams_samples.append(grams_per_item)

    return {
        "emissions_g_per_item_median": float(np.nanmedian(grams_samples)),
        "emissions_g_per_item_mean":   float(np.nanmean(grams_samples)),
    }

# ----------------------------
# Main (inference-only + emissions)
# ----------------------------
def main():
    set_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading CSV labels…")
    data = pd.read_csv(LABEL_CSV)
    if "codigo" not in data.columns:
        raise ValueError("'codigo' column not found in label CSV.")
    data["codigo"] = data["codigo"].astype(str)

    # Keep rows whose {codigo}.tif exists and has 64 bands
    data, stats = filter_existing_samples(IMG_DIR, data)
    print(f"Samples kept: {stats['kept']} | missing: {stats['missing']} | wrong_bands: {stats['wrong_bands']} | unreadable: {stats['unreadable']}")
    if len(data) == 0:
        raise RuntimeError("No valid samples after filtering. Check IMG_DIR or filenames.")

    dataset = AlphaEarthDataset(IMG_DIR, data)

    # deterministic split (only to build test set)
    n_total = len(dataset)
    n_train = int(TRAIN_SPLIT * n_total)
    n_val   = int(VAL_SPLIT * n_total)
    n_test  = n_total - n_train - n_val
    gen = torch.Generator().manual_seed(SEED)
    _, _, test_set = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = EffB3Regressor(in_ch=D_INPUT).to(DEVICE)

    # Load existing parameters (no training)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    print(f"Loaded model weights from {MODEL_PATH}")

    # 1) Accuracy (untracked)
    test_r2 = evaluate_r2(model, test_loader, DEVICE, phase="Testing")
    print(f"► Test R² (inference-only) = {test_r2:.4f} on {len(test_set)} items")
    save_test_r2_to_csv(R2_CSV, test_r2)

    # 2) Emissions measurement (repeat full test pass)
    metrics = measure_emissions(model, test_loader, DEVICE, n_items=len(test_set))

    # Save emissions CSV
    os.makedirs(os.path.dirname(EMISSIONS_CSV), exist_ok=True)
    pd.DataFrame([{
        "n_items": len(test_set),
        "repetitions": REPETITIONS,
        "emissions_g_per_item_median": metrics["emissions_g_per_item_median"],
        "emissions_g_per_item_mean": metrics["emissions_g_per_item_mean"],
        "test_r2": test_r2
    }]).to_csv(EMISSIONS_CSV, index=False)

    print(f"Per-item emissions (grams, averaged over test set) saved to: {EMISSIONS_CSV}")

if __name__ == "__main__":
    main()
