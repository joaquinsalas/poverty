# 3_dinov3_s5_infer_emissions.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"            # pick your GPU
os.environ["CODECARBON_SAVE_TO_API"] = "false"      # no dashboard posts
os.environ["CODECARBON_SAVE_TO_FILE"] = "false"     # avoid CC auto-CSV

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim  # (unused, but kept for compatibility)
from torch.utils.data import DataLoader, Dataset, random_split
from torch.backends import cudnn
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
import rasterio
import csv

from s5 import S5Block
from transformers import AutoModel
from sklearn.decomposition import PCA

# ---- CodeCarbon (OFFLINE tracker) ----
try:
    from codecarbon import OfflineEmissionsTracker
except ImportError:
    OfflineEmissionsTracker = None
    print("WARNING: codecarbon not installed. `pip install codecarbon` to record emissions.")

# ----------------------------
# Config
# ----------------------------
IMG_DIR   = '/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A/'
LABEL_CSV = "/mnt/data-r1/JoaquinSalas/Documents/informs/research/2023.11.16census/2024.07.29malla470/data/ensemble_inferences_calidad_vivienda_2020.csv"
MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "s5_dinov3_regressor_dinov3.pth")  # checkpoint path
R2_CSV = "../data/s5_dinov3_test_r2.csv"
EMISSIONS_CSV = "../data/emissions/s5_dinov3_CO2_emissions_per_item_grams_gpu_infer.csv"

SEED = 42
BATCH_SIZE = 16          # imgs resized inside model forward
NUM_WORKERS = 4

# Emissions measurement
REPETITIONS = 3
COUNTRY_ISO = "MEX"
MEASURE_POWER_SECS = 1

# deterministic split to build test set only
TRAIN_SPLIT = 0.50
VAL_SPLIT   = 0.20

# Sentinel-2A config
D_INPUT = 12
S2_SCALE = 5e-5

# DINOv3 config
DINO_PRETRAINED = "facebook/dinov3-vitl16-pretrain-sat493m"
FREEZE_DINO = True
IMG_SIZE = 256  # must be multiple of 16 (ViT/16)
RGB_BANDS = (3, 2, 1)  # set to None to use PCA(12->3)
PERCENTILE_CLIP = (0.5, 99.5)

# sat stats for normalization
SAT_MEAN = (0.430, 0.411, 0.296)
SAT_STD  = (0.213, 0.156, 0.143)

# S5 head
D_MODEL = 512
N_LAYERS = 3
DROPOUT = 0.1
PRENORM = True

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

def _clip_percentiles(x, p_lo=0.5, p_hi=99.5):
    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)
    if hi <= lo:
        return x
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return x

# ----------------------------
# Dataset
# ----------------------------
class Sentinel2ADataset(Dataset):
    """
    CSV needs:
      - 'codigo' : filename stem ({IMG_DIR}/{codigo}.tif)
      - 'target' or 30 columns prediction_01..prediction_30 (averaged to target)

    Each .tif is (C,H,W) with C==12. We convert to 3-ch image:
      - If RGB_BANDS is set: select those 3 bands.
      - Else: per-image PCA(12->3) over flattened pixels.
    Output x: float32 torch tensor (3,H,W) in [0,1].
    """
    def __init__(self, img_dir, df):
        self.img_dir = img_dir
        self.df = df.reset_index(drop=True)
        if "target" not in self.df.columns:
            pred_cols = [f"prediction_{i:02d}" for i in range(1, 31)]
            self.df["target"] = self.df[pred_cols].mean(axis=1)

    def __len__(self):
        return len(self.df)

    def _load_tif(self, codigo):
        path = os.path.join(self.img_dir, f"{codigo}.tif")
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32) * S2_SCALE  # (C,H,W)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.shape[0] != D_INPUT:
            raise ValueError(f"{path} has {arr.shape[0]} bands; expected {D_INPUT}.")
        return arr

    def _to_three_channels(self, arr_chw):
        C, H, W = arr_chw.shape
        if PERCENTILE_CLIP is not None:
            for c in range(C):
                arr_chw[c] = _clip_percentiles(arr_chw[c], *PERCENTILE_CLIP)

        if RGB_BANDS is not None:
            rgb = arr_chw[list(RGB_BANDS)]  # (3,H,W)
            return rgb.astype(np.float32)

        # PCA(12->3) over pixels
        X = arr_chw.reshape(C, -1).T  # (H*W, 12)
        mu = X.mean(0, keepdims=True)
        sd = X.std(0, keepdims=True) + 1e-6
        Xn = (X - mu) / sd
        pca = PCA(n_components=3, svd_solver="randomized")
        Y = pca.fit_transform(Xn)  # (H*W,3)
        for k in range(3):
            y = Y[:, k]
            lo, hi = np.percentile(y, 1), np.percentile(y, 99)
            if hi > lo:
                y = (np.clip(y, lo, hi) - lo) / (hi - lo)
            else:
                y = (y - y.min()) / (y.max() - y.min() + 1e-8)
            Y[:, k] = y
        rgb = Y.T.reshape(3, H, W).astype(np.float32)
        return rgb

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        codigo = str(row["codigo"])
        arr = self._load_tif(codigo)                  # (12,H,W)
        img3 = self._to_three_channels(arr)           # (3,H,W) in [0,1]
        y = torch.tensor(float(row["target"]), dtype=torch.float32)
        x = torch.from_numpy(img3)
        return x, y

# ---------- filter to existing 12-band tifs
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

# ---------- collate: keep variable H,W; model will resize
def collate_noresize(batch):
    xs, ys = zip(*batch)  # xs: list of (3,H,W) CPU tensors in [0,1]
    return list(xs), torch.stack(ys)

# ----------------------------
# DINOv3 + S5 Regressor (inference)
# ----------------------------
class DinoS5Regressor(nn.Module):
    def __init__(self, dino_name=DINO_PRETRAINED, d_model=D_MODEL, n_layers=N_LAYERS,
                 d_output=1, dropout=DROPOUT, prenorm=PRENORM, freeze_dino=FREEZE_DINO,
                 sat_mean=SAT_MEAN, sat_std=SAT_STD, img_size=IMG_SIZE, hf_token=None):
        super().__init__()
        kw = {}
        if hf_token is None:
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            kw["token"] = hf_token
        self.dino = AutoModel.from_pretrained(dino_name, **kw)

        if freeze_dino:
            for p in self.dino.parameters():
                p.requires_grad = False

        self.img_size = img_size
        self.register_buffer("mean", torch.tensor(sat_mean).view(1,3,1,1), persistent=False)
        self.register_buffer("std",  torch.tensor(sat_std ).view(1,3,1,1), persistent=False)

        d_dino = self.dino.config.hidden_size
        self.prenorm = prenorm
        self.enc = nn.Linear(d_dino, d_model)
        self.s5_layers = nn.ModuleList([S5Block(dim=d_model, state_dim=d_model, bidir=False)
                                        for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, d_output)
        self.n_registers = getattr(self.dino.config, "num_register_tokens", 4)

    def _preprocess(self, x):
        # x: list of (3,H,W) CPU tensors in [0,1]
        if isinstance(x, list):
            x = torch.stack(x, dim=0)  # (B,3,H,W)
        dev = self.mean.device
        x = x.to(dev, non_blocking=True)
        x = torch.nn.functional.interpolate(x, size=(self.img_size, self.img_size),
                                            mode="bilinear", align_corners=False)
        x = x.clamp_(0, 1)
        x = (x - self.mean) / self.std
        return x

    def forward(self, imgs_3chw_list):
        pixel_values = self._preprocess(imgs_3chw_list)
        with torch.no_grad():  # inference-only
            out = self.dino(pixel_values=pixel_values, output_hidden_states=False)
        tokens = out.last_hidden_state
        patch_tokens = tokens[:, 1 + self.n_registers:, :]
        x = self.enc(patch_tokens)
        for layer, norm, drop in zip(self.s5_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm: z = norm(z)
            z = layer(z)
            z = drop(z)
            x = x + z
            if not self.prenorm: x = norm(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)

# ----------------------------
# Eval helpers (R² + emissions)
# ----------------------------
@torch.no_grad()
def evaluate_r2(model, loader, device, phase="Testing"):
    model.eval()
    preds_all, y_all = [], []
    for xb, yb in tqdm(loader, desc=phase, leave=False):
        yb = yb.to(device)
        preds = model(xb)               # xb is list of CPU tensors; model handles device
        preds_all.append(preds.cpu().numpy())
        y_all.append(yb.cpu().numpy())
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
                _ = model(xb)  # inference only
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

    # Keep rows whose {codigo}.tif exists and has 12 bands
    data, stats = filter_existing_samples(IMG_DIR, data)
    print(f"Samples kept: {stats['kept']} | missing: {stats['missing']} | wrong_bands: {stats['wrong_bands']} | unreadable: {stats['unreadable']}")
    if len(data) == 0:
        raise RuntimeError("No valid samples after filtering. Check IMG_DIR or filenames.")

    dataset = Sentinel2ADataset(IMG_DIR, data)

    # deterministic split (only to build test set)
    n_total = len(dataset)
    n_train = int(TRAIN_SPLIT * n_total)
    n_val   = int(VAL_SPLIT * n_total)
    n_test  = n_total - n_train - n_val
    gen = torch.Generator().manual_seed(SEED)
    _, _, test_set = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_noresize)

    # Model (inference-only)
    model = DinoS5Regressor(dino_name=DINO_PRETRAINED, freeze_dino=FREEZE_DINO).to(DEVICE)

    # Load existing checkpoint (no training)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:   print(f"⚠️ Missing keys: {missing}")
    if unexpected: print(f"⚠️ Unexpected keys: {unexpected}")
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
        "test_r2": test_r2,
        "dino_pretrained": DINO_PRETRAINED,
        "img_size": IMG_SIZE
    }]).to_csv(EMISSIONS_CSV, index=False)

    print(f"Per-item emissions (grams, averaged over test set) saved to: {EMISSIONS_CSV}")

if __name__ == "__main__":
    main()
