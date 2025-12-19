# s5_alphaearth_infer_emissions.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"           # pick your GPU
os.environ["CODECARBON_SAVE_TO_API"] = "false"     # no dashboard posts
os.environ["CODECARBON_SAVE_TO_FILE"] = "false"    # avoid CC auto-CSV

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.backends import cudnn
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
import rasterio
from s5 import S5Block  # pip/your local s5 module
import csv

# ---- CodeCarbon (OFFLINE tracker) ----
try:
    from codecarbon import OfflineEmissionsTracker
except ImportError:
    OfflineEmissionsTracker = None
    print("WARNING: codecarbon not installed. `pip install codecarbon` to record emissions.")

# ----------------------------
# Config
# ----------------------------
IMG_DIR = "/mnt/data-r1/data/alphaEarth"   # where {codigo}.tif live
LABEL_CSV = "../data/ensemble_inferences_calidad_vivienda_2020.csv"
MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "s5_alphaEarth_regressor.pth")  # checkpoint path
R2_CSV = "../data/s5_alphaEarth_test_r2.csv"
EMISSIONS_CSV = "../data/emissions/s5_alphaEarth_CO2_emissions_per_item_grams_gpu_infer.csv"

SEED = 42
BATCH_SIZE = 32          # sequences can be long
NUM_WORKERS = 4

# Only evaluate; emissions settings
REPETITIONS = 3
COUNTRY_ISO = "MEX"
MEASURE_POWER_SECS = 1

# Splits (solo para construir el test set)
TRAIN_SPLIT = 0.50
VAL_SPLIT = 0.20

D_INPUT = 64            # channels
D_MODEL = 512           # hidden size inside S5
N_LAYERS = 3
DROPOUT = 0.1
PRENORM = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cudnn.benchmark = True  # OK with variable seq lens; just a hint

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
    CSV needs:
      - 'codigo' : filename stem ({IMG_DIR}/{codigo}.tif)
      - 'target' or 30 columns prediction_01..prediction_30 (will be averaged)
    Each .tif is (C, H, W) with C==64; we reshape -> (H*W, 64)
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
            arr = src.read()  # (C, H, W)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.shape[0] != D_INPUT:
            raise ValueError(f"{path} has {arr.shape[0]} bands; expected {D_INPUT}.")
        arr = np.transpose(arr, (1, 2, 0)).reshape(-1, D_INPUT)  # (H*W, 64)
        return arr.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        codigo = str(row["codigo"])
        x = self._load_tif(codigo)                  # (L, 64), L = H*W (variable)
        y = np.float32(row["target"])
        x = torch.from_numpy(x)                     # float32
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

# ---------- filtering to only existing 64-band tifs (any HxW)
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

# ---------- collate for variable-length sequences
def collate_varlen(batch):
    xs, ys = zip(*batch)               # xs: list of (L_i, D_INPUT)
    lengths = [x.shape[0] for x in xs]
    maxlen = max(lengths)
    D = xs[0].shape[1]
    padded = torch.zeros(len(xs), maxlen, D, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        padded[i, :x.shape[0]] = x
    y = torch.stack(ys)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return padded, y, lengths

# ----------------------------
# Model (S5 for regression)
# ----------------------------
class S5Regressor(nn.Module):
    """
    Input: (B, L, D_INPUT)
    Encoder: Linear(D_INPUT -> D_MODEL) per token
    S5: stack of S5Blocks on (B, L, D_MODEL)
    Head: masked mean pool over L, then Linear(D_MODEL -> 1)
    """
    def __init__(self, d_input=D_INPUT, d_model=D_MODEL, n_layers=N_LAYERS,
                 d_output=1, dropout=DROPOUT, prenorm=PRENORM):
        super().__init__()
        self.prenorm = prenorm
        self.encoder = nn.Linear(d_input, d_model)
        self.s5_layers = nn.ModuleList([S5Block(dim=d_model, state_dim=d_model, bidir=False)
                                        for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, d_output)

    def forward(self, x, lengths=None):
        # x: (B, L, D_INPUT)
        x = self.encoder(x)  # (B, L, D_MODEL)
        for layer, norm, drop in zip(self.s5_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z)
            z = layer(z)      # (B, L, D_MODEL)
            z = drop(z)
            x = x + z
            if not self.prenorm:
                x = norm(x)
        # masked mean over valid tokens only
        if lengths is not None:
            B, L, _ = x.shape
            mask = (torch.arange(L, device=x.device).unsqueeze(0) < lengths.unsqueeze(1))  # (B,L)
            mask = mask.unsqueeze(-1)  # (B,L,1)
            x_sum = (x * mask).sum(dim=1)                        # (B, D_MODEL)
            denom = lengths.clamp(min=1).unsqueeze(1)            # (B,1)
            x = x_sum / denom
        else:
            x = x.mean(dim=1)
        out = self.head(x).squeeze(-1)  # (B,)
        return out

# ----------------------------
# Eval helpers
# ----------------------------
@torch.no_grad()
def evaluate_r2(model, loader, device, phase="Testing"):
    model.eval()
    preds_all, y_all = [], []
    for xb, yb, lengths in tqdm(loader, desc=phase, leave=False):
        xb, yb, lengths = xb.to(device), yb.to(device), lengths.to(device)
        preds = model(xb, lengths)
        preds_all.append(preds.detach().cpu().numpy())
        y_all.append(yb.detach().cpu().numpy())
    preds_all = np.concatenate(preds_all)
    y_all = np.concatenate(y_all)
    return float(r2_score(y_all, preds_all))

@torch.no_grad()
def measure_emissions(model, loader, device, n_items):
    """
    Repite pasadas completas del test para medir CO2e con CodeCarbon.
    Devuelve mediana y media de gramos por muestra.
    """
    grams_samples = []
    if OfflineEmissionsTracker is None:
        grams_samples = [np.nan] * REPETITIONS
    else:
        for _ in range(REPETITIONS):
            tracker = OfflineEmissionsTracker(
                country_iso_code=COUNTRY_ISO,
                measure_power_secs=MEASURE_POWER_SECS,
                save_to_file=False,    # no escribir CSV propio de CodeCarbon
                log_level="critical",
                tracking_mode="process"
            )
            tracker.start()
            for xb, _, lengths in tqdm(loader, desc="Measuring emissions", leave=False):
                xb, lengths = xb.to(device), lengths.to(device)
                _ = model(xb, lengths)
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

    # Keep rows whose {codigo}.tif exists and has 64 bands (any HxW)
    data, stats = filter_existing_samples(IMG_DIR, data)
    print(f"Samples kept: {stats['kept']} | missing: {stats['missing']} | wrong_bands: {stats['wrong_bands']} | unreadable: {stats['unreadable']}")
    if len(data) == 0:
        raise RuntimeError("No valid samples after filtering. Check IMG_DIR or filenames.")

    dataset = AlphaEarthDataset(IMG_DIR, data)

    # deterministic split (solo para construir test set)
    n_total = len(dataset)
    n_train = int(TRAIN_SPLIT * n_total)
    n_val   = int(VAL_SPLIT * n_total)
    n_test  = n_total - n_train - n_val
    gen = torch.Generator().manual_seed(SEED)
    _, _, test_set = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_varlen)

    # Model
    model = S5Regressor().to(DEVICE)

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
