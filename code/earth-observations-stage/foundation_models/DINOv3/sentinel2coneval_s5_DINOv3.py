import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # pick your GPU

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.backends import cudnn
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
import rasterio

import random
import csv


from s5 import S5Block  # pip/your local s5 module

# === NEW: DINOv3 ===
from transformers import AutoModel
from sklearn.decomposition import PCA

# ----------------------------
# Config
# ----------------------------
IMG_DIR = '/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A/'   # where {codigo}.tif live
LABEL_CSV = "/mnt/data-r1/JoaquinSalas/Documents/informs/research/2023.11.16census/2024.07.29malla470/data/ensemble_inferences_calidad_vivienda_2020.csv"
MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "s5_dinov3_regressor_dinov3.pth")  # checkpoint path
R2_CSV = "../data/s5_dinov3_test_r2.csv"

# Put these near your config:
SAT_MEAN = (0.430, 0.411, 0.296)
SAT_STD  = (0.213, 0.156, 0.143)
IMG_SIZE = 224   # recommend 224 (multiple of 16). You can keep 256 if you prefer.



SEED = 42
BATCH_SIZE = 16
EPOCHS = 300
LR = 1e-4
WEIGHT_DECAY = 0.01
PATIENCE = 20
VAL_SPLIT = 0.20
TRAIN_SPLIT = 0.50

# —— your original 64-band input is only used to create 3‑ch images for DINOv3
D_INPUT = 12

# S5 head dimensions (encoder maps DINO→D_MODEL)
D_MODEL = 512
N_LAYERS = 3
DROPOUT = 0.1
PRENORM = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cudnn.benchmark = True

# ----------------------------
# DINOv3 settings (EDIT HERE)
# ----------------------------
# Recommended default (fits common large GPUs). 7B requires massive memory.
DINO_PRETRAINED = "facebook/dinov3-vitl16-pretrain-sat493m"
# Alternatives:
#   "facebook/dinov3-vit7b16-pretrain-sat493m"  # 7B, 4096-dim tokens (very heavy)

FREEZE_DINO = True
IMG_SIZE = 256  # must be multiple of 16 (DINOv3 patch size)
# Use explicit band indices if you know which bands are RGB-like in your 64-band stack:
RGB_BANDS = (3, 2, 1)  # Red, Green, Blue  # e.g., (2,1,0) or (3,2,1). If None -> PCA(12->3)
# Optional per-band clipping to robustify outliers before PCA / selection
PERCENTILE_CLIP = (0.5, 99.5)

S2_SCALE = 5e-5  # or 1e-4 depending on how your TIFFs were written


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed=SEED):

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
    # scale 0..1
    x = (x - lo) / (hi - lo + 1e-8)
    return x

# ----------------------------
# Dataset
# ----------------------------
class Sentinel2ADataset(Dataset):
    """
    CSV needs:
      - 'codigo' : filename stem ({IMG_DIR}/{codigo}.tif)
      - 'target' or 30 columns prediction_01..prediction_30 (will be averaged)

    Each .tif is (C, H, W) with C==12 (assumed). We convert to 3-ch image:
      - If RGB_BANDS is set: select those 3 bands.
      - Else: per-image PCA(64->3) over (H*W, 64).
    Output x: float32 torch tensor in shape (3, H, W), values in [0,1].
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
            arr = src.read().astype(np.float32) * S2_SCALE  # Apply scale factor  # (C, H, W)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.shape[0] != D_INPUT:
            raise ValueError(f"{path} has {arr.shape[0]} bands; expected {D_INPUT}.")
        return arr.astype(np.float32)  # (64, H, W)

    def _to_three_channels(self, arr_chw):
        C, H, W = arr_chw.shape

        # optional robust clipping per band to suppress extreme outliers
        if PERCENTILE_CLIP is not None:
            for c in range(C):
                arr_chw[c] = _clip_percentiles(arr_chw[c], *PERCENTILE_CLIP)

        if RGB_BANDS is not None:
            assert len(RGB_BANDS) == 3, "RGB_BANDS must have length 3"
            rgb = arr_chw[list(RGB_BANDS)]  # (3, H, W)
            # min-max per channel to [0,1]
            #for k in range(3):
            #    ch = rgb[k]
            #    ch_min, ch_max = float(ch.min()), float(ch.max())
            #    if ch_max > ch_min:
            #        rgb[k] = (ch - ch_min) / (ch_max - ch_min)
            return rgb

        # PCA(64->3) over pixels
        X = arr_chw.reshape(C, -1).T  # (H*W, 12)
        # Normalize features per-band (zero-mean, unit-var) to stabilize PCA
        mu = X.mean(0, keepdims=True)
        sd = X.std(0, keepdims=True) + 1e-6
        Xn = (X - mu) / sd
        pca = PCA(n_components=3, svd_solver="randomized")
        Y = pca.fit_transform(Xn)  # (H*W, 3)
        # scale each component to [0,1] for image-like range
        for k in range(3):
            y = Y[:, k]
            lo, hi = np.percentile(y, 1), np.percentile(y, 99)
            if hi > lo:
                y = (np.clip(y, lo, hi) - lo) / (hi - lo)
            else:
                y = (y - y.min()) / (y.max() - y.min() + 1e-8)
            Y[:, k] = y
        rgb = Y.T.reshape(3, H, W).astype(np.float32)  # (3,H,W)
        return rgb

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        codigo = str(row["codigo"])
        arr = self._load_tif(codigo)                  # (64,H,W)
        img3 = self._to_three_channels(arr)           # (3,H,W) float32 in [0,1]
        y = np.float32(row["target"])
        x = torch.from_numpy(img3)
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

# ---------- collate: resize to fixed IMG_SIZE and stack
def collate_images_fixedsize(batch):
    xs, ys = zip(*batch)  # xs: list of (3,H,W) tensors in [0,1]
    B = len(xs)
    out = torch.zeros(B, 3, IMG_SIZE, IMG_SIZE, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        # x: (3,H,W) -> resize to (3,IMG_SIZE,IMG_SIZE) (bilinear)
        x = x.unsqueeze(0)  # (1,3,H,W)
        x = torch.nn.functional.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        out[i] = x.squeeze(0)
    y = torch.stack(ys)
    return out, y

def collate_noresize(batch):
    xs, ys = zip(*batch)                  # xs: list of (3,H,W) CPU tensors in [0,1]
    return list(xs), torch.stack(ys)




# ----------------------------
# DINOv3 + S5 Regressor
# ----------------------------
class DinoS5Regressor(nn.Module):
    def __init__(self, dino_name=DINO_PRETRAINED, d_model=D_MODEL, n_layers=N_LAYERS,
                 d_output=1, dropout=DROPOUT, prenorm=PRENORM, freeze_dino=FREEZE_DINO,
                 sat_mean=SAT_MEAN, sat_std=SAT_STD, img_size=IMG_SIZE, hf_token=None):
        super().__init__()
        # --- load model (use token if needed for gated access) ---

        kw = {}
        if hf_token is None:

            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            kw["token"] = hf_token  # or use_auth_token on older transformers
        self.dino = AutoModel.from_pretrained(dino_name, **kw)

        if freeze_dino:
            for p in self.dino.parameters():
                p.requires_grad = False

        self.img_size = img_size
        self.register_buffer("mean", torch.tensor(sat_mean).view(1,3,1,1), persistent=False)
        self.register_buffer("std",  torch.tensor(sat_std ).view(1,3,1,1), persistent=False)

        d_dino = self.dino.config.hidden_size  # 1024 for ViT-L/16
        self.prenorm = prenorm
        self.enc = nn.Linear(d_dino, d_model)
        self.s5_layers = nn.ModuleList([S5Block(dim=d_model, state_dim=d_model, bidir=False)
                                        for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, d_output)

        self.n_registers = getattr(self.dino.config, "num_register_tokens", 4)

    def _preprocess(self, x):
        """
        x: list of (3,H,W) CPU tensors in [0,1]
        returns: (B,3,IMG,IMG) on SAME device as self.mean/self.std
        """
        if isinstance(x, list):
            x = torch.stack(x, dim=0)  # (B,3,H,W)
        dev = self.mean.device  # same device as buffers (cpu or cuda)
        x = x.to(dev, non_blocking=True)
        x = torch.nn.functional.interpolate(x, size=(self.img_size, self.img_size),
                                            mode="bilinear", align_corners=False)
        x = x.clamp_(0, 1)
        x = (x - self.mean) / self.std
        return x

    def forward(self, imgs_3chw_list, lengths=None):
        pixel_values = self._preprocess(imgs_3chw_list)  # already on correct device
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.dino.parameters())):
            out = self.dino(pixel_values=pixel_values, output_hidden_states=False)
        tokens = out.last_hidden_state
        patch_tokens = tokens[:, 1 + self.n_registers:, :]
        x = self.enc(patch_tokens)
        for layer, norm, drop in zip(self.s5_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm: z = norm(z)
            z = layer(z);
            z = drop(z)
            x = x + z
            if not self.prenorm: x = norm(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)

# ----------------------------
# Train / Eval (unchanged API)
# ----------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running = 0.0
    # Train
    for xb, yb in tqdm(loader, desc="Train", leave=False):
        # xb stays on CPU; model() will move the processed batch to the model’s device
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        preds = model(xb)
        loss = nn.MSELoss()(preds, yb)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / max(1, len(loader))



@torch.no_grad()
def evaluate(model, loader, device, phase="Val/Test"):
    model.eval()
    losses, preds_all, y_all = [], [], []
    for xb, yb in tqdm(loader, desc=phase, leave=False):
        yb = yb.to(device)                     # keep xb on CPU
        preds = model(xb)
        loss = nn.MSELoss()(preds, yb)
        losses.append(loss.item())
        preds_all.append(preds.cpu().numpy())
        y_all.append(yb.cpu().numpy())
    preds_all = np.concatenate(preds_all)
    y_all = np.concatenate(y_all)
    r2 = r2_score(y_all, preds_all)
    return float(np.mean(losses)), float(r2)

# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading CSV labels...")
    data = pd.read_csv(LABEL_CSV)
    if "codigo" not in data.columns:
        raise ValueError("'codigo' column not found in label CSV.")
    data["codigo"] = data["codigo"].astype(str)

    data, stats = filter_existing_samples(IMG_DIR, data)
    print(f"Samples kept: {stats['kept']} | missing: {stats['missing']} | wrong_bands: {stats['wrong_bands']} | unreadable: {stats['unreadable']}")
    if len(data) == 0:
        raise RuntimeError("No valid samples after filtering. Check IMG_DIR or filenames.")

    dataset = Sentinel2ADataset(IMG_DIR, data)

    # deterministic split
    n_total = len(dataset)
    n_train = int(TRAIN_SPLIT * n_total)
    n_val   = int(VAL_SPLIT * n_total)
    n_test  = n_total - n_train - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_noresize)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True, collate_fn=collate_noresize)
    test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True, collate_fn=collate_noresize)

    # ----------------------------
    # Model + optim
    # ----------------------------
    model = DinoS5Regressor(dino_name=DINO_PRETRAINED, freeze_dino=FREEZE_DINO).to(DEVICE)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ----------------------------
    # Resume from ../models/s5_dinov3_regressor_dinov3.pth if present
    # ----------------------------
    start_epoch = 1
    best_val_r2 = -1e9

    if os.path.exists(MODEL_PATH):
        print(f"📦 Found checkpoint at {MODEL_PATH}. Loading and resuming training…")
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

        # Load model weights (supports both full checkpoint and raw state_dict)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
            if missing:   print(f"⚠️ Missing keys: {missing}")
            if unexpected: print(f"⚠️ Unexpected keys: {unexpected}")

            # Try optimizer/scheduler state if present
            if "optimizer_state" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                except Exception as e:
                    print(f"⚠️ Optimizer state incompatible; continuing without it. ({e})")

            if "scheduler_state" in ckpt:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state"])
                except Exception as e:
                    print(f"⚠️ Scheduler state incompatible; continuing without it. ({e})")

            best_val_r2 = ckpt.get("best_val_r2", best_val_r2)
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"✅ Resumed from epoch {start_epoch - 1} (best Val R²={best_val_r2:.4f}).")
        else:
            # Backward-compat: MODEL_PATH might be a raw state_dict
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            if missing:   print(f"⚠️ Missing keys: {missing}")
            if unexpected: print(f"⚠️ Unexpected keys: {unexpected}")
            print("✅ Loaded model weights only (no optimizer/scheduler). Starting from epoch 1.")
    else:
        print("🆕 No checkpoint found. Training from scratch.")

    # ----------------------------
    # Train loop with early stopping on Val R²
    # ----------------------------
    epochs_no_improve = 0
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        tr_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_loss, val_r2 = evaluate(model, val_loader, DEVICE, phase="Validation")
        print(f"  Train loss: {tr_loss:.4f} | Val loss: {val_loss:.4f} | Val R²: {val_r2:.4f}")

        improved = val_r2 > best_val_r2
        if improved:
            best_val_r2 = val_r2
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_r2": best_val_r2,
                "dino_pretrained": DINO_PRETRAINED,
                "img_size": IMG_SIZE,
            }, MODEL_PATH)
            print(f"💾 Saved new best checkpoint (Val R²={best_val_r2:.4f}).")
        else:
            epochs_no_improve += 1

        scheduler.step()

        if epochs_no_improve >= PATIENCE:
            print(f"⏹️ Early stopping after {epoch} epochs (no Val R² improvement for {PATIENCE}).")
            break

    print("\nEvaluating best model on test set...")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    test_loss, test_r2 = evaluate(model, test_loader, DEVICE, phase="Testing")
    print(f"TEST  | loss {test_loss:.4f} | R2 {test_r2:.4f}")

    save_test_r2_to_csv(R2_CSV, test_r2)

if __name__ == "__main__":
    main()
