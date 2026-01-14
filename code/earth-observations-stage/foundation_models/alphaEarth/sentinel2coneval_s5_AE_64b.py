import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # pick your GPU

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
from s5 import S5Block  # pip/your local s5 module

# ----------------------------
# Config
# ----------------------------
IMG_DIR = "/mnt/data-r1/data/alphaEarth"   # where {codigo}.tif live
LABEL_CSV = "../data/ensemble_inferences_calidad_vivienda_2020.csv"
MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "s5_alphaEarth_regressor.pth")  # checkpoint path
R2_CSV = "../data/s5_alphaEarth_test_r2.csv"

SEED = 42
BATCH_SIZE = 16          # keep modest, sequences can be long
EPOCHS = 300
LR = 1e-4
WEIGHT_DECAY = 0.01
PATIENCE = 20           # early stopping (no-improve epochs)
VAL_SPLIT = 0.20
TRAIN_SPLIT = 0.50      # remaining goes to test

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
        import csv
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
# Train / Eval
# ----------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running = 0.0
    for xb, yb, lengths in tqdm(loader, desc="Train", leave=False):
        xb, yb, lengths = xb.to(device), yb.to(device), lengths.to(device)
        optimizer.zero_grad(set_to_none=True)
        preds = model(xb, lengths)
        loss = nn.MSELoss()(preds, yb)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, device, phase="Val/Test"):
    model.eval()
    losses = []
    preds_all, y_all = [], []
    for xb, yb, lengths in tqdm(loader, desc=phase, leave=False):
        xb, yb, lengths = xb.to(device), yb.to(device), lengths.to(device)
        preds = model(xb, lengths)
        loss = nn.MSELoss()(preds, yb)
        losses.append(loss.item())
        preds_all.append(preds.detach().cpu().numpy())
        y_all.append(yb.detach().cpu().numpy())
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

    # Keep rows whose {codigo}.tif exists and has 64 bands (any HxW)
    data, stats = filter_existing_samples(IMG_DIR, data)
    print(f"Samples kept: {stats['kept']} | missing: {stats['missing']} | wrong_bands: {stats['wrong_bands']} | unreadable: {stats['unreadable']}")
    if len(data) == 0:
        raise RuntimeError("No valid samples after filtering. Check IMG_DIR or filenames.")

    dataset = AlphaEarthDataset(IMG_DIR, data)

    # deterministic split
    n_total = len(dataset)
    n_train = int(TRAIN_SPLIT * n_total)
    n_val   = int(VAL_SPLIT * n_total)
    n_test  = n_total - n_train - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_varlen)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True, collate_fn=collate_varlen)
    test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True, collate_fn=collate_varlen)

    # Model + optim
    model = S5Regressor().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ---- Resume logic ----
    start_epoch = 1
    best_val_r2 = -1e9
    if os.path.exists(MODEL_PATH):
        print(f"Found checkpoint at {MODEL_PATH}. Loading and resuming training…")
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        # Backward compatibility: allow pure state_dict or full checkpoint
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                except Exception:
                    print("⚠️ Optimizer state incompatible; continuing without it.")
            if "scheduler_state" in ckpt:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state"])
                except Exception:
                    print("⚠️ Scheduler state incompatible; continuing without it.")
            best_val_r2 = ckpt.get("best_val_r2", best_val_r2)
            start_epoch = ckpt.get("epoch", 0) + 1
        else:
            # old-style: model-only .state_dict()
            model.load_state_dict(ckpt)
            print("Loaded model weights only (no optimizer/scheduler).")

    # ---- Train loop with early stopping on Val R² ----
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
            # save a full checkpoint so we can resume cleanly
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_r2": best_val_r2,
            }, MODEL_PATH)
        else:
            epochs_no_improve += 1

        scheduler.step()

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping after {epoch} epochs (no val R² improvement for {PATIENCE}).")
            break

    print("\nEvaluating best model on test set...")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    test_loss, test_r2 = evaluate(model, test_loader, DEVICE, phase="Testing")
    print(f"TEST  | loss {test_loss:.4f} | R2 {test_r2:.4f}")

    save_test_r2_to_csv(R2_CSV, test_r2)

if __name__ == "__main__":
    main()


