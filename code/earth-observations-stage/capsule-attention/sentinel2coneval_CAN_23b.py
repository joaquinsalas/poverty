import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm
import rasterio
import csv
from torch.optim.lr_scheduler import OneCycleLR

# Restrict to GPU 2 (becomes cuda:0 within the process)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Spectral-band config (NEW) ----------------------
# Sentinel-2 scaling
SCALE_FACTOR = 0.00005

# Duplicate last original band before indices (optional)
DUPLICATE_LAST_BAND = False  # set True if you want 12->13 before appending indices

# How many spectral index channels to append
# Your reference code returns 10 indices
NUM_SPECTRAL_INDICES = 10

# If your TIFF has more than 6 bands, which band positions correspond to:
# blue, green, red, nir, swir1, swir2
# Default assumes first 6 channels already ordered like that.
# If your TIFF is in Sentinel-2 band order (B1..B12), set these accordingly.
BAND_MAP = {
    "blue": 0,
    "green": 1,
    "red": 2,
    "nir": 3,
    "swir1": 4,
    "swir2": 5,
}

# Optional: robust per-band clip+scale to [0,1] AFTER building all channels
ROBUST_CLIP = True
CLIP_LO = 0.5
CLIP_HI = 99.5

# ---------------------- Utility Functions ----------------------

def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read()  # (bands,H,W)
    return image

def compute_target(df):
    df["target"] = df[[f"prediction_{i:02d}" for i in range(1, 31)]].mean(axis=1)
    return df

def clip_scale01_per_band(arr_chw, p_lo=0.5, p_hi=99.5):
    """
    Robust per-band clip and scale to [0,1].
    arr_chw: (C,H,W) float32
    """
    C = arr_chw.shape[0]
    out = arr_chw.copy()
    for c in range(C):
        x = out[c]
        lo = np.percentile(x, p_lo)
        hi = np.percentile(x, p_hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            mn, mx = float(np.min(x)), float(np.max(x))
            if mx <= mn:
                out[c] = 0.0
            else:
                out[c] = (x - mn) / (mx - mn + 1e-8)
        else:
            x = np.clip(x, lo, hi)
            out[c] = (x - lo) / (hi - lo + 1e-8)
    return out.astype(np.float32)

def calculate_spectral_indices(image_chw, band_map=BAND_MAP):
    """
    Uses 6 bands to compute 10 indices (same as your reference):
      ndvi, evi, ndwi, ndbi, savi, nbr, evi2, msavi, ndi45, si

    image_chw: (C,H,W) float32, already scaled (radiometrically).
    band_map: dict with keys blue/green/red/nir/swir1/swir2 -> channel index.
    returns: (10,H,W) float32
    """
    b = image_chw[band_map["blue"]]
    g = image_chw[band_map["green"]]
    r = image_chw[band_map["red"]]
    n = image_chw[band_map["nir"]]
    s1 = image_chw[band_map["swir1"]]
    s2 = image_chw[band_map["swir2"]]
    eps = 1e-8

    ndvi  = (n - r) / (n + r + eps)
    evi   = 2.5 * ((n - r) / (n + 6 * r - 7.5 * b + 1 + eps))
    ndwi  = (g - n) / (g + n + eps)
    ndbi  = (s1 - n) / (s1 + n + eps)
    savi  = ((n - r) / (n + r + 0.5 + eps)) * 1.5
    nbr   = (n - s2) / (n + s2 + eps)
    evi2  = 2.5 * ((n - r) / (n + 2.4 * r + 1 + eps))
    msavi = (2 * n + 1 - np.sqrt((2 * n + 1) ** 2 - 8 * (n - r))) / 2
    ndi45 = (r - b) / (r + b + eps)
    si    = (b + g + r) / 3

    return np.stack([ndvi, evi, ndwi, ndbi, savi, nbr, evi2, msavi, ndi45, si]).astype(np.float32)

# ---------------------- Dataset Class (CHANGED) ----------------------

class SatelliteDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, df, transform=None,
                 scale=SCALE_FACTOR,
                 duplicate_last_band=DUPLICATE_LAST_BAND,
                 robust_clip=ROBUST_CLIP, clip_lo=CLIP_LO, clip_hi=CLIP_HI,
                 band_map=BAND_MAP,
                 append_indices=True):
        self.image_dir = image_dir
        self.df = df
        self.transform = transform
        self.scale = scale
        self.duplicate_last_band = duplicate_last_band
        self.robust_clip = robust_clip
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi
        self.band_map = band_map
        self.append_indices = append_indices

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        code = self.df.iloc[idx]["codigo"]
        image_path = f"{self.image_dir}/{code}.tif"
        image = load_image(image_path).astype(np.float32) * self.scale  # (C,H,W)

        # optional: duplicate last ORIGINAL band (before indices)
        if self.duplicate_last_band:
            image = np.concatenate([image, image[-1:]], axis=0)

        # append spectral indices -> (C+10,H,W)
        if self.append_indices:
            spectral = calculate_spectral_indices(image, self.band_map)  # (10,H,W)
            image = np.concatenate([image, spectral], axis=0)

        # robust per-band scaling to [0,1] for stability
        if self.robust_clip:
            image = clip_scale01_per_band(image, self.clip_lo, self.clip_hi)
        else:
            image = np.clip(image, 0.0, 1.0).astype(np.float32)

        features = torch.tensor(image, dtype=torch.float32)
        if self.transform:
            features = self.transform(features)

        target = torch.tensor(self.df.iloc[idx]["target"], dtype=torch.float32)
        return features, target

# ---------------------- Capsule Attention Network ----------------------

class PrimaryCapsules(nn.Module):
    def __init__(self, in_ch, num_caps=16, cap_dim=16, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.num_caps = num_caps
        self.cap_dim = cap_dim
        self.conv = nn.Conv2d(in_ch, num_caps * cap_dim, kernel_size, stride=stride, padding=padding)

    @staticmethod
    def squash(v, eps=1e-8):
        v2 = (v * v).sum(dim=-1, keepdim=True)
        scale = v2 / (1.0 + v2)
        v_norm = torch.sqrt(v2 + eps)
        return scale * (v / (v_norm + eps))

    def forward(self, x):
        z = self.conv(x)
        B, CD, Hp, Wp = z.shape
        z = z.view(B, self.num_caps, self.cap_dim, Hp, Wp)
        z = z.permute(0, 3, 4, 1, 2).contiguous()
        z = z.view(B, Hp * Wp * self.num_caps, self.cap_dim)
        return self.squash(z)

class CapsuleAttentionBlock(nn.Module):
    def __init__(self, cap_dim=16, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(cap_dim)
        self.attn = nn.MultiheadAttention(embed_dim=cap_dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(cap_dim)
        hidden = int(cap_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(cap_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, cap_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x

class CapsuleAttentionRegressor(nn.Module):
    def __init__(self, in_chans,
                 stem_dim=64,
                 num_caps=16,
                 cap_dim=32,
                 attn_layers=2,
                 attn_heads=4,
                 dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
            nn.Conv2d(stem_dim, stem_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
        )

        self.primary = PrimaryCapsules(stem_dim, num_caps=num_caps, cap_dim=cap_dim,
                                       kernel_size=3, stride=2, padding=1)

        self.blocks = nn.ModuleList([
            CapsuleAttentionBlock(cap_dim=cap_dim, num_heads=attn_heads, dropout=dropout)
            for _ in range(attn_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(cap_dim),
            nn.Linear(cap_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        caps = self.primary(x)
        for blk in self.blocks:
            caps = blk(caps)
        feat = caps.mean(dim=1)
        return self.head(feat).squeeze(-1)

# ---------------------- Model Definition (CHANGED) ----------------------

def create_model(in_chans):
    return CapsuleAttentionRegressor(
        in_chans=in_chans,
        stem_dim=64,
        num_caps=16,
        cap_dim=32,
        attn_layers=2,
        attn_heads=4,
        dropout=0.1
    )

# ---------------------- Data Preparation (CHANGED) ----------------------

def prepare_data(csv_path, image_dir, transform=None):
    df = pd.read_csv(csv_path)
    df = compute_target(df)

    # infer base channels from first file
    first_code = df.iloc[0]["codigo"]
    first_path = f"{image_dir}/{first_code}.tif"
    base_ch = load_image(first_path).shape[0]
    in_chans = base_ch + (1 if DUPLICATE_LAST_BAND else 0) + (NUM_SPECTRAL_INDICES if True else 0)

    dataset = SatelliteDataset(
        image_dir=image_dir,
        df=df,
        transform=transform,
        scale=SCALE_FACTOR,
        duplicate_last_band=DUPLICATE_LAST_BAND,
        robust_clip=ROBUST_CLIP,
        clip_lo=CLIP_LO,
        clip_hi=CLIP_HI,
        band_map=BAND_MAP,
        append_indices=True,
    )

    train_size = int(0.5 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size]), in_chans

# ---------------------- Training ----------------------

def train_model(model, train_loader, val_loader, criterion, optimizer, model_path,
                max_epochs=10000, patience=100):
    best_r2 = -np.inf
    patience_counter = 0
    train_history, val_history = [], []

    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=max_epochs,
        anneal_strategy="linear"
    )

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))

        model.eval()
        val_loss, preds, true_vals = 0.0, [], []
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(images)
                val_loss += criterion(outputs, targets).item()
                preds.extend(outputs.detach().cpu().numpy().flatten())
                true_vals.extend(targets.detach().cpu().numpy().flatten())

        val_loss /= max(1, len(val_loader))
        val_r2 = r2_score(true_vals, preds)

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}")

        if val_r2 > best_r2:
            best_r2 = val_r2
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter > patience:
                print("Early stopping triggered.")
                break

    return train_history, val_history

# ---------------------- Evaluation ----------------------

def evaluate_model(model, test_loader, model_path, results_path):
    state_dict = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"[Warn] Strict load failed ({e}); loading non-strict.")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    test_preds, test_true_vals = [], []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            test_preds.extend(outputs.detach().cpu().numpy().flatten())
            test_true_vals.extend(targets.detach().cpu().numpy().flatten())

    test_r2 = r2_score(test_true_vals, test_preds)
    print(f"Test R²: {test_r2:.4f}")

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Test R²"])
        writer.writerow([test_r2])

    return test_r2

# ---------------------- Main ----------------------

def main():
    csv_path = "../data/ensemble_inferences_calidad_vivienda_2020.csv"
    image_dir = "/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A/"
    model_path = "../models/best_model_CAN_spectral_23b.pth"
    history_path = "../models/training_history_CAN_spectral_23b.pth"
    results_path = "../data/test_results_CAN_spectral_r2_23b.csv"

    # Data
    (train_ds, val_ds, test_ds), in_chans = prepare_data(csv_path, image_dir)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, pin_memory=True)

    # Model
    model = create_model(in_chans=in_chans).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Optional resume
    LOAD_EXISTING_WEIGHTS = False
    if LOAD_EXISTING_WEIGHTS and os.path.isfile(model_path):
        state_dict = torch.load(model_path, map_location=device)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)

    # Train
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    train_history, val_history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, model_path,
        max_epochs=80, patience=10
    )

    # Save history
    torch.save({"train_history": train_history, "val_history": val_history}, history_path)

    # Evaluate
    evaluate_model(model, test_loader, model_path, results_path)

if __name__ == "__main__":
    main()
