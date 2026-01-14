import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Restrict to GPU 1

# https://sentinel.esa.int/documents/247904/685211/Sentinel-2-MSI-L2A-Product-Format-Specifications.pdf
import csv
import random
import numpy as np
import pandas as pd
import rasterio
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ---------------------- Config ----------------------
IMAGE_DIR = "/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A/"
CSV_FILE = "../data/ensemble_inferences_calidad_vivienda_2020.csv"
MODEL_PATH = "../models/model_sentinel2coneval_23b_20260106.pth"
HISTORY_PATH = "../models/training_history_sentinel2coneval_23b_20260106.pth"
RESULTS_PATH = "../data/sentinel2coneval_test_r2_23b_20260106.csv"

DUPLICATE_LAST_BAND = False         # duplicate last S2 band to reach 13 channels for ResNet stem
SCALE_FACTOR = 0.00005              # Sentinel-2 radiometric scaling

MODEL_NAME = "resnet50"
BATCH_SIZE = 32
NUM_EPOCHS = 80
PATIENCE = 100
INIT_LR = 1e-4
MAX_LR = 1e-3

# ---- NEW: reproducibility ----
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------- Reproducibility ----------------------
def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic runs (may be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    # make each dataloader worker deterministic
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------- Utility ----------------------
def channel_stats_from_loader(loader, n_batches=None, device="cpu", percentiles=(0.1, 1, 5, 50, 95, 99, 99.9),
                              clip_abs=None, verbose=True):
    it = iter(loader)
    xb, _ = next(it)
    C = xb.shape[1]

    n_total = np.zeros(C, dtype=np.int64)
    n_nan   = np.zeros(C, dtype=np.int64)
    n_inf   = np.zeros(C, dtype=np.int64)
    s1      = np.zeros(C, dtype=np.float64)
    s2      = np.zeros(C, dtype=np.float64)
    vmin    = np.full(C, np.inf, dtype=np.float64)
    vmax    = np.full(C, -np.inf, dtype=np.float64)

    max_samples_per_ch = 200_000
    samples = [np.empty((0,), dtype=np.float32) for _ in range(C)]

    def update_percentile_samples(x_ch, ch):
        if x_ch.size == 0:
            return
        cur = samples[ch]
        if cur.size >= max_samples_per_ch:
            return
        take = x_ch
        if x_ch.size > 50_000:
            idx = np.random.choice(x_ch.size, size=50_000, replace=False)
            take = x_ch[idx]
        new = np.concatenate([cur, take.astype(np.float32)], axis=0)
        if new.size > max_samples_per_ch:
            new = new[:max_samples_per_ch]
        samples[ch] = new

    def process_batch(xb):
        nonlocal n_total, n_nan, n_inf, s1, s2, vmin, vmax
        xb_ = xb.to(device, non_blocking=True) if device != "cpu" else xb
        x = xb_.detach().cpu().numpy()

        if clip_abs is not None:
            x = np.clip(x, -clip_abs, clip_abs)

        B, C_, H, W = x.shape
        assert C_ == C
        x = x.transpose(1, 0, 2, 3).reshape(C, -1)

        for ch in range(C):
            xc = x[ch]
            n_total[ch] += xc.size

            nan_mask = np.isnan(xc)
            inf_mask = np.isinf(xc)
            n_nan[ch] += nan_mask.sum()
            n_inf[ch] += inf_mask.sum()

            finite = xc[~nan_mask & ~inf_mask]
            if finite.size == 0:
                continue

            vmin[ch] = min(vmin[ch], float(finite.min()))
            vmax[ch] = max(vmax[ch], float(finite.max()))
            s1[ch]  += float(finite.sum(dtype=np.float64))
            s2[ch]  += float((finite.astype(np.float64) ** 2).sum(dtype=np.float64))

            update_percentile_samples(finite, ch)

    process_batch(xb)

    batches_done = 1
    pbar = tqdm(it, total=(n_batches - 1) if n_batches else None, disable=not verbose, desc="Channel stats")
    for xb, _ in pbar:
        process_batch(xb)
        batches_done += 1
        if n_batches and batches_done >= n_batches:
            break

    n_finite = n_total - n_nan - n_inf
    mean = np.where(n_finite > 0, s1 / np.maximum(1, n_finite), np.nan)
    var  = np.where(n_finite > 1, (s2 / np.maximum(1, n_finite)) - mean**2, np.nan)
    var  = np.maximum(var, 0.0)
    std  = np.sqrt(var)

    pct_cols = {f"p{p:g}": np.full(C, np.nan, dtype=np.float64) for p in percentiles}
    for ch in range(C):
        if samples[ch].size == 0:
            continue
        for p in percentiles:
            pct_cols[f"p{p:g}"][ch] = float(np.percentile(samples[ch], p))

    df = pd.DataFrame({
        "ch": np.arange(C),
        "count": n_total,
        "finite_count": n_finite,
        "nan_frac": n_nan / np.maximum(1, n_total),
        "inf_frac": n_inf / np.maximum(1, n_total),
        "min": vmin,
        "max": vmax,
        "mean": mean,
        "std": std,
        **pct_cols
    })
    return df


# ---------------------- Helpers ----------------------
def save_test_metrics_to_csv(path: str, r2_value: float, rmse_value: float, mae_value: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["test_r2", "test_rmse", "test_mae"])
        writer.writerow([r2_value, rmse_value, mae_value])


def load_image(image_path: str) -> np.ndarray:
    with rasterio.open(image_path) as src:
        image = src.read()
    return image


# ---------------------- Dataset ----------------------
class SatelliteDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir: str, df: pd.DataFrame, transform=None):
        self.image_dir = image_dir
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        code = self.df.iloc[idx]["codigo"]
        image_path = f"{self.image_dir}/{code}.tif"
        image = load_image(image_path).astype(np.float32) * SCALE_FACTOR

        if DUPLICATE_LAST_BAND:
            last_channel = image[-1, :, :][np.newaxis, :, :]
            image = np.concatenate([image, last_channel], axis=0)

        spectral_indices = self.calculate_spectral_indices(image)
        features = np.concatenate([image, spectral_indices], axis=0)  # (C, H, W)
        features = torch.tensor(features, dtype=torch.float32)

        if self.transform:
            features = self.transform(features)

        target = torch.tensor(self.df.iloc[idx]["target"], dtype=torch.float32)
        return features, target

    @staticmethod
    def calculate_spectral_indices(image: np.ndarray) -> np.ndarray:
        blue, green, red, nir, swir1, swir2 = image[:6]
        eps = 1e-8

        ndvi  = (nir - red) / (nir + red + eps)
        evi   = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps))
        ndwi  = (green - nir) / (green + nir + eps)
        ndbi  = (swir1 - nir) / (swir1 + nir + eps)
        savi  = ((nir - red) / (nir + red + 0.5 + eps)) * 1.5
        nbr   = (nir - swir2) / (nir + swir2 + eps)
        evi2  = 2.5 * ((nir - red) / (nir + 2.4 * red + 1 + eps))
        msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2
        ndi45 = (red - blue) / (red + blue + eps)
        si    = (blue + green + red) / 3

        indices = np.stack([ndvi, evi, ndwi, ndbi, savi, nbr, evi2, msavi, ndi45, si]).astype(np.float32)
        return indices


# ---------------------- Data ----------------------
def load_dataframe(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df["target"] = df[[f"prediction_{i:02d}" for i in range(1, 31)]].mean(axis=1)
    return df


# ---------------------- Model ----------------------
def create_model(in_chans: int) -> nn.Module:
    model = timm.create_model(MODEL_NAME, in_chans=in_chans, pretrained=True)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    else:
        model.reset_classifier(num_classes=1)
    return model


# ---------------------- Metrics ----------------------
def compute_metrics(trues, preds):
    trues = np.asarray(trues, dtype=np.float64).reshape(-1)
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    r2 = r2_score(trues, preds)
    rmse = float(np.sqrt(mean_squared_error(trues, preds)))
    mae = float(mean_absolute_error(trues, preds))
    return r2, rmse, mae


# ---------------------- Train / Eval ----------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler):
    model.train()
    running_loss = 0.0
    preds, trues = [], []

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        preds.extend(outputs.detach().cpu().numpy().reshape(-1))
        trues.extend(labels.detach().cpu().numpy().reshape(-1))

    train_loss = running_loss / max(1, len(loader))
    train_r2, train_rmse, train_mae = compute_metrics(trues, preds)
    return train_loss, train_r2, train_rmse, train_mae


def validate(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    preds, trues = [], []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images).squeeze()
            val_loss += criterion(outputs, targets).item()
            preds.extend(outputs.detach().cpu().numpy().reshape(-1))
            trues.extend(targets.detach().cpu().numpy().reshape(-1))

    val_loss /= max(1, len(loader))
    val_r2, val_rmse, val_mae = compute_metrics(trues, preds)
    return val_loss, val_r2, val_rmse, val_mae


def test(model, loader, return_arrays: bool = False):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Testing", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images).squeeze()
            preds.extend(outputs.detach().cpu().numpy().reshape(-1))
            trues.extend(targets.detach().cpu().numpy().reshape(-1))

    r2, rmse, mae = compute_metrics(trues, preds)
    if return_arrays:
        return r2, rmse, mae, np.asarray(trues), np.asarray(preds)
    return r2, rmse, mae


# ---------------------- Main ----------------------
def main():
    # ---- NEW: fix randomness for split + sampling + stats sampling ----
    set_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)

    # Work out channel count
    base_ch = 12 + (1 if DUPLICATE_LAST_BAND else 0)
    in_chans = base_ch + 10  # 10 indices => 22 or 23 (if duplicate) ; your comment says 24, but 10 indices => +10

    # Data
    df = load_dataframe(CSV_FILE)
    dataset = SatelliteDataset(IMAGE_DIR, df, transform=None)

    train_size = int(0.5 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # ---- NEW: reproducible split ----
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=gen)

    # ---- NEW: reproducible DataLoader shuffling/workers ----
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
        worker_init_fn=seed_worker, generator=gen, num_workers=4, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True,
        worker_init_fn=seed_worker, generator=gen, num_workers=4, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True,
        worker_init_fn=seed_worker, generator=gen, num_workers=4, persistent_workers=True
    )

    stats_df = channel_stats_from_loader(test_loader, n_batches=50, device="cpu", clip_abs=None)
    print(stats_df.sort_values("std", ascending=False).head(30))
    stats_df.to_csv("../data/channel_stats_test.csv", index=False)

    # Model
    model = create_model(in_chans).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS
    )

    # Train with early stopping (track train+val metrics each epoch)
    best_r2 = -np.inf
    patience_counter = 0
    train_history, val_history = [], []

    for epoch in range(NUM_EPOCHS):
        train_loss, train_r2, train_rmse, train_mae = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler
        )
        val_loss, val_r2, val_rmse, val_mae = validate(model, val_loader, criterion)

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train loss={train_loss:.4f} r2={train_r2:.4f} rmse={train_rmse:.4f} mae={train_mae:.4f} | "
            f"Val loss={val_loss:.4f} r2={val_r2:.4f} rmse={val_rmse:.4f} mae={val_mae:.4f}"
        )

        if val_r2 > best_r2:
            best_r2 = val_r2
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter > PATIENCE:
                print("Early stopping")
                break

    # Save history
    torch.save({"train_history": train_history, "val_history": val_history}, HISTORY_PATH)

    # Test best model
    state = torch.load(MODEL_PATH, map_location=device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f"[Warn] Strict load failed on test load: {e}\n[Info] Retrying with strict=False.")
        model.load_state_dict(state, strict=False)

    test_r2, test_rmse, test_mae = test(model, test_loader)
    print(f"Test r2={test_r2:.4f} rmse={test_rmse:.4f} mae={test_mae:.4f}")
    save_test_metrics_to_csv(RESULTS_PATH, test_r2, test_rmse, test_mae)


if __name__ == "__main__":
    main()
