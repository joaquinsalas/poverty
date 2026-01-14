# vit_infer_emissions.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"          # Restrict to GPU 1
os.environ["CODECARBON_SAVE_TO_API"] = "false"    # no dashboard posts
os.environ["CODECARBON_SAVE_TO_FILE"] = "false"   # avoid CC auto-CSV

import time
import csv
import numpy as np
import pandas as pd
import rasterio
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import r2_score
from tqdm import tqdm

# ---- CodeCarbon (OFFLINE tracker) ----
try:
    from codecarbon import OfflineEmissionsTracker
except ImportError:
    OfflineEmissionsTracker = None
    print("WARNING: codecarbon not installed. `pip install codecarbon` to record emissions.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ---------------------- Config ----------------------
CSV_PATH      = "../data/ensemble_inferences_calidad_vivienda_2020.csv"
IMAGE_DIR     = "/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A/"
MODEL_PATH    = "../models/model_sentinel2coneval_vit_12b_20250801.pth"   # pre-trained ViT checkpoint (13-ch)
RESULTS_PATH  = "../data/sentinel2coneval_vit_12b_r2_test_20250801.csv"   # append test R²
EMISSIONS_CSV = "../data/emissions/vit_12b_CO2_emissions_per_item_grams_gpu_infer.csv"

BATCH_SIZE  = 32
NUM_WORKERS = 4
SEED        = 42

# Emissions measurement setup
REPETITIONS        = 3
COUNTRY_ISO        = "MEX"
MEASURE_POWER_SECS = 1

SCALE_FACTOR = 5e-5  # Sentinel-2 radiometric scaling

# ---------------------- Utilities ----------------------
def save_test_r2_to_csv(path: str, r2_value: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, mode="a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["Model", "Test_R2"])
        w.writerow(["vit_base_patch16_224_in13", f"{r2_value:.6f}"])

def load_image(image_path: str) -> np.ndarray:
    with rasterio.open(image_path) as src:
        image = src.read()  # (C,H,W)
    return image

def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    df["target"] = df[[f"prediction_{i:02d}" for i in range(1, 31)]].mean(axis=1)
    return df

# ---------------------- Dataset ----------------------
class SatelliteDataset(torch.utils.data.Dataset):
    """
    Returns 13-channel tensors resized to 224x224 for ViT.
    """
    def __init__(self, image_dir: str, df: pd.DataFrame):
        self.image_dir = image_dir
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        code = self.df.iloc[idx]["codigo"]
        image_path = f"{self.image_dir}/{code}.tif"

        image = load_image(image_path).astype(np.float32) * SCALE_FACTOR  # (12,H,W)
        # Pad to 13 bands by duplicating last band
        last = image[-1:, :, :]
        image = np.concatenate([image, last], axis=0)  # (13,H,W)

        # To tensor and resize to 224x224
        x = torch.tensor(image, dtype=torch.float32)               # (13,H,W)
        x = F.interpolate(x.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)

        y = torch.tensor(self.df.iloc[idx]["target"], dtype=torch.float32)
        return x, y

# ---------------------- Data Prep ----------------------
def prepare_data(csv_path: str, image_dir: str, seed: int = SEED):
    df = pd.read_csv(csv_path)
    df = compute_target(df)
    dataset = SatelliteDataset(image_dir, df)
    # 50% train, 20% val, 30% test (to match your prior splits)
    train_size = int(0.5 * len(dataset))
    val_size   = int(0.2 * len(dataset))
    test_size  = len(dataset) - train_size - val_size
    gen = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size, test_size], generator=gen)

# ---------------------- Model ----------------------
def create_vit_model() -> nn.Module:
    """
    ViT-B/16 at 224 with 13 input channels and a 1-unit regression head.
    pretrained=False to avoid downloading 3-ch weights; we load our own ckpt.
    """
    model = timm.create_model("vit_base_patch16_224", in_chans=13, pretrained=False)
    # Replace head with a single-output regressor
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, 1)
    else:
        model.reset_classifier(num_classes=1)
    return model

# ---------------------- Inference + Emissions ----------------------
def evaluate_r2(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating (R²)", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images).squeeze()
            preds.extend(outputs.detach().cpu().numpy().flatten())
            trues.extend(targets.detach().cpu().numpy().flatten())
    return r2_score(trues, preds)

def measure_emissions(model: nn.Module, loader: DataLoader, n_items: int) -> dict:
    grams_samples = []
    if OfflineEmissionsTracker is None:
        grams_samples = [np.nan] * REPETITIONS
    else:
        for _ in range(REPETITIONS):
            tracker = OfflineEmissionsTracker(
                country_iso_code=COUNTRY_ISO,
                measure_power_secs=MEASURE_POWER_SECS,
                save_to_file=False,    # do not write CodeCarbon CSV
                log_level="critical",
                tracking_mode="process"
            )
            tracker.start()
            with torch.no_grad():
                for images, _ in tqdm(loader, desc="Measuring emissions", leave=False):
                    images = images.to(device, non_blocking=True)
                    _ = model(images).squeeze()
            emissions_kg = tracker.stop()  # kg CO2e
            grams_per_item = (np.nan if emissions_kg is None
                              else float(emissions_kg) * 1000.0 / max(1, n_items))
            grams_samples.append(grams_per_item)

    emissions_g_median = float(np.nanmedian(grams_samples))
    emissions_g_mean   = float(np.nanmean(grams_samples))
    return {
        "emissions_g_per_item_median": emissions_g_median,
        "emissions_g_per_item_mean":   emissions_g_mean
    }

# ---------------------- Main ----------------------
def main():
    # Data: only test split for inference/emissions
    _, _, test_ds = prepare_data(CSV_PATH, IMAGE_DIR, seed=SEED)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=True, num_workers=NUM_WORKERS)
    n_items = len(test_ds)

    # Model + checkpoint
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")
    model = create_vit_model().to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f"[Warn] Strict load failed ({e}); loading non-strict.")
        model.load_state_dict(state, strict=False)

    # 1) Accuracy pass (untracked)
    test_r2 = evaluate_r2(model, test_loader)
    print(f"► Test R² (inference-only) = {test_r2:.4f} on {n_items} items")
    save_test_r2_to_csv(RESULTS_PATH, test_r2)

    # 2) Emissions measurement (repeat full test pass)
    metrics = measure_emissions(model, test_loader, n_items)

    # Save emissions CSV
    os.makedirs(os.path.dirname(EMISSIONS_CSV), exist_ok=True)
    pd.DataFrame([{
        "n_items": n_items,
        "repetitions": REPETITIONS,
        "emissions_g_per_item_median": metrics["emissions_g_per_item_median"],
        "emissions_g_per_item_mean": metrics["emissions_g_per_item_mean"],
        "test_r2": test_r2
    }]).to_csv(EMISSIONS_CSV, index=False)

    print(f"Per-item emissions (grams, averaged over test set) saved to: {EMISSIONS_CSV}")

if __name__ == "__main__":
    main()
