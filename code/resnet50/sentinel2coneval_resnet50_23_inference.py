# infer_resnet50_s2_24ch_emissions.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"          # Restrict to GPU 1
os.environ["CODECARBON_SAVE_TO_API"] = "false"    # no dashboard posts
os.environ["CODECARBON_SAVE_TO_FILE"] = "false"   # avoid CC auto-CSV

# https://sentinel.esa.int/documents/247904/685211/Sentinel-2-MSI-L2A-Product-Format-Specifications.pdf
import csv
import time
import numpy as np
import pandas as pd
import rasterio
import timm
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ---- CodeCarbon (OFFLINE tracker) ----
try:
    from codecarbon import OfflineEmissionsTracker
except ImportError:
    OfflineEmissionsTracker = None
    print("WARNING: codecarbon not installed. `pip install codecarbon` to record emissions.")

# ---------------------- Config ----------------------
IMAGE_DIR    = "/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A/"
CSV_FILE     = "../data/ensemble_inferences_calidad_vivienda_2020.csv"
MODEL_PATH   = "../models/model_sentinel2coneval_23b_20250731.pth"
HISTORY_PATH = "../models/training_history_sentinel2coneval_23b_20250731.pth"  # unused here but kept
RESULTS_PATH = "../data/sentinel2coneval_test_r2_23b_20250731.csv"             # R² CSV (append)
EMISSIONS_CSV= "../data/emissions/resnet50_23b_CO2_emissions_per_item_grams_gpu_infer.csv"

DUPLICATE_LAST_BAND = True        # 12 bands + duplicate -> 13
SCALE_FACTOR        = 0.00005     # Sentinel-2 radiometric scaling

MODEL_NAME  = "resnet50"
BATCH_SIZE  = 32
NUM_WORKERS = 4
SEED        = 42

# Emissions measurement setup
REPETITIONS         = 3
COUNTRY_ISO         = "MEX"
MEASURE_POWER_SECS  = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


# ---------------------- Helpers ----------------------
def save_test_r2_to_csv(path: str, r2_value: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["test_r2"])
        writer.writerow([r2_value])


def load_image(image_path: str) -> np.ndarray:
    with rasterio.open(image_path) as src:
        image = src.read()  # (bands, H, W)
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
        image = load_image(image_path).astype(np.float32) * SCALE_FACTOR  # (12, H, W)

        # Duplicate last band to make it 13 bands
        if DUPLICATE_LAST_BAND:
            last_channel = image[-1, :, :][np.newaxis, :, :]
            image = np.concatenate([image, last_channel], axis=0)  # (13, H, W)

        # Compute 11 spectral indices -> final (24, H, W)
        spectral_indices = self.calculate_spectral_indices(image)
        features = np.concatenate([image, spectral_indices], axis=0)  # (24, H, W)

        features = torch.tensor(features, dtype=torch.float32)
        if self.transform:
            features = self.transform(features)

        target = torch.tensor(self.df.iloc[idx]["target"], dtype=torch.float32)
        return features, target

    @staticmethod
    def calculate_spectral_indices(image: np.ndarray) -> np.ndarray:
        """
        Expects first 6 channels to be: blue, green, red, nir, swir1, swir2.
        Returns (11, H, W).
        """
        blue, green, red, nir, swir1, swir2 = image[:6]
        eps = 1e-8

        ndvi = (nir - red) / (nir + red + eps)
        evi  = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps))
        ndwi = (green - nir) / (green + nir + eps)
        ndbi = (swir1 - nir) / (swir1 + nir + eps)
        savi = 1.5 * ((nir - red) / (nir + red + 0.5 + eps))
        nbr  = (nir - swir2) / (nir + swir2 + eps)
        evi2 = 2.5 * ((nir - red) / (nir + 2.4 * red + 1 + eps))
        msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2
        nmdi = (nir - (swir1 - swir2)) / (nir + (swir1 - swir2) + eps)
        ndi45 = (red - blue) / (red + blue + eps)
        si = (blue + green + red) / 3

        indices = np.stack(
            [ndvi, evi, ndwi, ndbi, savi, nbr, evi2, msavi, nmdi, ndi45, si],
            axis=0
        ).astype(np.float32)
        return indices


# ---------------------- Data ----------------------
def load_dataframe(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df["target"] = df[[f"prediction_{i:02d}" for i in range(1, 31)]].mean(axis=1)
    return df


# ---------------------- Model ----------------------
def create_model(in_chans: int) -> nn.Module:
    # pretrained=False: we will load our checkpoint; avoids 3ch pretrain download
    model = timm.create_model(MODEL_NAME, in_chans=in_chans, pretrained=False)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    else:
        model.reset_classifier(num_classes=1)
    return model


# ---------------------- Inference + Carbon ----------------------
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
    """
    Repeat a full pass through test loader, measuring CO2e with CodeCarbon.
    Returns dict with median/mean grams per item and wall-clock seconds.
    """
    grams_samples, secs_samples = [], []

    if OfflineEmissionsTracker is None:
        grams_samples = [np.nan] * REPETITIONS
        secs_samples = [np.nan] * REPETITIONS
    else:
        for _ in range(REPETITIONS):
            tracker = OfflineEmissionsTracker(
                country_iso_code=COUNTRY_ISO,
                measure_power_secs=MEASURE_POWER_SECS,
                save_to_file=False,
                log_level="critical",
                tracking_mode="process"
            )
            start = time.time()
            tracker.start()
            with torch.no_grad():
                for images, _ in tqdm(loader, desc="Measuring emissions", leave=False):
                    images = images.to(device, non_blocking=True)
                    _ = model(images).squeeze()
            emissions_kg = tracker.stop()  # kg CO2e
            secs = time.time() - start

            grams_per_item = (np.nan if emissions_kg is None
                              else float(emissions_kg) * 1000.0 / max(1, n_items))
            grams_samples.append(grams_per_item)
            secs_samples.append(secs)

    return {
        "emissions_g_per_item_median": float(np.nanmedian(grams_samples)),
        "emissions_g_per_item_mean": float(np.nanmean(grams_samples)),
        "seconds_total_median": float(np.nanmedian(secs_samples)),
        "seconds_total_mean": float(np.nanmean(secs_samples))
    }


# ---------------------- Main ----------------------
def main():
    # Channel count: 12 + 1 duplicate + 11 indices = 24
    base_ch = 12 + (1 if DUPLICATE_LAST_BAND else 0)  # 13
    in_chans = base_ch + 11                            # 24

    # Data (we only use the test split for inference/emissions)
    df = load_dataframe(CSV_FILE)
    dataset = SatelliteDataset(IMAGE_DIR, df, transform=None)
    train_size = int(0.5 * len(dataset))
    val_size   = int(0.2 * len(dataset))
    test_size  = len(dataset) - train_size - val_size
    gen = torch.Generator().manual_seed(SEED)
    _, _, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=gen)

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=True, num_workers=NUM_WORKERS
    )
    n_items = len(test_ds)

    # Model (24-ch) and weights
    model = create_model(in_chans).to(device)
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
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
        "seconds_total_median": metrics["seconds_total_median"],
        "seconds_total_mean": metrics["seconds_total_mean"],
        "items_per_second_mean": (n_items / metrics["seconds_total_mean"]
                                  if metrics["seconds_total_mean"] and metrics["seconds_total_mean"] > 0 else np.nan),
        "test_r2": test_r2
    }]).to_csv(EMISSIONS_CSV, index=False)

    print(f"Per-item emissions (grams, averaged over test set) saved to: {EMISSIONS_CSV}")


if __name__ == "__main__":
    main()


