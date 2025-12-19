# satellite_resnet_infer_cc_gpu.py -------------------------------------------
import os, warnings, csv, time
warnings.filterwarnings("ignore")

# --- GPU & CodeCarbon env ---
os.environ["CUDA_VISIBLE_DEVICES"] = "2"          # restrict to GPU 2
os.environ["CODECARBON_SAVE_TO_API"]  = "false"   # no dashboard posts
os.environ["CODECARBON_SAVE_TO_FILE"] = "false"   # CC won't write its own CSV

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim  # not used, kept for compatibility if you reuse file
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm
import timm
import rasterio

# ---- CodeCarbon (OFFLINE tracker) ----
try:
    from codecarbon import OfflineEmissionsTracker
except ImportError:
    OfflineEmissionsTracker = None
    print("WARNING: codecarbon not installed. `pip install codecarbon` to record emissions.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ---------------------- Utility Functions ----------------------
def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read()
    return image

def compute_target(df):
    df['target'] = df[[f'prediction_{i:02d}' for i in range(1, 31)]].mean(axis=1)
    return df

# ---------------------- Dataset Class ----------------------
class SatelliteDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, df, transform=None):
        self.image_dir = image_dir
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = f"{self.image_dir}/{self.df.iloc[idx]['codigo']}.tif"
        image = load_image(image_path) * 0.00005  # radiometric scaling
        # duplicate last band to get 13 channels
        image = np.concatenate([image, image[-1:]], axis=0)
        features = torch.tensor(image, dtype=torch.float32)

        if self.transform:
            features = self.transform(features)

        target = torch.tensor(self.df.iloc[idx]['target'], dtype=torch.float32)
        return features, target

# ---------------------- Model Definition ----------------------
def create_model():
    # pretrained=False to avoid downloading 3-ch weights; we load our own ckpt
    model = timm.create_model("resnet50", in_chans=13, pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

# ---------------------- Data Preparation ----------------------
def prepare_data(csv_path, image_dir, transform=None, seed=42):
    df = pd.read_csv(csv_path)
    df = compute_target(df)
    dataset = SatelliteDataset(image_dir, df, transform)
    # same split proportions as your script: 50% train, 20% val, 30% test
    train_size = int(0.5 * len(dataset))
    val_size   = int(0.2 * len(dataset))
    test_size  = len(dataset) - train_size - val_size
    gen = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size, test_size], generator=gen)

# ---------------------- Inference + Carbon ----------------------
def run_inference_and_emissions(model, test_loader, model_path, emissions_csv,
                                repetitions=3, country_iso="MEX", measure_power_secs=1):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"[Warn] Strict load failed ({e}); loading non-strict.")
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # --- 1) Accuracy pass (untracked) ---
    preds, trues = [], []
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating (R²)", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images).squeeze()
            preds.extend(outputs.detach().cpu().numpy().flatten())
            trues.extend(targets.detach().cpu().numpy().flatten())
    r2 = r2_score(trues, preds)
    n_items = len(trues)
    print(f"► Test R² (inference-only) = {r2:.4f} on {n_items} items")

    # --- 2) Emissions measurement (repeat full test pass) ---
    grams_samples = []
    if OfflineEmissionsTracker is None:
        grams_samples = [np.nan] * repetitions
    else:
        for _ in range(repetitions):
            tracker = OfflineEmissionsTracker(
                country_iso_code=country_iso,
                measure_power_secs=measure_power_secs,
                save_to_file=False,     # avoid CC CSV
                log_level="critical",   # silence logs
                tracking_mode="process"
            )
            tracker.start()
            with torch.no_grad():
                for images, _ in tqdm(test_loader, desc="Measuring emissions", leave=False):
                    images = images.to(device, non_blocking=True)
                    _ = model(images).squeeze()
            emissions_kg = tracker.stop()  # kg CO2e
            grams_per_item = (np.nan if emissions_kg is None
                              else float(emissions_kg) * 1000.0 / max(1, n_items))
            grams_samples.append(grams_per_item)

    emissions_g_median = float(np.nanmedian(grams_samples))
    emissions_g_mean   = float(np.nanmean(grams_samples))

    # Save CSV
    os.makedirs(os.path.dirname(emissions_csv), exist_ok=True)
    pd.DataFrame([{
        "n_items": n_items,
        "repetitions": repetitions,
        "emissions_g_per_item_median": emissions_g_median,
        "emissions_g_per_item_mean": emissions_g_mean,
        "test_r2": r2
    }]).to_csv(emissions_csv, index=False)

    print(f"Per-item emissions (grams, averaged over test set) saved to: {emissions_csv}")
    return r2, emissions_g_median, emissions_g_mean

# ---------------------- Main ----------------------
def main():
    # Paths (edit if needed)
    csv_path     = "../data/ensemble_inferences_calidad_vivienda_2020.csv"
    image_dir    = "/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A/"
    model_path   = "../models/best_model_sentinel2coneval_resnet50_12_20250926.pth"
    emissions_csv= "../data/emissions/resnet50_12_CO2_emissions_per_item_grams_gpu_infer.csv"

    # Data: we only need the test split for inference
    _, _, test_ds = prepare_data(csv_path, image_dir)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)

    # Model
    model = create_model()

    # Inference + Emissions
    run_inference_and_emissions(
        model=model,
        test_loader=test_loader,
        model_path=model_path,
        emissions_csv=emissions_csv,
        repetitions=3,            # robust median over repeats
        country_iso="MEX",        # grid factor (optional, adjust as needed)
        measure_power_secs=1
    )

if __name__ == "__main__":
    main()
