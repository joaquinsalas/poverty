# satellite_xgb_target_cc_gpu_infer.py ---------------------------------------
import os, platform, pickle, warnings, time
warnings.filterwarnings("ignore")  # silence deprecation and device chatter

# Use only GPU 2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Hard-disable CodeCarbon network & file sinks (override any config/env)
os.environ["CODECARBON_SAVE_TO_API"] = "false"
os.environ["CODECARBON_SAVE_TO_FILE"] = "false"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from tqdm.auto import tqdm

# --- CodeCarbon (OFFLINE, no API, no file sink) ---
try:
    from codecarbon import OfflineEmissionsTracker
except ImportError:
    OfflineEmissionsTracker = None
    print("WARNING: codecarbon not installed. `pip install codecarbon` to record emissions.")

# ------------------- rutas -------------------------------------------
if platform.system() == "Windows":
    IMG_DIR   = r"E:\sentinel_images\BaseDatos_Sentinel2A"
    CSV_IN    = r"E:\ensemble_inferences_calidad_vivienda_2020.csv"
    MODEL_DIR = r"E:\xgb_models"
    DATA_OUT  = r"E:\data"
else:
    IMG_DIR   = "/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A/"
    CSV_IN    = "../data/ensemble_inferences_calidad_vivienda_2020.csv"
    MODEL_DIR = "../models"
    DATA_OUT  = "../data"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_OUT, exist_ok=True)

# ---- set your trained model & scaler filenames here -----------------
MODEL_PATH  = os.path.join(MODEL_DIR, "xgb_target_12b.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_target_12b.pkl")

# ------------------- 1. Cargar meta-datos y objetivo -----------------
df = pd.read_csv(CSV_IN)
df["target"] = df[[f"prediction_{i:02d}" for i in range(1, 31)]].mean(axis=1)

# ------------------- 2. Dataset de Sentinel-2 ------------------------
def load_image(path):
    with rasterio.open(path) as src:
        return src.read()  # (bands, H, W)

class SatelliteDataset(Dataset):
    def __init__(self, meta, img_dir):
        self.meta, self.img_dir = meta, img_dir

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img = load_image(f"{self.img_dir}/{row['codigo']}.tif") * 5e-5
        img = img[:, 24:72:2, 24:72:2]  # downsample/crop
        feat = img.astype(np.float32)
        return feat.ravel(), row["target"], row.get("codigo", idx)

# ------------------- 3. Construir matriz de características ----------
dataset = SatelliteDataset(df, IMG_DIR)
loader = DataLoader(dataset, batch_size=256, num_workers=4)

X_list, y_list, id_list = [], [], []
for feats, target, codigo in tqdm(loader, desc="Extrayendo características", total=len(loader)):
    X_list.append(feats)
    y_list.append(target)
    id_list.extend(list(codigo))

X   = torch.cat(X_list).numpy()
y   = torch.cat(y_list).numpy()
ids = np.array(id_list)
print("Matriz X:", X.shape, "  vector y:", y.shape)

# ------------------- 4. Partición test (sin re-entrenar) -------------
all_idx = np.arange(len(X))
_, test_idx, _, test_y = train_test_split(
    all_idx, y, test_size=0.5, random_state=42, shuffle=True
)
test_X   = X[test_idx]
test_ids = ids[test_idx]

# ------------------- 5. Cargar scaler y modelo entrenado -------------
if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    raise FileNotFoundError(f"Missing model or scaler.\nMODEL_PATH={MODEL_PATH}\nSCALER_PATH={SCALER_PATH}")

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
with open(MODEL_PATH, "rb") as f:
    model: XGBRegressor = pickle.load(f)

# Ensure GPU prediction with quiet logs (no gpu_hist deprecation / mismatch)
model.set_params(tree_method="hist", device="cuda", predictor="auto", verbosity=0)

# Transform ONLY with the loaded scaler (no refit)
test_X = scaler.transform(test_X)

# ------------------- 6. Inferencia y R² (opcional) -------------------
pred = model.predict(test_X)
r2  = r2_score(test_y, pred)
print(f"\n► Test R² (inference-only) = {r2:.4f}")

# ------------------- 7. Emisiones (GPU, GRAMOS, promedio por ítem) ---
EMISSIONS_DIR = os.path.join(DATA_OUT, "emissions")
os.makedirs(EMISSIONS_DIR, exist_ok=True)
EMISSIONS_CSV = os.path.join(EMISSIONS_DIR, "xgb_23_CO2_emissions_per_item_grams_gpu_infer.csv")

REPETITIONS = 3
MEASURE_POWER_SECS = 1

records = []
print("\nMidiendo huella de carbono (gramos CO2e por ítem, GPU, sólo inferencia) ...")

grams_samples = []  # <-- initialize

if OfflineEmissionsTracker is None:
    # Fallback: write NaNs if CodeCarbon not installed
    grams_samples = [np.nan] * REPETITIONS
else:
    for _r in range(REPETITIONS):
        tracker = OfflineEmissionsTracker(
            country_iso_code="MEX",
            measure_power_secs=MEASURE_POWER_SECS,
            save_to_file=False,      # don't let CodeCarbon write/append CSV
            log_level="critical",    # silence library logs
            tracking_mode="process"
        )
        tracker.start()
        _ = model.predict(test_X)    # full test inference on GPU
        emissions_kg = tracker.stop()  # total kg CO2e for the run
        # Convert to grams and average per item
        emissions_g_per_item = (np.nan if emissions_kg is None
                                else float(emissions_kg) * 1000.0 / len(test_X))
        grams_samples.append(emissions_g_per_item)

emissions_g_median = float(np.nanmedian(grams_samples))
emissions_g_mean   = float(np.nanmean(grams_samples))

records.append({
    "n_items": int(len(test_X)),
    "repetitions": REPETITIONS,
    "emissions_g_per_item_median": emissions_g_median,
    "emissions_g_per_item_mean": emissions_g_mean
})

pd.DataFrame.from_records(records).to_csv(EMISSIONS_CSV, index=False)
print(f"Per-item emissions (grams, averaged over test set) saved to: {EMISSIONS_CSV}")