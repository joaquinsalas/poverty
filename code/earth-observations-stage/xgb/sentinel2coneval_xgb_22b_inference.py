# satellite_xgb_target_cc_gpu_infer.py ---------------------------------------
import os, platform, pickle, warnings, time
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

try:
    from codecarbon import OfflineEmissionsTracker
except ImportError:
    OfflineEmissionsTracker = None
    print("WARNING: codecarbon not installed. `pip install codecarbon` to record emissions.")

# ------------------- paths -------------------------------------------
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

MODEL_PATH  = os.path.join(MODEL_DIR, "xgb_target_23.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_target_23.pkl")

# ------------------- 1. metadata + target ----------------------------
df = pd.read_csv(CSV_IN)
df["target"] = df[[f"prediction_{i:02d}" for i in range(1, 31)]].mean(axis=1)

# ------------------- 2. dataset (MATCH TRAINING FEATURES) ------------
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

        # same crop/downsample as training
        img = img[:, 24:72:2, 24:72:2]

        # same indices as training (assumes at least 6 bands)
        blue, green, red, nir, sw1, sw2 = img[:6]
        eps = 1e-8
        idx_stack = np.stack([
            (nir-red)/(nir+red+eps),
            2.5*(nir-red)/(nir+6*red-7.5*blue+1+eps),
            (green-nir)/(green+nir+eps),
            (sw1-nir)/(sw1+nir+eps),
            1.5*(nir-red)/(nir+red+0.5+eps),
            (nir-sw2)/(nir+sw2+eps),
            2.5*(nir-red)/(nir+2.4*red+1+eps),
            (2*nir+1-np.sqrt((2*nir+1)**2-8*(nir-red)))/2,
            (nir-(sw1-sw2))/(nir+(sw1-sw2)+eps),
            (red-blue)/(red+blue+eps),
            (blue+green+red)/3
        ])

        feat = np.concatenate([img, idx_stack], axis=0).astype(np.float32)
        return feat.ravel(), float(row["target"]), row.get("codigo", idx)

# ------------------- 3. build feature matrix -------------------------
dataset = SatelliteDataset(df, IMG_DIR)
loader = DataLoader(dataset, batch_size=256, num_workers=4)

X_list, y_list, id_list = [], [], []
for feats, target, codigo in tqdm(loader, desc="Extrayendo características", total=len(loader)):
    X_list.append(feats)
    y_list.append(target)
    id_list.extend(list(codigo))

X   = torch.cat(X_list).numpy()
y   = torch.tensor(y_list).numpy() if not isinstance(y_list[0], torch.Tensor) else torch.cat(y_list).numpy()
ids = np.array(id_list)
print("Matriz X:", X.shape, "  vector y:", y.shape)

# ------------------- 4. test split (same seed) ------------------------
all_idx = np.arange(len(X))
_, test_idx, _, test_y = train_test_split(
    all_idx, y, test_size=0.5, random_state=42, shuffle=True
)
test_X   = X[test_idx]
test_ids = ids[test_idx]

# ------------------- 5. load scaler + trained model -------------------
if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    raise FileNotFoundError(f"Missing model or scaler.\nMODEL_PATH={MODEL_PATH}\nSCALER_PATH={SCALER_PATH}")

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
with open(MODEL_PATH, "rb") as f:
    model: XGBRegressor = pickle.load(f)

# GPU prediction setup
model.set_params(tree_method="hist", device="cuda", predictor="auto", verbosity=0)

# transform using loaded scaler only
test_X = scaler.transform(test_X)

# ------------------- 6. inference + R2 --------------------------------
pred = model.predict(test_X)
r2 = r2_score(test_y, pred)
print(f"\n► Test R² (inference-only) = {r2:.4f}")

# ------------------- 7. emissions per item ----------------------------
EMISSIONS_DIR = os.path.join(DATA_OUT, "emissions")
os.makedirs(EMISSIONS_DIR, exist_ok=True)
EMISSIONS_CSV = os.path.join(
    EMISSIONS_DIR, "xgb_23_CO2_emissions_per_item_grams_gpu_infer_rev.csv"
)

REPETITIONS = 3
MEASURE_POWER_SECS = 1

print("\nMidiendo huella de carbono (gramos CO2e por ítem, GPU, sólo inferencia) ...")

if OfflineEmissionsTracker is None:
    grams_samples = [np.nan] * REPETITIONS
else:
    grams_samples = []
    for _r in range(REPETITIONS):
        tracker = OfflineEmissionsTracker(
            country_iso_code="MEX",
            measure_power_secs=MEASURE_POWER_SECS,
            save_to_file=False,
            log_level="critical",
            tracking_mode="process",
        )
        tracker.start()
        _ = model.predict(test_X)
        emissions_kg = tracker.stop()
        grams_samples.append(
            np.nan if emissions_kg is None else float(emissions_kg) * 1000.0 / len(test_X)
        )

out = pd.DataFrame([{
    "n_items": int(len(test_X)),
    "repetitions": REPETITIONS,
    "emissions_g_per_item_median": float(np.nanmedian(grams_samples)),
    "emissions_g_per_item_mean": float(np.nanmean(grams_samples)),
}])
out.to_csv(EMISSIONS_CSV, index=False)
print(f"Per-item emissions saved to: {EMISSIONS_CSV}")
