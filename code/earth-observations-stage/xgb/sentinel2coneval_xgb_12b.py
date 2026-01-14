# satellite_xgb_target.py ------------------------------------------------
import os, platform, pickle, warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"          # usa la GPU 2

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import rasterio
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from tqdm.auto import tqdm        # ya no hace falta pin_memory=True aquí

# ------------------- rutas -------------------------------------------
if platform.system() == "Windows":
    IMG_DIR  = r"E:\sentinel_images\BaseDatos_Sentinel2A"
    CSV_IN   = r"E:\ensemble_inferences_calidad_vivienda_2020.csv"
    MODEL_DIR= r"E:\xgb_models"
else:
    IMG_DIR  = "/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A/"
    CSV_IN   = "../data/ensemble_inferences_calidad_vivienda_2020.csv"
    MODEL_DIR= "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------- 1. Cargar meta-datos y objetivo -----------------
df = pd.read_csv(CSV_IN)
df["target"] = df[[f"prediction_{i:02d}" for i in range(1, 31)]].mean(axis=1)

# ------------------- 2. Dataset de Sentinel-2 ------------------------
def load_image(path):
    with rasterio.open(path) as src:
        return src.read()                          # (bands,H,W)

class SatelliteDataset(Dataset):
    def __init__(self, meta, img_dir):
        self.meta, self.img_dir = meta, img_dir

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img = load_image(f"{self.img_dir}/{row['codigo']}.tif") * 5e-5
        #if img.shape[0] == 12:                     # duplica última banda
        #    img = np.concatenate([img, img[-1:]], axis=0)
        img = img[:, 24:72:2, 24:72:2]
        #blue, green, red, nir, sw1, sw2 = img[:6]
        eps=1e-8
        #idx_stack = np.stack([
        #    (nir-red)/(nir+red+eps),
        #    2.5*(nir-red)/(nir+6*red-7.5*blue+1+eps),
        #    (green-nir)/(green+nir+eps),
        #    (sw1-nir)/(sw1+nir+eps),
        #    1.5*(nir-red)/(nir+red+0.5+eps),
        #    (nir-sw2)/(nir+sw2+eps),
        #    2.5*(nir-red)/(nir+2.4*red+1+eps),
        #    (2*nir+1-np.sqrt((2*nir+1)**2-8*(nir-red)))/2,
        #    (nir-(sw1-sw2))/(nir+(sw1-sw2)+eps),
        #    (red-blue)/(red+blue+eps),
        #    (blue+green+red)/3
        #])
        feat = img.astype(np.float32)
        #feat = img.astype(np.float32)
        return feat.ravel(), row["target"]

# ------------------- 3. Construir matriz de características ----------
dataset = SatelliteDataset(df, IMG_DIR)
loader = DataLoader(dataset, batch_size=256, num_workers=4)

X_list, y_list = [], []
for feats, target in tqdm(loader, desc="Extrayendo características", total=len(loader)):
    X_list.append(feats)
    y_list.append(target)

X = torch.cat(X_list).numpy()
y = torch.cat(y_list).numpy()
print("Matriz X:", X.shape, "  vector y:", y.shape)

# ------------------- 4. Partición train / test -----------------------
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.5, random_state=42)

scaler = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
test_X  = scaler.transform(test_X)

# ------------------- 5. Espacio de hiper-parámetros ------------------
param_dist = {
    "eta":              np.linspace(0.01, 0.3, 30),
    "max_depth":        np.arange(3, 11),
    "subsample":        np.linspace(0.5, 1.0, 20),
    "colsample_bytree": np.linspace(0.5, 1.0, 20),
    "gamma":            np.linspace(0, 1, 20),
    "min_child_weight": np.arange(1, 7)
}

base_model = XGBRegressor(
    n_estimators=100,
    objective="reg:squarederror",
    tree_method="gpu_hist", predictor="gpu_predictor"
)

search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=200,
        cv=3,
        verbose=1,          # ≤ 0: silencioso; 1: breve; 2: detallado
        n_jobs=-1,
        scoring="neg_mean_squared_error",
        random_state=42)


import threading, time

def monitor(search_obj, interval=60):
    while not getattr(search_obj, "best_index_", None):
        time.sleep(interval)
    while search_obj.n_candidates_ > search_obj._fit_iter:
        best = search_obj.best_score_
        done = search_obj._fit_iter
        print(f"[{time.strftime('%X')}] {done}/{total_fits} fits · best mse={-best:.4f}")
        time.sleep(interval)

t = threading.Thread(target=monitor, args=(search,))
t.start()
search.fit(train_X, train_y)
t.join()

best = search.best_estimator_
best.fit(train_X, train_y)

pred = best.predict(test_X)
r2 = r2_score(test_y, pred)
print(f"\n► Test R² = {r2:.4f}")

# ------------------- 6. Guardar modelo y scaler ---------------------
with open(os.path.join(MODEL_DIR, "xgb_target_12b.pkl"), "wb") as f:
    pickle.dump(best, f)
with open(os.path.join(MODEL_DIR, "scaler_target_12b.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# ------------------- 7. Registrar resultados ------------------------
pd.DataFrame([{
    "r2_test": r2,
    **search.best_params_
}]).to_csv(os.path.join('../data/', "xgb_12b_target_summary.csv"), index=False)
print("\nModel & summary saved in", MODEL_DIR)