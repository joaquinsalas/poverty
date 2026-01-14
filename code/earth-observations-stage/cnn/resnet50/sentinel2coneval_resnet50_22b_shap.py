# satellite_resnet_shap_cc_gpu_23b_with_infer_emissions.py ----------------------
import os, warnings, csv, time
warnings.filterwarnings("ignore")

# GPU + CodeCarbon sinks OFF
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CODECARBON_SAVE_TO_API"] = "false"
os.environ["CODECARBON_SAVE_TO_FILE"] = "false"

import numpy as np
import pandas as pd
import rasterio
import timm
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

import shap
import matplotlib.pyplot as plt

# ---- CodeCarbon (OFFLINE tracker) ----
try:
    from codecarbon import OfflineEmissionsTracker
except ImportError:
    OfflineEmissionsTracker = None
    print("WARNING: codecarbon not installed. `pip install codecarbon` to record emissions.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


# ---------------------- Config ----------------------
IMAGE_DIR    = "/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A/"
CSV_FILE     = "../data/ensemble_inferences_calidad_vivienda_2020.csv"


MODEL_PATH   = "../models/model_sentinel2coneval_23b_20260106.pth"

RESULTS_PATH  = "../data/sentinel2coneval_test_metrics_23b_20260106.csv"
EMISSIONS_CSV = "../data/emissions/resnet50_23b_CO2_emissions_per_item_grams_gpu_infer_20260106.csv"

OUT_DIR = "../data/shap_resnet50_23b_20260106/"
FIG_DIR = os.path.join(OUT_DIR, "figures")

DUPLICATE_LAST_BAND = False  # MUST match training script (False)





# SHAP outputs


# Data/feature settings
SCALE_FACTOR = 0.00005

MODEL_NAME  = "resnet50"
BATCH_SIZE  = 32
NUM_WORKERS = 4
SEED        = 42

# Emissions measurement
REPETITIONS        = 3
COUNTRY_ISO        = "MEX"
MEASURE_POWER_SECS = 1

# SHAP controls
N_EXPLAIN            = 10
N_CHANNEL_IMPORTANCE = 30 #400
N_BACKGROUND         = 10 # 32
SHAP_BATCH           = 5  # reduce if OOM



def channel_stats_from_loader(loader, n_batches=None, device="cpu", percentiles=(0.1, 1, 5, 50, 95, 99, 99.9),
                              clip_abs=None, verbose=True):
    """
    Computes per-channel stats over a DataLoader yielding (x,y) with x shape (B,C,H,W).

    Returns a DataFrame with:
      ch, count, nan_frac, inf_frac, min, max, mean, std, p{...}

    Notes:
      - Streaming mean/std (sum, sumsq). Min/max exact.
      - Percentiles are estimated by sampling values per channel (reservoir-ish).
        If you want exact percentiles, you'll need to store all pixels (usually too big).
    """
    # --- init on first batch ---
    it = iter(loader)
    xb, _ = next(it)
    C = xb.shape[1]

    # accumulators (float64 for stability)
    n_total = np.zeros(C, dtype=np.int64)
    n_nan   = np.zeros(C, dtype=np.int64)
    n_inf   = np.zeros(C, dtype=np.int64)
    s1      = np.zeros(C, dtype=np.float64)
    s2      = np.zeros(C, dtype=np.float64)
    vmin    = np.full(C, np.inf, dtype=np.float64)
    vmax    = np.full(C, -np.inf, dtype=np.float64)

    # percentile sampling buffers (store limited number per channel)
    # keep at most ~200k values per channel by default (adjust if needed)
    max_samples_per_ch = 200_000
    samples = [np.empty((0,), dtype=np.float32) for _ in range(C)]

    def update_percentile_samples(x_ch, ch):
        # x_ch: 1D numpy array of finite values
        if x_ch.size == 0:
            return
        cur = samples[ch]
        if cur.size >= max_samples_per_ch:
            return
        # take a random subset if huge
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
        if device != "cpu":
            xb_ = xb.to(device, non_blocking=True)
        else:
            xb_ = xb
        x = xb_.detach().cpu().numpy()  # (B,C,H,W)

        if clip_abs is not None:
            x = np.clip(x, -clip_abs, clip_abs)

        B, C_, H, W = x.shape
        assert C_ == C

        # reshape to (C, Npix)
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

    # process first batch
    process_batch(xb)

    # process remaining
    batches_done = 1
    pbar = tqdm(it, total=(n_batches - 1) if n_batches else None, disable=not verbose, desc="Channel stats")
    for xb, _ in pbar:
        process_batch(xb)
        batches_done += 1
        if n_batches and batches_done >= n_batches:
            break

    # finalize mean/std
    n_finite = n_total - n_nan - n_inf
    mean = np.where(n_finite > 0, s1 / np.maximum(1, n_finite), np.nan)
    var  = np.where(n_finite > 1, (s2 / np.maximum(1, n_finite)) - mean**2, np.nan)
    var  = np.maximum(var, 0.0)
    std  = np.sqrt(var)

    # percentiles from samples
    pct_cols = {}
    for p in percentiles:
        pct_cols[f"p{p:g}"] = np.full(C, np.nan, dtype=np.float64)

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



def load_image(image_path: str) -> np.ndarray:
    with rasterio.open(image_path) as src:
        return src.read().astype(np.float32)  # (bands,H,W)

def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    df["target"] = df[[f"prediction_{i:02d}" for i in range(1, 31)]].mean(axis=1)
    return df

def calculate_spectral_indices(image: np.ndarray) -> np.ndarray:
    """
    Expects first 6 channels to be: blue, green, red, nir, swir1, swir2.
    Returns (11, H, W).
    """
    blue, green, red, nir, swir1, swir2 = image[:6]
    eps = 1e-8

    ndvi  = (nir - red) / (nir + red + eps)
    evi   = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps))
    ndwi  = (green - nir) / (green + nir + eps)
    ndbi  = (swir1 - nir) / (swir1 + nir + eps)
    savi  = 1.5 * ((nir - red) / (nir + red + 0.5 + eps))
    nbr   = (nir - swir2) / (nir + swir2 + eps)
    evi2  = 2.5 * ((nir - red) / (nir + 2.4 * red + 1 + eps))
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2
    #nmdi  = (nir - (swir1 - swir2)) / (nir + (swir1 - swir2) + eps) #21
    ndi45 = (red - blue) / (red + blue + eps) #22
    si    =  (blue + green + red) / 3 #23

    return np.stack([ndvi, evi, ndwi, ndbi, savi, nbr, evi2, msavi, ndi45, si], axis=0).astype(np.float32)
    #return np.stack([ndvi, evi, ndwi, ndbi, savi, nbr, evi2, msavi, nmdi, ndi45, si], axis=0).astype(np.float32)


# ---------------------- Dataset ----------------------
class SatelliteDataset(torch.utils.data.Dataset):
    """
    Builds channels like your inference script:
      base = 12 bands scaled
      optional +1 duplicated last band
      +10 spectral indices
    total = (12 + dup) + 10
    """
    def __init__(self, image_dir: str, df: pd.DataFrame, transform=None):
        self.image_dir = image_dir
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        code = self.df.iloc[idx]["codigo"]
        image_path = f"{self.image_dir}/{code}.tif"
        image = load_image(image_path) * SCALE_FACTOR  # (12,H,W)

        if DUPLICATE_LAST_BAND:
            last = image[-1:, :, :]  # (1,H,W)
            image = np.concatenate([image, last], axis=0)  # (13,H,W)

        spectral = calculate_spectral_indices(image)          # (10,H,W)
        features = np.concatenate([image, spectral], axis=0)  # (22 or 23,H,W)

        x = torch.tensor(features, dtype=torch.float32)
        if self.transform:
            x = self.transform(x)

        y = torch.tensor(self.df.iloc[idx]["target"], dtype=torch.float32)
        return x, y


# ---------------------- Model ----------------------
def create_model(in_chans: int) -> nn.Module:
    model = timm.create_model(MODEL_NAME, in_chans=in_chans, pretrained=False)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    else:
        model.reset_classifier(num_classes=1)
    return model


# ---------------------- Inference + Carbon ----------------------


def measure_emissions(model: nn.Module, loader: DataLoader, n_items: int) -> dict:
    grams_samples, secs_samples = [], []

    if OfflineEmissionsTracker is None:
        grams_samples = [np.nan] * REPETITIONS
        secs_samples  = [np.nan] * REPETITIONS
    else:
        for _ in range(REPETITIONS):
            tracker = OfflineEmissionsTracker(
                country_iso_code=COUNTRY_ISO,
                measure_power_secs=MEASURE_POWER_SECS,
                save_to_file=False,
                log_level="critical",
                tracking_mode="process",
            )
            start = time.time()
            tracker.start()
            with torch.no_grad():
                for images, _ in tqdm(loader, desc="Measuring emissions", leave=False):
                    images = images.to(device, non_blocking=True)
                    _ = model(images).squeeze()
            emissions_kg = tracker.stop()
            secs = time.time() - start

            grams_per_item = (np.nan if emissions_kg is None
                              else float(emissions_kg) * 1000.0 / max(1, n_items))
            grams_samples.append(grams_per_item)
            secs_samples.append(secs)

    return {
        "emissions_g_per_item_median": float(np.nanmedian(grams_samples)),
        "emissions_g_per_item_mean":   float(np.nanmean(grams_samples)),
        "seconds_total_median":        float(np.nanmedian(secs_samples)),
        "seconds_total_mean":          float(np.nanmean(secs_samples)),
    }


# ---------------------- SHAP helpers ----------------------
@torch.no_grad()
def predict_batch(model, xb):
    return model(xb).squeeze(-1)

def normalize_shap(sv):
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv)

    if sv.ndim == 5 and sv.shape[-1] == 1:
        sv = sv[..., 0]
    if sv.ndim == 5 and sv.shape[1] == 1:
        sv = sv[:, 0, ...]
    if sv.ndim == 5 and sv.shape[0] == 1:
        sv = sv[0, ...]

    if sv.ndim != 4:
        raise ValueError(f"Unexpected SHAP shape: {sv.shape}")
    return sv

def build_background_indices(train_ds, n_background=32, seed=0):
    rng = np.random.default_rng(seed)
    n = min(n_background, len(train_ds))
    idx = rng.choice(len(train_ds), size=n, replace=False)
    return idx.tolist()

def sample_indices(ds, n, seed=0, exclude=None):
    exclude = set([] if exclude is None else exclude)
    all_idx = np.array([i for i in range(len(ds)) if i not in exclude], dtype=int)
    rng = np.random.default_rng(seed)
    n = min(n, len(all_idx))
    pick = rng.choice(all_idx, size=n, replace=False)
    return pick.tolist()

def load_subset_as_batch(ds, indices):
    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False, pin_memory=True)
    xb, yb = next(iter(loader))
    return xb, yb

def run_shap(model, background_x, explain_x, batch_size=8):
    model.eval().to(device)
    background_x = background_x.to(device)
    explain_x    = explain_x.to(device)
    explain_x.requires_grad_(True)

    explainer = shap.GradientExplainer(model, background_x)

    chunks = []
    n = explain_x.shape[0]
    n_batches = (n + batch_size - 1) // batch_size

    for b in tqdm(range(n_batches), desc="SHAP batches", total=n_batches):
        i0 = b * batch_size
        i1 = min(n, i0 + batch_size)
        sv = explainer.shap_values(explain_x[i0:i1])
        chunks.append(normalize_shap(sv))

    return np.concatenate(chunks, axis=0)  # (N,C,H,W)

def estimate_channel_importance(model, train_ds, bg_x, n_channel_importance=200, seed=123,
                                batch_size=8, exclude_bg_idx=None):
    idx_used = sample_indices(train_ds, n_channel_importance, seed=seed, exclude=exclude_bg_idx)
    x_ci, _ = load_subset_as_batch(train_ds, idx_used)
    sv = run_shap(model, bg_x, x_ci, batch_size=batch_size)     # (N,C,H,W)
    per_sample_ch = np.mean(np.abs(sv), axis=(2, 3))             # (N,C)
    ch_imp = np.mean(per_sample_ch, axis=0)                      # (C,)
    return ch_imp, idx_used


# ---------------------- Visualization helpers ----------------------
def robust_norm(img2d, p_low=2, p_high=98, eps=1e-8):
    lo = np.percentile(img2d, p_low)
    hi = np.percentile(img2d, p_high)
    x = (img2d - lo) / (hi - lo + eps)
    return np.clip(x, 0, 1)

def make_rgb(x_chw):
    # Assumes first 6 channels: (blue, green, red, nir, swir1, swir2)
    r = robust_norm(x_chw[2])
    g = robust_norm(x_chw[1])
    b = robust_norm(x_chw[0])
    return np.stack([r, g, b], axis=-1)

def save_rgb_png(rgb, out_png):
    plt.figure(figsize=(4, 4))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png, dpi=250, bbox_inches="tight", pad_inches=0)
    plt.close()

def save_sigma_abs_shap_png(sig, out_png, add_colorbar=True):
    plt.figure(figsize=(4, 4))
    im = plt.imshow(sig)
    plt.axis("off")
    if add_colorbar:
        plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close()



S2_BAND_ACRONYMS_12 = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]
INDEX_ACRONYMS_10   = ["NDVI","EVI","NDWI","NDBI","SAVI","NBR","EVI2","MSAVI","NDI45","SI"]

def build_channel_labels(duplicate_last_band: bool):
    bands = S2_BAND_ACRONYMS_12.copy()
    if duplicate_last_band:
        bands.append(f"{bands[-1]}_dup")  # B12_dup
    return bands + INDEX_ACRONYMS_10



def save_channel_importance_plot(ch_imp, labels, out_png):
    plt.figure(figsize=(12, 3.8))
    x = np.arange(len(ch_imp))
    plt.bar(x, ch_imp)
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.xlabel("Channel")
    plt.ylabel("Mean |SHAP|")
    plt.title("Channel importance (mean absolute SHAP)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()




def quick_stats(name, x):
    # x: torch (N,C,H,W) or numpy
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    nan = np.isnan(x).mean(axis=(0,2,3))
    mn  = np.nanmean(x, axis=(0,2,3))
    sd  = np.nanstd(x, axis=(0,2,3))
    mx  = np.nanmax(x, axis=(0,2,3))
    mi  = np.nanmin(x, axis=(0,2,3))
    df = pd.DataFrame({"ch": np.arange(x.shape[1]), "nan_frac": nan, "min": mi, "max": mx, "mean": mn, "std": sd})
    print(name, "top std channels:\n", df.sort_values("std", ascending=False))
    return df



# ---------------------- Main ----------------------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # in_chans matches inference construction
    base_ch = 12 + (1 if DUPLICATE_LAST_BAND else 0)  # 12 or 13
    in_chans = base_ch + 10  # 22 or 23 (matches training indices=10)

    # Data + split (same as inference script)
    df = compute_target(pd.read_csv(CSV_FILE))
    dataset = SatelliteDataset(IMAGE_DIR, df, transform=None)
    train_size = int(0.5 * len(dataset))
    val_size   = int(0.2 * len(dataset))
    test_size  = len(dataset) - train_size - val_size
    gen = torch.Generator().manual_seed(SEED)
    train_ds, _, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=gen)

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=True, num_workers=NUM_WORKERS
    )
    n_items = len(test_ds)

    #stats_df = channel_stats_from_loader(test_loader, n_batches=50, device="cpu", clip_abs=None)
    #print(stats_df.sort_values("std", ascending=False).head(30))
    #stats_df.to_csv("../data/channel_stats_test.csv", index=False)

    # Model + weights
    model = create_model(in_chans=in_chans).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f"[Warn] Strict load failed ({e}); loading non-strict.")
        model.load_state_dict(state, strict=False)
    model.eval()
    w = model.conv1.weight if hasattr(model, "conv1") else None
    if w is not None:
        print("Model conv1 in_chans =", w.shape[1], " | data in_chans =", in_chans)
        assert w.shape[1] == in_chans, "Mismatch: model expects different number of input channels"

    # ------------------ (A) Inference accuracy + emissions ------------------
    #test_r2 = evaluate_r2(model, test_loader)
    #print(f"► Test R² (inference-only) = {test_r2:.4f} on {n_items} items")


    metrics = measure_emissions(model, test_loader, n_items)
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
        "in_chans": in_chans,
        "duplicate_last_band": DUPLICATE_LAST_BAND
    }]).to_csv(EMISSIONS_CSV, index=False)
    print(f"Per-item emissions CSV saved to: {EMISSIONS_CSV}")

    # ------------------ (B) SHAP explain set ------------------
    bg_idx = build_background_indices(train_ds, n_background=N_BACKGROUND, seed=0)
    bg_x, _ = load_subset_as_batch(train_ds, bg_idx)

    ex_idx = sample_indices(test_ds, N_EXPLAIN, seed=1)
    ex_x, ex_y = load_subset_as_batch(test_ds, ex_idx)


    quick_stats("BG", bg_x)
    quick_stats("EX", ex_x)

    with torch.no_grad():
        preds = predict_batch(model, ex_x.to(device)).detach().cpu().numpy().astype(float)
    trues = ex_y.numpy().astype(float)

    shap_vals = run_shap(model, bg_x, ex_x, batch_size=SHAP_BATCH)  # (N,C,H,W)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "shap_values_explain.npy"), shap_vals)
    pd.DataFrame({"train_background_idx": bg_idx}).to_csv(os.path.join(OUT_DIR, "background_indices.csv"), index=False)
    pd.DataFrame({"test_explain_idx": ex_idx}).to_csv(os.path.join(OUT_DIR, "explain_indices.csv"), index=False)

    # Map explain samples to codigo (original df indices)
    test_orig_idx = np.array(test_ds.indices)[np.array(ex_idx)]
    codigos = df.iloc[test_orig_idx]["codigo"].astype(str).values

    pd.DataFrame({
        "explain_row": np.arange(len(ex_idx)),
        "test_subset_idx": ex_idx,
        "original_df_idx": test_orig_idx,
        "codigo": codigos,
        "y_true": trues,
        "y_pred": preds,
        "error": preds - trues
    }).to_csv(os.path.join(OUT_DIR, "explain_predictions.csv"), index=False)

    # Figures: RGB + Sigma(|SHAP|)
    x_np = ex_x.detach().cpu().numpy()
    for i in tqdm(range(x_np.shape[0]), desc="Saving SHAP figures", total=x_np.shape[0]):
        rgb = make_rgb(x_np[i])
        sig = np.sum(np.abs(shap_vals[i]), axis=0)
        sig = robust_norm(sig)
        code = codigos[i]
        save_rgb_png(rgb, os.path.join(FIG_DIR, f"{i:02d}_RGB_{code}.png"))
        save_sigma_abs_shap_png(sig, os.path.join(FIG_DIR, f"{i:02d}_SIGMA_ABS_SHAP_{code}.png"), add_colorbar=True)

    # ------------------ (C) Channel importance (global-ish) ------------------
    ch_imp, ci_idx = estimate_channel_importance(
        model=model,
        train_ds=train_ds,
        bg_x=bg_x,
        n_channel_importance=N_CHANNEL_IMPORTANCE,
        seed=123,
        batch_size=SHAP_BATCH,
        exclude_bg_idx=bg_idx
    )

    pd.DataFrame({"train_channel_importance_idx": ci_idx}).to_csv(
        os.path.join(OUT_DIR, "channel_importance_indices.csv"), index=False
    )

    labels = build_channel_labels(DUPLICATE_LAST_BAND)
    assert len(labels) == in_chans, f"labels ({len(labels)}) != in_chans ({in_chans})"

    pd.DataFrame({
        "channel": np.arange(len(ch_imp)),
        "label": labels,
        "mean_abs_shap": ch_imp
    }).to_csv(os.path.join(OUT_DIR, "channel_importance.csv"), index=False)

    save_channel_importance_plot(ch_imp, labels, os.path.join(FIG_DIR, "channel_importance.png"))






    print(f"Saved SHAP outputs to: {OUT_DIR}")
    print(f"Figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()
