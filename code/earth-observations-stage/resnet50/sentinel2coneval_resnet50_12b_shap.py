# satellite_resnet_shap_cc_gpu.py -------------------------------------------
import os, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import pandas as pd
import numpy as np
from tqdm import tqdm
import timm
import rasterio

import shap
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


# ---------------------- Utility Functions ----------------------
def load_image(image_path):
    with rasterio.open(image_path) as src:
        return src.read()

def compute_target(df):
    df["target"] = df[[f"prediction_{i:02d}" for i in range(1, 31)]].mean(axis=1)
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
        codigo = self.df.iloc[idx]["codigo"]
        image_path = f"{self.image_dir}/{codigo}.tif"
        image = load_image(image_path) * 0.00005  # radiometric scaling
        image = np.concatenate([image, image[-1:]], axis=0)  # 13 chans
        x = torch.tensor(image, dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        y = torch.tensor(self.df.iloc[idx]["target"], dtype=torch.float32)
        return x, y


# ---------------------- Model Definition ----------------------
def create_model():
    model = timm.create_model("resnet50", in_chans=13, pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


# ---------------------- Data Preparation ----------------------
def prepare_data(csv_path, image_dir, transform=None, seed=42):
    df = pd.read_csv(csv_path)
    df = compute_target(df)
    dataset = SatelliteDataset(image_dir, df, transform)
    train_size = int(0.5 * len(dataset))
    val_size   = int(0.2 * len(dataset))
    test_size  = len(dataset) - train_size - val_size
    gen = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size, test_size], generator=gen)


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
    """Sample indices from ds, excluding any in `exclude`."""
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
        x_chunk = explain_x[i0:i1]
        sv = explainer.shap_values(x_chunk)
        chunks.append(normalize_shap(sv))

    return np.concatenate(chunks, axis=0)  # (N,C,H,W)


# ---------------------- Visualization helpers ----------------------
def robust_norm(img2d, p_low=2, p_high=98, eps=1e-8):
    lo = np.percentile(img2d, p_low)
    hi = np.percentile(img2d, p_high)
    x = (img2d - lo) / (hi - lo + eps)
    return np.clip(x, 0, 1)

def make_rgb(x_chw, band_idx_rgb=(3, 2, 1)):
    r = robust_norm(x_chw[band_idx_rgb[0]])
    g = robust_norm(x_chw[band_idx_rgb[1]])
    b = robust_norm(x_chw[band_idx_rgb[2]])
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

def save_channel_importance_plot(ch_imp, out_png):
    plt.figure(figsize=(8, 3.5))
    plt.bar(np.arange(len(ch_imp)), ch_imp)
    plt.xlabel("Channel")
    plt.ylabel("Mean |SHAP|")
    plt.title("Channel importance (mean absolute SHAP)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ---------------------- Channel importance over N_CHANNEL_IMPORTANCE ----------------------
def estimate_channel_importance(model, train_ds, bg_x, n_channel_importance=200, seed=123,
                                batch_size=8, exclude_bg_idx=None):
    """
    Estimates global channel importance by computing SHAP on a random subset of
    size n_channel_importance (from train_ds), excluding the background indices.
    Returns:
      ch_imp: (C,) mean abs shap per channel
      idx_used: indices (in train_ds) that were explained for channel importance
    """
    idx_used = sample_indices(train_ds, n_channel_importance, seed=seed, exclude=exclude_bg_idx)
    x_ci, _ = load_subset_as_batch(train_ds, idx_used)
    sv = run_shap(model, bg_x, x_ci, batch_size=batch_size)  # (N,C,H,W)
    per_sample_ch = np.mean(np.abs(sv), axis=(2, 3))          # (N,C)
    ch_imp = np.mean(per_sample_ch, axis=0)                   # (C,)
    return ch_imp, idx_used


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
    csv_path   = "../data/ensemble_inferences_calidad_vivienda_2020.csv"
    image_dir  = "/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A/"
    model_path = "../models/best_model_sentinel2coneval_resnet50_12_20250926.pth"

    out_dir = "../data/shap_resnet50_12_20250926/"
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Keep explain images at 10 (for paper selection)
    N_EXPLAIN = 10

    # NEW: separate knob for channel-importance estimation
    N_CHANNEL_IMPORTANCE = 400  # <-- adjust this (e.g., 200, 500, 1000)

    # Background for GradientExplainer
    N_BACKGROUND = 32

    SHAP_BATCH = 5  # reduce if OOM

    # Data splits
    train_ds, _, test_ds = prepare_data(csv_path, image_dir, seed=42)

    # Load model
    model = create_model()
    state_dict = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    # Background
    bg_idx = build_background_indices(train_ds, n_background=N_BACKGROUND, seed=0)
    bg_x, _ = load_subset_as_batch(train_ds, bg_idx)

    # ---------------------- Explain set (10 images) ----------------------
    ex_idx = sample_indices(test_ds, N_EXPLAIN, seed=1)
    ex_x, ex_y = load_subset_as_batch(test_ds, ex_idx)

    quick_stats("BG", bg_x)
    quick_stats("EX", ex_x)

    with torch.no_grad():
        preds = predict_batch(model, ex_x.to(device)).detach().cpu().numpy().astype(float)
    trues = ex_y.numpy().astype(float)

    shap_vals = run_shap(model, bg_x, ex_x, batch_size=SHAP_BATCH)  # (N,C,H,W)

    np.save(os.path.join(out_dir, "shap_values_explain.npy"), shap_vals)
    pd.DataFrame({"train_background_idx": bg_idx}).to_csv(os.path.join(out_dir, "background_indices.csv"), index=False)
    pd.DataFrame({"test_explain_idx": ex_idx}).to_csv(os.path.join(out_dir, "explain_indices.csv"), index=False)

    # Map explain samples to codigo
    df_all = compute_target(pd.read_csv(csv_path))
    test_orig_idx = np.array(test_ds.indices)[np.array(ex_idx)]
    codigos = df_all.iloc[test_orig_idx]["codigo"].astype(str).values

    pd.DataFrame({
        "explain_row": np.arange(len(ex_idx)),
        "test_subset_idx": ex_idx,
        "original_df_idx": test_orig_idx,
        "codigo": codigos,
        "y_true": trues,
        "y_pred": preds,
        "error": preds - trues
    }).to_csv(os.path.join(out_dir, "explain_predictions.csv"), index=False)

    # Save separate figures: RGB and SigmaAbs(SHAP)
    x_np = ex_x.detach().cpu().numpy()
    for i in tqdm(range(x_np.shape[0]), desc="Saving figures", total=x_np.shape[0]):
        rgb = make_rgb(x_np[i], band_idx_rgb=(3, 2, 1))  # adjust if needed
        sig = np.sum(np.abs(shap_vals[i]), axis=0)
        sig = robust_norm(sig)

        code = codigos[i]
        save_rgb_png(rgb, os.path.join(fig_dir, f"{i:02d}_RGB_{code}.png"))
        save_sigma_abs_shap_png(sig, os.path.join(fig_dir, f"{i:02d}_SIGMA_ABS_SHAP_{code}.png"), add_colorbar=True)

    # ---------------------- Channel importance on N_CHANNEL_IMPORTANCE samples ----------------------
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
        os.path.join(out_dir, "channel_importance_indices.csv"), index=False
    )

    pd.DataFrame({
        "channel": np.arange(len(ch_imp)),
        "mean_abs_shap": ch_imp
    }).to_csv(os.path.join(out_dir, "channel_importance.csv"), index=False)

    save_channel_importance_plot(ch_imp, os.path.join(fig_dir, "channel_importance.png"))

    print(f"Saved outputs to: {out_dir}")
    print("Key files:")
    print(" - shap_values_explain.npy")
    print(" - explain_predictions.csv")
    print(" - channel_importance.csv")
    print(" - channel_importance_indices.csv")
    print("Figures in:")
    print(f" - {fig_dir}")


if __name__ == "__main__":
    main()
