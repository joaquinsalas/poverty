# s5_alphaEarth_shap_channels.py ---------------------------------------------
import os, warnings
warnings.filterwarnings("ignore")

# pick GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from tqdm.auto import tqdm
import rasterio

import shap
import matplotlib.pyplot as plt

from s5 import S5Block  # pip/your local s5 module

# ----------------------------
# Config
# ----------------------------
IMG_DIR   = "/mnt/data-r1/data/alphaEarth"   # {codigo}.tif live here
LABEL_CSV = "../data/ensemble_inferences_calidad_vivienda_2020.csv"
MODEL_PATH = "../models/s5_alphaEarth_regressor.pth"  # trained checkpoint (required)

OUT_DIR = "../data/shap_s5_alphaEarth/"
FIG_DIR = os.path.join(OUT_DIR, "figures")

SEED = 42

# dataset / tensors
D_INPUT = 64                 # channels in each tif
SCALE = 1.0                  # set if you need radiometric scaling
MAX_TOKENS = 512            # optional: e.g. 4096 to subsample H*W tokens

# splitting (only to pick BG / explain / CI subsets deterministically)
TRAIN_SPLIT = 0.50
VAL_SPLIT   = 0.20

# SHAP knobs
N_BACKGROUND = 8
N_EXPLAIN = 5
N_CHANNEL_IMPORTANCE = 400
SHAP_BATCH = 1               # reduce if OOM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_target(df):
    if "target" not in df.columns:
        pred_cols = [f"prediction_{i:02d}" for i in range(1, 31)]
        df["target"] = df[pred_cols].mean(axis=1)
    return df

def build_background_indices(train_ds, n_background=32, seed=0):
    rng = np.random.default_rng(seed)
    n = min(n_background, len(train_ds))
    return rng.choice(len(train_ds), size=n, replace=False).tolist()

def sample_indices(ds, n, seed=0, exclude=None):
    exclude = set([] if exclude is None else exclude)
    all_idx = np.array([i for i in range(len(ds)) if i not in exclude], dtype=int)
    rng = np.random.default_rng(seed)
    n = min(n, len(all_idx))
    return rng.choice(all_idx, size=n, replace=False).tolist()

def load_subset_as_batch(ds, indices):
    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False, pin_memory=True,
                        collate_fn=collate_varlen)
    xb, yb, lengths = next(iter(loader))
    return xb, yb, lengths

def normalize_shap(sv):
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv)

    # expected either (N,L,D) or (N,L,D,1)
    if sv.ndim == 4 and sv.shape[-1] == 1:
        sv = sv[..., 0]
    if sv.ndim != 3:
        raise ValueError(f"Unexpected SHAP shape: {sv.shape} (expected (N,L,D) or (N,L,D,1))")
    return sv


def save_channel_importance_plot(ch_imp, out_png):
    plt.figure(figsize=(8, 3.5))
    plt.bar(np.arange(len(ch_imp)), ch_imp)
    plt.xlabel("Channel")
    plt.ylabel("Mean |SHAP|")
    plt.title("Channel importance (mean absolute SHAP)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def quick_stats(name, x, lengths=None):
    # x: torch (B,L,D)
    x = x.detach().cpu().numpy()
    if lengths is None:
        valid = np.ones((x.shape[0], x.shape[1]), dtype=bool)
    else:
        L = x.shape[1]
        lengths = lengths.detach().cpu().numpy()
        valid = (np.arange(L)[None, :] < lengths[:, None])

    # compute per-channel stats over valid tokens
    B, L, D = x.shape
    x2 = x.reshape(B * L, D)
    v2 = valid.reshape(B * L)
    x2 = x2[v2]

    nan = np.isnan(x2).mean(axis=0)
    mn  = np.nanmean(x2, axis=0)
    sd  = np.nanstd(x2, axis=0)
    mx  = np.nanmax(x2, axis=0)
    mi  = np.nanmin(x2, axis=0)

    df = pd.DataFrame({"ch": np.arange(D), "nan_frac": nan, "min": mi, "max": mx, "mean": mn, "std": sd})
    print(name, "top std channels:\n", df.sort_values("std", ascending=False).head(10))
    return df


# ----------------------------
# Dataset (variable-length sequences)
# ----------------------------
class AlphaEarthDataset(Dataset):
    """
    Needs:
      - 'codigo' in CSV (filename stem: {IMG_DIR}/{codigo}.tif)
      - 'target' or prediction_01..prediction_30 (averaged)
    Each tif is (C,H,W) with C==64; reshape -> (H*W, 64) optionally subsampled to MAX_TOKENS.
    """
    def __init__(self, img_dir, df, scale=SCALE, max_tokens=MAX_TOKENS):
        self.img_dir = img_dir
        self.df = df.reset_index(drop=True)
        self.scale = scale
        self.max_tokens = max_tokens
        compute_target(self.df)

    def __len__(self):
        return len(self.df)

    def _load_tif(self, codigo):
        path = os.path.join(self.img_dir, f"{codigo}.tif")
        with rasterio.open(path) as src:
            arr = src.read()  # (C,H,W)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.shape[0] != D_INPUT:
            raise ValueError(f"{path} has {arr.shape[0]} bands; expected {D_INPUT}.")
        if self.scale != 1.0:
            arr = arr * self.scale

        x = np.transpose(arr, (1, 2, 0)).reshape(-1, D_INPUT).astype(np.float32)  # (L,64)

        if self.max_tokens is None:
            return x

        L = x.shape[0]
        T = int(self.max_tokens)

        # deterministic RNG per codigo
        rng = np.random.default_rng(abs(hash(str(codigo))) % (2 ** 32))

        if L >= T:
            idx = rng.choice(L, size=T, replace=False)
            x = x[idx]
        else:
            # pad with zeros to exactly T
            pad = np.zeros((T - L, D_INPUT), dtype=np.float32)
            x = np.concatenate([x, pad], axis=0)

        return x

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        codigo = str(row["codigo"])
        x = torch.from_numpy(self._load_tif(codigo))  # (T,64) fixed
        y = torch.tensor(np.float32(row["target"]), dtype=torch.float32)
        return x, y


def filter_existing_samples(img_dir, df):
    keep_rows = []
    missing, wrong_bands, unreadable = 0, 0, 0
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Scanning .tif files"):
        codigo = str(getattr(row, "codigo"))
        path = os.path.join(img_dir, f"{codigo}.tif")
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            with rasterio.open(path) as src:
                if src.count != D_INPUT:
                    wrong_bands += 1
                    continue
        except Exception:
            unreadable += 1
            continue
        keep_rows.append(row)
    fdf = pd.DataFrame(keep_rows, columns=df.columns)
    stats = {"kept": len(fdf), "missing": missing, "wrong_bands": wrong_bands, "unreadable": unreadable}
    return fdf, stats

def collate_varlen(batch):
    xs, ys = zip(*batch)
    lengths = [x.shape[0] for x in xs]
    maxlen = max(lengths)
    D = xs[0].shape[1]
    padded = torch.zeros(len(xs), maxlen, D, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        padded[i, :x.shape[0]] = x
    y = torch.stack(ys)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return padded, y, lengths


# ----------------------------
# Model (same as training)
# ----------------------------
class S5Regressor(nn.Module):
    """
    Input: (B, L, D_INPUT)
    Head: masked mean pool -> Linear -> scalar
    """
    def __init__(self, d_input=D_INPUT, d_model=512, n_layers=3, dropout=0.1, prenorm=True):
        super().__init__()
        self.prenorm = prenorm
        self.encoder = nn.Linear(d_input, d_model)
        self.s5_layers = nn.ModuleList([S5Block(dim=d_model, state_dim=d_model, bidir=False)
                                        for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, lengths=None):
        x = self.encoder(x)  # (B,L,Dm)
        for layer, norm, drop in zip(self.s5_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z)
            z = layer(z)
            z = drop(z)
            x = x + z
            if not self.prenorm:
                x = norm(x)

        if lengths is not None:
            B, L, _ = x.shape
            mask = (torch.arange(L, device=x.device).unsqueeze(0) < lengths.unsqueeze(1))  # (B,L)
            mask = mask.unsqueeze(-1)  # (B,L,1)
            x_sum = (x * mask).sum(dim=1)                 # (B,Dm)
            denom = lengths.clamp(min=1).unsqueeze(1)     # (B,1)
            x = x_sum / denom
        else:
            x = x.mean(dim=1)

        return self.head(x).squeeze(-1)  # (B,)


# ----------------------------
# SHAP core
# ----------------------------
def run_shap(model, background_x, background_len, explain_x, explain_len, batch_size=4):
    """
    Returns shap values with shape (N, L, D_INPUT).
    NOTE: SHAP will also attribute importance to padded tokens; we will mask them out later.
    """
    model.eval().to(DEVICE)
    background_x = background_x.to(DEVICE)
    background_len = background_len.to(DEVICE)
    explain_x = explain_x.to(DEVICE)
    explain_len = explain_len.to(DEVICE)
    explain_x.requires_grad_(True)

    # Wrap model so explainer sees a single-input function
    def f(x):
        # x: (B,L,D)
        # Use the lengths captured from outer scope? No: we pass fixed background lengths only for background.
        # For explain batches we will temporarily set a global var.
        raise RuntimeError("Use ModelWrapper below.")

    class ModelWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
            self._lengths = None

        def set_lengths(self, lengths):
            self._lengths = lengths

        def forward(self, x):
            y = self.base(x, self._lengths)  # (B,)
            return y.unsqueeze(-1)  # (B,1)  <-- key fix

    wrapped = ModelWrapper(model).to(DEVICE)
    wrapped.set_lengths(background_len)
    explainer = shap.GradientExplainer(wrapped, background_x)

    chunks = []
    n = explain_x.shape[0]
    n_batches = (n + batch_size - 1) // batch_size

    for b in tqdm(range(n_batches), desc="SHAP batches", total=n_batches):
        i0 = b * batch_size
        i1 = min(n, i0 + batch_size)
        x_chunk = explain_x[i0:i1]
        l_chunk = explain_len[i0:i1]
        wrapped.set_lengths(l_chunk)
        sv = explainer.shap_values(x_chunk)
        chunks.append(normalize_shap(sv))

    return np.concatenate(chunks, axis=0)  # (N,L,D)


def channel_importance_from_shap(shap_vals, lengths):
    """
    shap_vals: (N,L,D)
    lengths:   (N,)
    Returns mean |SHAP| per channel over valid tokens only: (D,)
    """
    N, L, D = shap_vals.shape
    lengths = np.asarray(lengths, dtype=int)
    mask = (np.arange(L)[None, :] < lengths[:, None])  # (N,L)
    abs_sv = np.abs(shap_vals)                         # (N,L,D)
    abs_sv = abs_sv * mask[..., None]                  # zero out padding
    denom = mask.sum()                                 # total valid tokens over all samples
    denom = max(1, int(denom))
    ch_imp = abs_sv.sum(axis=(0, 1)) / denom           # (D,)
    return ch_imp


# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Loading label CSV...")
    df = pd.read_csv(LABEL_CSV)
    if "codigo" not in df.columns:
        raise ValueError("'codigo' column not found in label CSV.")
    df["codigo"] = df["codigo"].astype(str)
    df = compute_target(df)

    # keep only usable 64-band tifs
    df, stats = filter_existing_samples(IMG_DIR, df)
    print(f"Samples kept: {stats['kept']} | missing: {stats['missing']} | wrong_bands: {stats['wrong_bands']} | unreadable: {stats['unreadable']}")
    if len(df) == 0:
        raise RuntimeError("No valid samples after filtering.")

    dataset = AlphaEarthDataset(IMG_DIR, df, scale=SCALE, max_tokens=MAX_TOKENS)

    # deterministic split (only for choosing bg / explain / CI subsets)
    n_total = len(dataset)
    n_train = int(TRAIN_SPLIT * n_total)
    n_val   = int(VAL_SPLIT * n_total)
    n_test  = n_total - n_train - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_ds, _, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    # Load trained model
    model = S5Regressor().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()

    # ---------------------- Background ----------------------
    bg_idx = build_background_indices(train_ds, n_background=N_BACKGROUND, seed=0)
    bg_x, _, bg_len = load_subset_as_batch(train_ds, bg_idx)
    quick_stats("BG", bg_x, bg_len)

    # ---------------------- Explain set (N_EXPLAIN) ----------------------
    ex_idx = sample_indices(test_ds, N_EXPLAIN, seed=1)
    ex_x, ex_y, ex_len = load_subset_as_batch(test_ds, ex_idx)
    quick_stats("EX", ex_x, ex_len)

    # Map explain samples to codigo (original df rows)
    # random_split stores original indices in .indices
    test_orig_idx = np.array(test_ds.indices)[np.array(ex_idx)]
    codigos = df.iloc[test_orig_idx]["codigo"].astype(str).values

    # ---------------------- SHAP for explain set ----------------------
    shap_vals = run_shap(model, bg_x, bg_len, ex_x, ex_len, batch_size=SHAP_BATCH)  # (N,L,D)

    np.save(os.path.join(OUT_DIR, "shap_values_explain.npy"), shap_vals)
    pd.DataFrame({"train_background_idx": bg_idx}).to_csv(os.path.join(OUT_DIR, "background_indices.csv"), index=False)
    pd.DataFrame({"test_explain_idx": ex_idx}).to_csv(os.path.join(OUT_DIR, "explain_indices.csv"), index=False)

    pd.DataFrame({
        "explain_row": np.arange(len(ex_idx)),
        "test_subset_idx": ex_idx,
        "original_df_idx": test_orig_idx,
        "codigo": codigos,
        "y_true": ex_y.detach().cpu().numpy().astype(float),
        "length_tokens": ex_len.detach().cpu().numpy().astype(int),
    }).to_csv(os.path.join(OUT_DIR, "explain_metadata.csv"), index=False)

    # ---------------------- Channel importance (global) ----------------------
    ci_idx = sample_indices(train_ds, N_CHANNEL_IMPORTANCE, seed=123, exclude=bg_idx)
    ci_x, _, ci_len = load_subset_as_batch(train_ds, ci_idx)

    sv_ci = run_shap(model, bg_x, bg_len, ci_x, ci_len, batch_size=SHAP_BATCH)  # (N,L,D)
    ch_imp = channel_importance_from_shap(sv_ci, ci_len.detach().cpu().numpy())

    pd.DataFrame({"train_channel_importance_idx": ci_idx}).to_csv(
        os.path.join(OUT_DIR, "channel_importance_indices.csv"), index=False
    )
    pd.DataFrame({
        "channel": np.arange(len(ch_imp)),
        "mean_abs_shap": ch_imp
    }).to_csv(os.path.join(OUT_DIR, "channel_importance.csv"), index=False)

    save_channel_importance_plot(ch_imp, os.path.join(FIG_DIR, "channel_importance.png"))

    print(f"\nSaved outputs to: {OUT_DIR}")
    print("Key files:")
    print(" - shap_values_explain.npy              (N_EXPLAIN, L, 64)")
    print(" - explain_metadata.csv")
    print(" - channel_importance.csv              (64 channels)")
    print(" - channel_importance_indices.csv")
    print("Figures in:")
    print(f" - {FIG_DIR}")
    print("\nNotes:")
    print(" - No R²/MSE/MAE computed (only SHAP + channel importance).")
    if MAX_TOKENS is not None:
        print(f" - MAX_TOKENS={MAX_TOKENS}: tokens are subsampled per image to cap SHAP cost.")

if __name__ == "__main__":
    main()
