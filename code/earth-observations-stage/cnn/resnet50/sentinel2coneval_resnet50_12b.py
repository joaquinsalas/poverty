import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import timm
import rasterio
import csv
from torch.optim.lr_scheduler import OneCycleLR

# Restrict to GPU 2 (becomes cuda:0 within the process)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Reproducibility ----------------------

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic behavior (slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int):
    # Ensures each DataLoader worker has a deterministic seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
        # duplicate last band to get 13 channels (ResNet stem expects 13 here)
        image = np.concatenate([image, image[-1:]], axis=0)
        features = torch.tensor(image, dtype=torch.float32)

        if self.transform:
            features = self.transform(features)

        target = torch.tensor(self.df.iloc[idx]['target'], dtype=torch.float32)
        return features, target

# ---------------------- Model Definition ----------------------

def create_model():
    model = timm.create_model("resnet50", in_chans=13, pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

# ---------------------- Data Preparation ----------------------

def prepare_data(csv_path, image_dir, transform=None, seed: int = 42):
    df = pd.read_csv(csv_path)
    df = compute_target(df)
    dataset = SatelliteDataset(image_dir, df, transform)

    train_size = int(0.5 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    gen = torch.Generator().manual_seed(seed)  # reproducible split
    return random_split(dataset, [train_size, val_size, test_size], generator=gen)

# ---------------------- Metrics ----------------------

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return r2, rmse, mae

# ---------------------- Training ----------------------

def train_model(model, train_loader, val_loader, criterion, optimizer, model_path,
                max_epochs=10000, patience=100):
    best_r2 = -np.inf
    patience_counter = 0
    train_history, val_history = [], []

    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=max_epochs,
        anneal_strategy='linear'
    )

    for epoch in range(max_epochs):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        train_preds, train_true = [], []

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy().flatten())
            train_true.extend(labels.detach().cpu().numpy().flatten())

        train_loss = running_loss / max(1, len(train_loader))
        train_r2, train_rmse, train_mae = compute_metrics(train_true, train_preds)

        # ---- Validation ----
        model.eval()
        val_loss, val_preds, val_true = 0.0, [], []
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(images).squeeze()
                val_loss += criterion(outputs, targets).item()
                val_preds.extend(outputs.detach().cpu().numpy().flatten())
                val_true.extend(targets.detach().cpu().numpy().flatten())

        val_loss /= max(1, len(val_loader))
        val_r2, val_rmse, val_mae = compute_metrics(val_true, val_preds)

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(
            f"Epoch {epoch+1} | "
            f"Train: loss={train_loss:.4f} r2={train_r2:.4f} rmse={train_rmse:.4f} mae={train_mae:.4f} | "
            f"Val: loss={val_loss:.4f} r2={val_r2:.4f} rmse={val_rmse:.4f} mae={val_mae:.4f}"
        )

        # ---- Early stopping on val R2 ----
        if val_r2 > best_r2:
            best_r2 = val_r2
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter > patience:
                print("Early stopping triggered.")
                break

    return train_history, val_history

# ---------------------- Evaluation ----------------------

def evaluate_model(model, test_loader, model_path, results_path):
    # Load best weights and evaluate
    state_dict = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"[Warn] Strict load failed ({e}); loading non-strict.")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    test_preds, test_true = [], []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images).squeeze()
            test_preds.extend(outputs.detach().cpu().numpy().flatten())
            test_true.extend(targets.detach().cpu().numpy().flatten())

    test_r2, test_rmse, test_mae = compute_metrics(test_true, test_preds)
    print(f"Test: r2={test_r2:.4f} rmse={test_rmse:.4f} mae={test_mae:.4f}")

    # Save metrics to CSV
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Test_R2', 'Test_RMSE', 'Test_MAE'])
        writer.writerow([test_r2, test_rmse, test_mae])

    return test_r2, test_rmse, test_mae

# ---------------------- Main ----------------------

def main():
    SEED = 42
    set_seed(SEED)

    # Paths
    csv_path = "../data/ensemble_inferences_calidad_vivienda_2020.csv"
    image_dir = "/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A/"
    model_path = "../models/best_model_sentinel2coneval_resnet50_12_20260112a.pth"
    history_path = "../models/training_history_resnet50_12_20260112a.pth"
    results_path = "../data/test_results_resnet50_12_metrics_20260112a.csv"

    # Prepare data (reproducible split)
    train_ds, val_ds, test_ds = prepare_data(csv_path, image_dir, seed=SEED)

    # Deterministic DataLoader shuffling
    g = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, pin_memory=True,
        worker_init_fn=seed_worker, generator=g, num_workers=4, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False, pin_memory=True,
        worker_init_fn=seed_worker, generator=g, num_workers=4, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False, pin_memory=True,
        worker_init_fn=seed_worker, generator=g, num_workers=4, persistent_workers=True
    )

    # Model setup
    model = create_model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # If a previous model exists, load its weights before training
    if os.path.isfile(model_path):
        print(f"[Info] Found existing model at {model_path}. Loading weights.")
        state_dict = torch.load(model_path, map_location=device)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"[Warn] Strict load failed ({e}); loading non-strict.")
            model.load_state_dict(state_dict, strict=False)
        model.to(device)

    # Train
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    train_history, val_history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, model_path,
        max_epochs=80, patience=100
    )

    # Save history
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    torch.save({'train_history': train_history, 'val_history': val_history}, history_path)

    # Evaluate (R2, RMSE, MAE)
    evaluate_model(model, test_loader, model_path, results_path)

if __name__ == "__main__":
    main()


