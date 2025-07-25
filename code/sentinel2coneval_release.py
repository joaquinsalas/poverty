import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm
import timm
import rasterio
import csv

# Restrict to GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
        image = load_image(image_path) * 0.00005
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

def prepare_data(csv_path, image_dir, transform=None):
    df = pd.read_csv(csv_path)
    df = compute_target(df)
    dataset = SatelliteDataset(image_dir, df, transform)
    train_size = int(0.5 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])

# ---------------------- Training ----------------------

def train_model(model, train_loader, val_loader, criterion, optimizer, model_path, max_epochs=10000, patience=100):
    best_r2 = -np.inf
    patience_counter = 0
    train_history, val_history = [], []

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss, preds, true_vals = 0.0, [], []

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False):
                outputs = model(images).squeeze()
                val_loss += criterion(outputs, targets).item()
                preds.extend(outputs.numpy())
                true_vals.extend(targets.numpy())

        val_loss /= len(val_loader)
        val_r2 = r2_score(true_vals, preds)

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}")

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
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_preds, test_true_vals = [], []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing", leave=False):
            outputs = model(images).squeeze()
            test_preds.extend(outputs.numpy().flatten())
            test_true_vals.extend(targets.numpy().flatten())

    test_r2 = r2_score(test_true_vals, test_preds)
    print(f"Test R²: {test_r2:.4f}")

    # Save test R² to CSV file
    with open(results_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Test R²'])
        writer.writerow([test_r2])

    return test_r2

# ---------------------- Main ----------------------

def main():
    # Paths
    csv_path = "../data/ensemble_inferences_calidad_vivienda_2020.csv"
    image_dir = "/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A/"
    model_path = "../models/best_model_sentinel2coneval_20250724.pth"
    history_path = "../models/training_history_20250724.pth"
    results_path = "../data/test_results_r2_20250724.csv"

    # Prepare data
    train_ds, val_ds, test_ds = prepare_data(csv_path, image_dir)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Model setup
    model = create_model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train
    train_history, val_history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, model_path,
        max_epochs=10000, patience=10
    )

    # Save history
    torch.save({'train_history': train_history, 'val_history': val_history}, history_path)

    # Evaluate
    # Evaluate and save results
    evaluate_model(model, test_loader, model_path, results_path)

if __name__ == "__main__":
    main()
