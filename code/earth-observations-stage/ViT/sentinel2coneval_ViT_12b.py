# %% Vision Transformer Regression Script
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
import rasterio
from tqdm import tqdm
from sklearn.metrics import r2_score
import timm

import csv

from torch.optim.lr_scheduler import OneCycleLR



# Load tabular data
csv_file = "../data/ensemble_inferences_calidad_vivienda_2020.csv"
data = pd.read_csv(csv_file)
data['target'] = data[[f'prediction_{i:02d}' for i in range(1, 31)]].mean(axis=1)

# Load image
def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read()  # (bands, height, width)
    return image

# Custom Dataset
class SatelliteDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, df):
        self.image_dir = image_dir
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = f"{self.image_dir}/{self.df.iloc[idx]['codigo']}.tif"
        image = load_image(image_path) * 0.00005  # Apply scale factor

        # Pad to 13 bands
        last_channel = image[-1, :, :]
        image = np.concatenate([image, last_channel[np.newaxis, :, :]], axis=0)

        #spectral_indices = self.calculate_spectral_indices(image)
        #features = np.concatenate((image, spectral_indices), axis=0)  # (24, H, W)
        features = image
        features = torch.tensor(features, dtype=torch.float32)

        # Resize to 224x224
        features = F.interpolate(features.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)

        target = torch.tensor(self.df.iloc[idx]['target'], dtype=torch.float32)
        return features, target

    def calculate_spectral_indices(self, image):
        blue, green, red, nir, swir1, swir2 = image[:6]
        epsilon = 1e-8

        ndvi = (nir - red) / (nir + red + epsilon)
        evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1 + epsilon))
        ndwi = (green - nir) / (green + nir + epsilon)
        ndbi = (swir1 - nir) / (swir1 + nir + epsilon)
        savi = ((nir - red) / (nir + red + 0.5 + epsilon)) * 1.5
        nbr = (nir - swir2) / (nir + swir2 + epsilon)
        evi2 = 2.5 * ((nir - red) / (nir + 2.4 * red + 1 + epsilon))
        msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2
        nmdi = (nir - (swir1 - swir2)) / (nir + (swir1 - swir2) + epsilon)
        ndi45 = (red - blue) / (red + blue + epsilon)
        si = (blue + green + red) / 3

        return np.stack([ndvi, evi, ndwi, ndbi, savi, nbr, evi2, msavi, nmdi, ndi45, si])

# Dataset and DataLoaders
dataset = SatelliteDataset('/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A/', data)
train_size = int(0.5 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ViT Model
model = timm.create_model("vit_base_patch16_224", in_chans=13, pretrained=True)
model.head = nn.Linear(model.head.in_features, 1)
model.to(device)

# Loss & Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


num_epochs = 100

scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    steps_per_epoch=len(train_loader),
    epochs=num_epochs
)



# Training loop
best_r2 = -np.inf
patience = 20
patience_counter = 0
train_history, val_history = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss, preds, true_vals = 0.0, [], []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validating", leave=False):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images).squeeze()
            val_loss += criterion(outputs, targets).item()
            preds.extend(outputs.cpu().numpy())
            true_vals.extend(targets.cpu().numpy())

    val_loss /= len(val_loader)
    val_r2 = r2_score(true_vals, preds)
    train_history.append(train_loss)
    val_history.append(val_loss)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R2: {val_r2:.4f}")

    if val_r2 > best_r2:
        best_r2 = val_r2
        patience_counter = 0
        torch.save(model.state_dict(), "../models/model_sentinel2coneval_vit_12b_20250801.pth")
    else:
        patience_counter += 1
    if patience_counter > patience:
        print("Early stopping")
        break

# Save training history
torch.save({'train_loss': train_history, 'val_loss': val_history}, "../models/history_sentinel2coneval_vit_12b_20250801.pth")

# Test evaluation
model.load_state_dict(torch.load("../models/model_sentinel2coneval_vit_12b_20250801.pth"))
model.eval()

test_preds, test_true = [], []
with torch.no_grad():
    for images, targets in tqdm(test_loader, desc="Testing", leave=False):
        if images.size(0) == 0:
            continue
        images, targets = images.to(device), targets.to(device)
        outputs = model(images).view(-1)
        test_preds.extend(outputs.cpu().numpy())
        test_true.extend(targets.cpu().numpy())

print(f"Test R²: {r2_score(test_true, test_preds):.4f}")
# Save test R² to CSV
output_path = "../data/sentinel2coneval_vit_12b_r2_test_20250801.csv"
with open(output_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Model", "Test_R2"])
    writer.writerow(["vit_base_patch16_224", f"{r2_score(test_true, test_preds):.6f}"])

print(f"Saved test R² to {output_path}")