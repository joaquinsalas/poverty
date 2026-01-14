# Convolutional Neural Networks (CNN)

This directory contains **convolutional neural network–based regressors** used to estimate **mesh-level poverty indicators** from **Sentinel-2 satellite imagery**.  
All models in this folder follow the same experimental protocol described in the article, enabling controlled comparison across CNN architectures.

The CNNs operate on multispectral image patches extracted around 469 m mesh cells and are trained in a fully supervised manner.

---

## Directory structure

### `DenseNet-169/`
DenseNet-based convolutional regressors.

---

### `EfficientNetB312-24b/`
EfficientNet-B3 models with **raw bands and spectral-index augmentation**.

---

### `efficientNetB3/`
EfficientNet-B3 baseline models using **raw Sentinel-2 bands only**.

---

### `resnet50/`
ResNet-50–based convolutional regressors.

---

## Common characteristics

Across all CNN subfolders:

- Input: Sentinel-2 multispectral image patches
- Target: Mesh-level poverty indicators derived from CPV 2020 ensemble predictions
- Data splits: Consistent train / validation / test partitions
- Optimization: Modern schedulers and early stopping
- Evaluation: R² as primary performance metric
- Optional interpretability and emissions analysis, depending on the model

---

