# MCDLT 

This folder contains a **Python (v3.10.13)** implementation of the **MCDLT (Multi-order Graph based Clustering via Dynamical Low-Rank Tensor approximation)** algorithm.

The MCDLT algorithm was originally introduced in the paper:

> **Multi-order Graph Based Clustering via Dynamical Low-Rank Tensor Approximation**  
> Nian Wang, Zhigao Cui, Aihua Li, Yuanliang Xue, Rong Wang, and Feiping Nie  
> *Neurocomputing*, 2025

This implementation is based on the authors’ original **MATLAB** code, which is publicly available at:  
https://github.com/NianWang-HJJGCDX/MCDLT

The purpose of this Python version is to facilitate integration with the datasets and analysis scripts used throughout this repository.

---

## Scripts

### `mcdlt_imgs_12b.py`
This script executes **MCDLT** algorithm using **12,694** samples loaded from `x_imgs_feats.npy`. The algorithm is run repeatedly using a range of cluster counts from **10 to 100**.

**Input**
- **X**: A matrix containing **Sentinel-2 image patches** of size **24 × 24 pixels** with **12 spectral bands**.
- Each patch is flattened into a **6,912-dimensional feature vector** (24 × 24 × 12).

**Output**
- For each cluster count, the script generates an array `y_cls` of length **12,694**, where each element indicates the cluster assignment of the corresponding sample.
- The resulting cluster label arrays are saved as NumPy files named `clusters_cls_10.npy` through `clusters_cls_100.npy`, one file per cluster count, in the `results_imgs_12b` folder.

---

### `mcdlt_efnb3_rgb.py`
This script executes **MCDLT** algorithm using feature vectors derived from **Sentinel-2 RGB imagery**.

**Features**
- Input images consist of the **RGB bands** of Sentinel-2 patches.
- Each patch is passed through an **EfficientNet-B3** model pretrained on ImageNet.
- The resulting **1,536-dimensional embedding** is used as the feature representation for clustering.

#### Input
- **X**: A matrix containing **EfficientNet-B3** embeddings from **Sentinel-2 image patches**.

#### Output
- For each cluster count, the script generates an array `y_cls` of length **12,694**. Arrays are saved in NumPy files named `clusters_cls_10.npy` through `clusters_cls_100.npy` in the `results_efnb3_rgb` folder.

---

### `mcdlt_efnb3_kpca.py`
This script executes **MCDLT** algorithm using feature vectors derived from **Sentinel-2 12 bands + 10 spectral indices**.

**Features**
- The 22-band Sentinel-2 data are projected into **3 channels** using **Kernel PCA (KPCA)** with a **Radial Basis Function (RBF) kernel**.
- The resulting 3-channel representation is treated as an RGB-like image and passed through a pretrained **EfficientNet-B3** model.
- Each image patch is encoded as a **1,536-dimensional embedding vector**.

#### Output
- For each cluster count, the script generates an array `y_cls` of length **12,694**. Arrays are saved in NumPy files named `clusters_cls_10.npy` through `clusters_cls_100.npy` in the `results_efnb3_kpca` folder.

---

### `plot_R2_mcdlt.py`

---

### `mcdlt_co2_emissions.py`

---









