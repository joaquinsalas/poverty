
# A Two-Stage Approach to Improve Poverty Mapping Spatial Resolution

# Overview

This repository presents a remote sensing–based pipeline for poverty assessment using Sentinel-2 satellite imagery and machine learning models. The methodology integrates census-based socioeconomic indicators with satellite data to estimate poverty levels at higher spatial resolutions.

The pipeline is divided into two main stages:

## Census-based regression stage
In the first stage, a regression framework is trained using census reference data associated with multidimensional poverty indicators at the municipal level. This stage establishes a consistent and institutionally aligned baseline for poverty estimation.

The implementation of this stage is available in the code/census-stage directory. It includes multiple regression approaches for mapping census variables to poverty indicators, such as neural networks (NN), support vector regression (SVR), and gradient-boosted decision trees (XGBoost). In addition, an ensemble strategy is implemented to combine the predictions of individual models, improving robustness and generalization performance. 

## Earth Observations-based inference stage

In the second stage, Sentinel-2 satellite imagery is linked to the poverty estimates obtained from the census-based regression stage, enabling prediction at finer spatial resolutions through image-based learning.

The models for this stage are implemented in the code/earth-observations-stage directory and use the spectral and spatial information contained in multispectral Sentinel-2 imagery. This stage incorporates a range of learning paradigms, including convolutional neural networks (CNNs), transformer-based architectures, graph-based methods, and Capsule Attention Networks (CAN). Together, these models capture complex spectral–spatial relationships in satellite data to support scalable, high-resolution poverty mapping.

# Reproducibility and code structure

Download the BaseDatos_Sentinel2A folder from Google Drive, which contains the images used for the model’s training, validation, and test datasets.

Once the download is complete, place the BaseDatos_Sentinel2A folder inside the project’s data directory, preserving the original file structure.

Afterward, simply run the program, as the code is already configured to automatically access the data from this location.

# Notebook functionality

Each notebook implements the following components in a structured and reproducible manner:

1.- Preprocessing and normalization of Sentinel-2 multispectral imagery.

2.- Computation of spectral indices and construction of multi-channel inputs.

3.- Model-specific training configurations and optimization strategies.

4.- Model performance evaluation using standardized statistical metrics (MSE and R²).

5.- Estimation of the computational environmental impact using CodeCarbon.

# Requirements

Python ≥ 3.9

TensorFlow / PyTorch (model-dependent)

Rasterio, NumPy, Pandas

Albumentations

CodeCarbon

(See individual notebooks for detailed dependencies.)

# Weights
The weights for the model can be downloaded from https://drive.google.com/drive/folders/1zfHciQVy74tDgzzAtkCHHnb0wp3xAF0L?usp=sharing


## Paper (PDF)

- **Download:** [MDPI Remote Sensing paper (PDF)](https://github.com/joaquinsalas/poverty/tree/main/docs/MDPI_remote_sensing.pdf)

- @misc{salas2026poverty,
  author       = {Joaquín Salas, Marivel Zea-Ortiz, Pablo Vera, and Danielle Wood},
  title        = {A Two-Stage Approach to Improve Poverty Mapping Spatial Resolution},
  year         = {2026},
  howpublished = {under review Remote Sensing},
  note         = {Accessed: 2026-01-15}
}










