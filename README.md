
# Fine-Scale Poverty Mapping from Satellite Imagery: A Comparative Study of Classical, CNN, Transformer, and Foundation Models

# Overview

This repository presents a remote sensing–based pipeline for poverty assessment using Sentinel-2 satellite imagery and machine learning models. The methodology integrates census-based socioeconomic indicators with satellite data to estimate poverty levels at higher spatial resolutions.

The pipeline is divided into two main stages:

## Census-based regression stage

A regression model is trained using census reference values corresponding to CONEVAL multidimensional poverty indicators at the municipal level. This model provides a consistent poverty estimation baseline.

## Satellite-based inference stage

In the second stage, Sentinel-2 satellite imagery is linked to the estimated poverty values, enabling poverty prediction at finer spatial resolutions using deep learning models.

# Reproducibility and Code Structure

This repository provides the source code required to replicate the experiments conducted for each of the evaluated models. For every architecture, an independent notebook is included.

In particular, the notebook based on EfficientNetB3 is proposed as the reference model, serving as the methodological baseline for comparison with the other architectures.

# Notebook Functionality

Each notebook implements the following components in a structured and reproducible manner:

1.- Preprocessing and normalization of Sentinel-2 multispectral imagery

2.- Computation of spectral indices and construction of multi-channel inputs

3.- Model-specific training configurations and optimization strategies

4.- Model performance evaluation using standardized statistical metrics (MSE and R²)

5.- Estimation of the computational environmental impact using CodeCarbon

# Requirements

Python ≥ 3.9

TensorFlow / PyTorch (model-dependent)

Rasterio, NumPy, Pandas

Albumentations

CodeCarbon

(See individual notebooks for detailed dependencies.)
