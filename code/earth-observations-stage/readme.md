# Earth-observations stage

This directory contains the **Earth Observation modeling stage** of the poverty estimation pipeline.  
It groups all satellite-based regression models used to map mesh-level socioeconomic indicators from remote sensing data, following the taxonomy and experimental design described in the article.

Each subdirectory corresponds to a **model family**, with consistent data splits, targets, and evaluation protocols to enable controlled comparison across architectures.

---

## Directory structure

### `capsule_attention/`
Capsule-based neural architectures for regression from satellite imagery.

---

### `cnn/`
Convolutional Neural Network baselines.

---

### `transformers/`
Vision Transformer (ViT)–based regression models.

---

### `foundation_models/`
Large pretrained or foundation-model-based pipelines applied to Earth observation data.

---

### `graph_models/`
Graph-based formulations operating on spatial or relational structures.

---

### `feature_extraction/`
Intermediate feature-generation utilities.

---

### `xgb/`
Tree-based regression models using handcrafted or flattened satellite features.


---

## Notes

- All folders follow a **common target definition** based on CPV 2020 ensemble poverty indicators.
- Scripts assume relative paths to shared `data/`, `models/`, and `figures/` directories.
- Training, inference, interpretability, and emissions tracking are kept explicit and reproducible.
- The structure mirrors the **model taxonomy and experimental comparisons** presented in the article.

---


