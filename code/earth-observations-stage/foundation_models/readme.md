# Foundation Models

This directory contains **foundation-model–based pipelines** used to estimate **mesh-level poverty indicators** from **Sentinel-2 satellite imagery**.  
These models rely on large-scale pretraining on Earth observation data and are adapted to downstream regression tasks at fine spatial resolution.


---

## Directory structure

### `DINOv3/`
Vision Transformer–based models built on **DINOv3-style self-supervised representations**.


---

### `alphaEarth/`
Earth observation foundation models derived from the **AlphaEarth** representation pipeline.


---

## Common characteristics

Across all foundation-model subfolders:

- Input: Sentinel-2 image patches aligned to 469 m mesh cells
- Backbone: Large pretrained vision models
- Training: Lightweight fine-tuning or regression heads
- Evaluation: R² on held-out mesh cells
- Inference: Scalable batch processing

---

## Scope

This directory isolates **foundation models** from other model families such as:
- Classical regressors (XGB, SVR)
- CNNs
- Transformers trained from scratch
- Capsule attention and graph-based models

This separation enables a clear analysis of **performance, transferability, and computational cost** across modeling paradigms.

