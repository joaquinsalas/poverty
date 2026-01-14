# Transformer-based Models

This directory contains **transformer-based architectures** used to estimate **mesh-level poverty indicators** from **Sentinel-2 satellite imagery**.  
These models rely on attention mechanisms to capture long-range dependencies within image patches, without relying on convolutional inductive biases.


---

## Directory structure

### `Swin/`
Models based on the **Swin Transformer** architecture.


---

### `ViT/`
Models based on the **Vision Transformer (ViT)** architecture.



---

## Common characteristics

Across all transformer subfolders:

- Input: Multispectral Sentinel-2 image patches
- Architecture: Pure attention-based vision models
- Training: Supervised regression on mesh-level targets
- Evaluation: R² on held-out mesh cells
- Comparison: Benchmarked against CNNs, graph models, and foundation models


