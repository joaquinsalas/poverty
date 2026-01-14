# Graph-based Models

This directory contains **graph-based learning approaches** for estimating **mesh-level poverty indicators** from Earth observation data.  
Unlike pixel- or patch-centric models, these methods explicitly model **spatial relationships and neighborhood structure** between mesh cells.

 

---

## Directory structure

### `MCDLT/`
Graph-based models built on **multi-cell dependency learning**.

 

---

### `SDSGC/`
Graph convolutional models based on **spatially defined graph convolutions**.

 

---

## Common characteristics

Across all graph-model subfolders:

- Nodes: 469 m mesh grid cells
- Node features: Satellite-derived predictors or embeddings
- Edges: Spatial adjacency or distance-based connectivity
- Training: Supervised regression
- Evaluation: R² on held-out mesh cells



