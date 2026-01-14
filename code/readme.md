# Code

This directory contains the **core implementation of the poverty-mapping pipeline**, organized into distinct stages that reflect the methodology described in the article.

The codebase is structured to separate **census-based modeling** from **Earth observation–based modeling**.

---

## Directory structure


### `census-stage/`
Census-based modeling stage.

This directory contains code that operates exclusively on **census and administrative data**, including:
- Feature engineering from CPV variables
- Classical regression models
- Ensemble construction
- Baseline poverty indicator estimation

 

---

### `earth-observations-stage/`
Earth observation–based modeling stage.

This directory contains all models that use **Sentinel-2 satellite imagery** and derived features to estimate mesh-level poverty indicators.  
It includes multiple model families, such as:
- Convolutional neural networks
- Transformer-based architectures
- Graph-based models
- Foundation models
- Classical regressors (e.g., XGBoost)

 



