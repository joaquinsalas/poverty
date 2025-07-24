# poverty
Poverty assessment from remote sensing, using Sentinel-2, and machine learning


The pipeline is divided in two. First, there is a regressor that evaluates census reference values to CONEVAL indicators in their multidimensional poverty model at municipal level.
The regressor thus constructed is used to estimate the poverty at higher resolution levels. Then, there is a second stage where the satellite images are related to these poverty assessments.

