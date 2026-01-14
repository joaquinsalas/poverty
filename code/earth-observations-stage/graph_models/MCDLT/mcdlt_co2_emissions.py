#import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from codecarbon import EmissionsTracker

#======================================================
#read data file
df = pd.read_csv('../data/ensemble_inferences_calidad_vivienda_2020.csv')
y_ref = df[[f"prediction_{i:02d}" for i in range(1, 31)]].mean(axis=1)

#======================================================
#create partition
random.seed(0)
nrec = len(y_ref)
ntrain = int(0.5*nrec)

#partition indices
ind = np.arange(0,nrec)
random.shuffle(ind)
ind_train = ind[0:ntrain]
ind_test = ind[ntrain:nrec]

#======================================================
#poverty values
y_train = y_ref[ind_train]
y_test = y_ref[ind_test]

#======================================================
#read clusters
num_clusters = 44
y_cls = np.load(f'results_efnb3_rgb/clusters_cls_{num_clusters}.npy')
y_cls_train = y_cls[ind_train]
y_cls_test = y_cls[ind_test]

#======================================================
#function to compute predictions
def compute_predictions():
    #assign a poverty value to each cluster
    y_pval = np.zeros(num_clusters)
    for k in range(num_clusters):
        y_pval[k] = np.mean(y_train[y_cls_train==k])
        
    #compute predictions for the test set
    y_pred = np.zeros(len(ind_test))
    for k in range(num_clusters):
        y_pred[y_cls_test==k] = y_pval[k]
    return y_pred

#======================================================
#compute predictions for the test set using CO2 tracker
tracker = EmissionsTracker()
tracker.start()
try:
    y_pred = compute_predictions()
finally:
    tracker.stop()
    
#=====================================
#compute performance
mse = np.mean((y_pred - y_test)**2.0)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_pred - y_test))
sigma2 = np.var(y_test)
R2 = 1.0 - mse/sigma2

print(f'RMSE = {rmse:0.6f}')
print(f'MAE = {mae:0.6f}')
print(f'sigma^2 = {sigma2:0.6f}')
print(f'R^2 = {R2:0.4f}')

print('ok')

