#import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

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

#poverty values
y_train = y_ref[ind_train]
y_test = y_ref[ind_test]

#======================================================
#read clusters
path = 'results_imgs_12b/'
num_clusters = np.arange(10,57)
nc = len(num_clusters)
y_cls = np.zeros((nrec, nc))
for k, ncl in enumerate(num_clusters):
    y_cls[:,k] = np.load(f'{path}clusters_cls_{ncl}.npy')

#======================================================
#assign a poverty value to each cluster
y_pval = []
for k, ncl in enumerate(num_clusters):
    yp = []
    for cls in range(ncl):
        yp.append(np.mean(y_train[y_cls[ind_train,k]==cls]))
    y_pval.append(yp)

#======================================================
#compute predictions for the test set
y_pred = np.zeros((len(ind_test), nc))
for k, ncl in enumerate(num_clusters):
    for cls in range(ncl):
        y_pred[y_cls[ind_test,k]==cls,k] = y_pval[k][cls]

#======================================================
#compute performance
sigma2 = np.var(y_test)
mse = np.zeros(nc)
R2 = np.zeros(nc)
for k in range(nc):
    mse[k] = np.mean((y_pred[:,k] - y_test)**2.0)
    R2[k] = 1.0 - mse[k]/sigma2

#======================================================
#plot results
fig = plt.figure(figsize=(6,4))
plt.rcParams['font.size'] = '12'
plt.grid('on')

ind = np.arange(0,nc)
indv = ind[np.isnan(R2)==False]

plt.plot(num_clusters[indv], R2[indv], linewidth=3)
plt.xlabel('number of clusters')
plt.ylabel(r'$R^2$', fontsize=16)
plt.title('Performance of SDSGC')
plt.show()



