import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#======================================================
#import modules
#import modules
import numpy as np
import pandas as pd
import rasterio
import random
import cv2

from tensorflow.keras import models, saving
from tensorflow.keras.applications import DenseNet169

from codecarbon import EmissionsTracker

#======================================================
#load model
model = saving.load_model('../model_densenet169_sched_12bands.keras')

#======================================================
#read data file to load images according to the code
df = pd.read_csv('../ensemble_inferences_calidad_vivienda_2020.csv')
y_ref = df[[f"prediction_{i:02d}" for i in range(1, 31)]].mean(axis=1)
code = df['codigo']

print(len(y_ref))
print(np.min(y_ref), np.max(y_ref))

#======================================================
#function to load the images
path = '/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A/'
scale = 0.00005  #scale factor
def load_images(inds):
    x_imgs = []
    for idx in inds:
        filename = f'{path}{code[idx]}.tif'
        src = rasterio.open(filename)
        img = src.read()
        img = scale * img
        src.close()

        #swap image channels
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 0, 1)
        
        x_imgs.append(img)
        
    return np.array(x_imgs)

#======================================================
#create partition
random.seed(0)
nrec = len(y_ref)
ntrain = int(0.5*nrec)
nval = int(0.2*nrec)

#partition indices
ind = np.arange(0,nrec)
random.shuffle(ind)
ind_train = ind[0:ntrain]
ind_val = ind[ntrain:ntrain+nval]
ind_test = ind[ntrain+nval:nrec]

#repeat training data
repeats = 1
ind_train = np.repeat(ind_train, repeats, axis=0)
random.shuffle(ind_train)

print(len(ind_train), len(ind_val), len(ind_test))

#======================================================
#load images of the test set
x_test = load_images(ind_test)

#poverty values
y_test = np.array(y_ref[ind_test])

print(np.shape(x_test))
print(np.shape(y_test))

#======================================================
#function to compute predictions
height, width = (224, 224)
nsel = 32

def compute_predictions(x_imgs):
    nimgs = len(x_imgs)
    y_pred = []
    for i in np.arange(0, nimgs, nsel):
        imax = np.min((i+nsel,nimgs))
        xt = []
        for j in np.arange(i,imax):
            imr = cv2.resize(x_imgs[j], (width,height), interpolation=cv2.INTER_CUBIC)
            xt.append(imr)
        xt = np.array(xt)
        yp = model.predict(xt, verbose=0)
        for j in np.arange(0,len(yp)):
            y_pred.append( yp[j][0] )
    return y_pred

#======================================================
#compute predictions for the test set using CO2 tracker
from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()
try:
    y_pred = compute_predictions(x_test)
finally:
    tracker.stop()

#y_pred = compute_predictions(x_test)

print(np.shape(y_pred))


#======================================================
#save data
filename = 'predictions_densenet169_sched_12bands.csv'
df = pd.DataFrame({'ind_test':ind_test, 'y_test':y_test, 'y_pred':y_pred})
df.to_csv(filename, index=False)

#======================================================
#compute R^2
mse = np.mean((y_pred - y_test)**2.0)
sigma2 = np.mean((y_test - np.mean(y_test))**2.0)
R2 = 1.0 - mse/sigma2

print(f'MSE = {mse:0.6f}')
print(f'sigma^2 = {sigma2:0.6f}')
print(f'R^2 = {R2:0.4f}')

print('ok')
