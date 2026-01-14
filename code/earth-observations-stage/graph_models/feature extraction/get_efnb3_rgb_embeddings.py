import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#=============================================================
#import modules
import numpy as np
import pandas as pd
import rasterio
import cv2

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB3

#=============================================================
#load EfficientNet model pre-trained with imagenet
height, width, bands = 224, 224, 3
cnn_source = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(width,height,bands))

#build the model
model = Sequential()
model.add( cnn_source )
model.add( GlobalAveragePooling2D() )

#=============================================================
#read data file
df = pd.read_csv('ensemble_inferences_calidad_vivienda_2020.csv')
y_ref = df[[f"prediction_{i:02d}" for i in range(1, 31)]].mean(axis=1)
code = df['codigo']

#=============================================================
#function to load the images
path = '/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A/'
scale = 0.00005  #scale factor
rgb_bands = (4,3,2)
def load_images(inds):
    x_imgs = []
    for idx in inds:
        filename = f'{path}{code[idx]}.tif'
        src = rasterio.open(filename)
        img = src.read(rgb_bands)
        img = scale * img
        src.close()

        #swap image channels
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 0, 1)
        
        x_imgs.append(img)
        
    return np.array(x_imgs)

#=============================================================
#load the images
nrec = len(y_ref)
inds = np.arange(0, nrec)
x_imgs = load_images(inds)

#=============================================================
#compute embeddings for the images
scale_img = 255
x_feats = []
for image in x_imgs:
    x_im = scale_img * image / np.max(image)
    x_im = cv2.resize(x_im, (width,height), interpolation=cv2.INTER_CUBIC)
    x_im = np.reshape(x_im, (1, height, width, bands))
    feats = model.predict(x_im, verbose=0)
    x_feats.append(feats)

x_feats = np.array(x_feats)

#=============================================================
#save embeddings to numpy file
x_feats = np.reshape(x_feats, (len(x_feats),-1))
np.save('../data/x_efnb3_rgb_feats.npy', x_feats)

