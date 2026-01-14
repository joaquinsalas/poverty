import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#=============================================================
#import modules
import numpy as np
import pandas as pd
import rasterio
import cv2
from sklearn.decomposition import KernelPCA

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
#function to obtain the spectral indices
def spectral_indices(img):
    blue, green, red = img[1], img[2], img[3]
    nir, sw1, sw2 = img[7], img[10], img[11]
    eps = 1e-8
    ndvi  = (nir - red)  / (nir + red  + eps)
    evi   = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + eps)
    ndwi  = (green - nir) / (green + nir + eps)
    ndbi  = (sw1 - nir)  / (sw1 + nir + eps)
    savi  = 1.5 * (nir - red) / (nir + red + 0.5 + eps)
    nbr   = (nir - sw2)  / (nir + sw2 + eps)
    evi2  = 2.5 * (nir - red) / (nir + 2.4*red + 1 + eps)
    msavi = (2*nir + 1 - np.sqrt((2*nir + 1)*2 - 8*(nir - red))) / 2
    ndi45 = (red - blue) / (red + blue + eps)
    si    = (blue + green + red) / 3
    return np.stack([ndvi,evi,ndwi,ndbi,savi,nbr,evi2,msavi,ndi45,si])

#=============================================================
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

        #add spectral indices
        spectral = spectral_indices(img)
        img = np.concatenate([img, spectral], axis=0)

        #swap image channels
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 0, 1)
        x_imgs.append(img)
    return np.array(x_imgs)

#=============================================================
#function to reduce image bands using kernel PCA
def reduce_bands(image, n_components):
    height, width, bands = np.shape(image)
    img_flat = np.reshape(image, (height*width, bands))
    kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=None, n_jobs=-1)
    img_res = kpca.fit_transform(img_flat)
    img_res = np.reshape(img_res, (height, width, n_components))
    return img_res

#=============================================================
#function to normalize an image
scale_img = 255
def normalize_image(image):
    height, width, bands = np.shape(image)
    img_res = np.zeros((height, width, bands))
    for band in range(bands):
        xb = image[:,:,band]
        img_res[:,:,band] = scale_img * (xb - np.min(xb))/(np.max(xb) - np.min(xb))
    return img_res

#=============================================================
#load the images
nrec = len(y_ref)
inds = np.arange(0, nrec)
x_imgs = load_images(inds)

#=============================================================
#compute embeddings for the images
n_components = 3
x_feats = []
for image in x_imgs:
    x_im = reduce_bands(image, n_components)
    x_im = normalize_image(x_im)
    x_im = cv2.resize(x_im, (width,height), interpolation=cv2.INTER_CUBIC)
    x_im = np.reshape(x_im, (1, height, width, bands))
    feat = model.predict(x_im, verbose=0)
    x_feats.append(feat[0])

x_feats = np.array(x_feats)

#=============================================================
#save embeddings to numpy file
x_feats = np.reshape(x_feats, (len(x_feats),-1))
np.save('../data/x_efnb3_kpca_feats.npy', x_feats)
