#import modules
import numpy as np
import pandas as pd
import rasterio
import cv2

#=============================================================
#read data file
df = pd.read_csv('ensemble_inferences_calidad_vivienda_2020.csv')
y_ref = df[[f"prediction_{i:02d}" for i in range(1, 31)]].mean(axis=1)
code = df['codigo']

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
#resize images
height, width = (24, 24)
x_res = []
for image in x_imgs:
    x_im = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    x_res.append(x_im)
x_res = np.array(x_res)

#reshape images to one-dimensional vectors
x_res = np.reshape(x_res, (len(x_res), -1))

#=============================================================
#save image features to numpy file
np.save('../data/x_imgs_feats.npy', x_res)

