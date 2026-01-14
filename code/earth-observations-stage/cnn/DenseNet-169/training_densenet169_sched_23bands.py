import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#=============================================================
#import modules
import numpy as np
import pandas as pd
import math
import rasterio
import random
import cv2

import tensorflow as tf
import albumentations as albu

from tensorflow.keras.models import Sequential
from tensorflow.keras import models, layers, losses, optimizers
#from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import DenseNet169

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
    nmdi  = (nir - (sw1 - sw2)) / (nir + (sw1 - sw2) + eps)
    ndi45 = (red - blue) / (red + blue + eps)
    si    = (blue + green + red) / 3
    return np.stack([ndvi,evi,ndwi,ndbi,savi,nbr,evi2,msavi,nmdi,ndi45,si])

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

#=============================================================
#load images
x_train = load_images(ind_train)
x_val = load_images(ind_val)

#poverty values
y_train = np.array(y_ref[ind_train])
y_val = np.array(y_ref[ind_val])

#=============================================================
#load DenseNet-169 model pre-trained with imagenet
height, width, channels = 224, 224, 23

cnn_source = DenseNet169(weights='imagenet', include_top=False, input_shape=(width, height, 3))

#new model without top layers
conv_base = DenseNet169(weights=None, include_top=False, input_shape=(width, height, channels))

#copy weights of the trained model to the new model adding regularization
penalty = 0.1 
regularizer = tf.keras.regularizers.l2(penalty)

for i in np.arange(3,len(conv_base.layers)):
    layer_source = cnn_source.get_layer(index=i)
    layer_dest = conv_base.get_layer(index=i)
    w = layer_source.get_weights()
    layer_dest.set_weights(w)
    for attr in ['kernel_regularizer']:
        if hasattr(layer_dest, attr):
            setattr(layer_dest, attr, regularizer)

#=============================================================
#add layer to have an output for the predictions of the mean and standard deviation
model = Sequential()
model.add( conv_base )
model.add( GlobalAveragePooling2D() )
model.add( Dropout(0.5) )
model.add( Dense(1, activation='sigmoid') )

#=============================================================
batch_size = 32

def resize_img(img, shape):
    return cv2.resize(img, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_CUBIC)

#data generator for training and validation
class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size=batch_size, dim=(height, width), channels=channels, augment=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.dim = dim
        self.channels = channels
        self.indexes = np.arange(0, len(self.x))
        self.augment = augment
        
    def __on_epoch_end(self):
        self.indexes = np.arange(0, len(self.x))
        np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx+1) * self.batch_size]
        
        #generate mini-batch of X
        X = []
        for i, ID in enumerate(batch_indexes):
            img = self.x[ID]
            img = resize_img(img, (self.dim[0], self.dim[1]))
            X.append(img)
        X = np.array(X)
      
        #generate mini-batch of y
        y = self.y[batch_indexes]
        
        #augmentation on the training dataset
        if self.augment==True:
            X = self.__augment_batch(X)
            
        return X, y
    
    #augmentation for one image
    def __random_transform(self, img):
        composition = albu.Compose([albu.HorizontalFlip(p=0.5),
                                    albu.VerticalFlip(p=0.5),
                                    albu.GridDistortion(p=0.2),
                                    albu.ElasticTransform(p=0.2)])
        return composition(image=img)['image']
    
    #augmentation for batch of images
    def __augment_batch(self, img_batch):
        for i in np.arange(0, len(img_batch)):
            img_batch[i] = self.__random_transform(img_batch[i])
        return img_batch
    
train_gen = DataGenerator(x_train, y_train, augment=True)
val_gen = DataGenerator(x_val, y_val, augment=False)

#=============================================================
#define class OneCycleLR
class OneCycleLR(tf.keras.callbacks.Callback):
    def __init__(self, max_lr, total_steps, div_factor=25, pct_start=0.3, final_div_factor=1e4):
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.div_factor = div_factor
        self.pct_start = pct_start
        self.final_div_factor = final_div_factor

        self.initial_lr = max_lr / div_factor
        self.min_lr_phase_two = self.initial_lr / final_div_factor

        self.step_num = 0
        self.phase_one_steps = int(self.pct_start * total_steps)
        self.phase_two_steps = total_steps - self.phase_one_steps

    def on_train_begin(self, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.initial_lr)

    def on_train_batch_end(self, batch, logs=None):
        self.step_num += 1

        if self.step_num <= self.phase_one_steps:
            # Linear increase in phase one
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (self.step_num / self.phase_one_steps)
        else:
            # Linear decrease in phase two
            lr = self.max_lr - (self.max_lr - self.min_lr_phase_two) * \
                 ((self.step_num - self.phase_one_steps) / self.phase_two_steps)

        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

#=============================================================
#compile the model
learning_rate = 1.0e-4
mse = tf.keras.losses.MeanSquaredError()
#opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
opt = tf.keras.optimizers.Adam()

model.compile(optimizer=opt, loss=mse)

#=============================================================
#fit the model
epochs = 100
steps_per_epoch = len(x_train) // batch_size
total_training_steps = epochs * steps_per_epoch
scheduler = OneCycleLR(max_lr=1e-3, total_steps=total_training_steps)

res = model.fit(train_gen,
                validation_data=val_gen,
                callbacks = [scheduler],
                epochs=epochs, verbose=1)

#=============================================================
#save training history
filename = 'training_history_densenet169_sched_23bands.csv'
train_loss = res.history['loss']
val_loss = res.history['val_loss']
df = pd.DataFrame({'train_loss':train_loss, 'val_loss':val_loss})
df.to_csv(filename)

#=============================================================
#saving the trained model
model.save('model_densenet169_sched_23bands.keras')













