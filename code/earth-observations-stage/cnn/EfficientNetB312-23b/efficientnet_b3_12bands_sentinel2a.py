#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[2]:


import numpy as np
import pandas as pd
import rasterio
import cv2
import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from sklearn.model_selection import train_test_split
import albumentations as albu
from codecarbon import EmissionsTracker


# In[3]:


# ------------------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------------------
df = pd.read_csv('../../../../data/ensemble_inferences_calidad_vivienda_2020.csv')
y_ref = df[[f"prediction_{i:02d}" for i in range(1, 31)]].mean(axis=1)
code = df['codigo']

print(len(y_ref))
print(np.min(y_ref), np.max(y_ref))


# In[4]:


def spectral_indices(im):
    # Asegúrate de que im.shape == (12, H, W)
    blue  = im[1]   # B2
    green = im[2]   # B3
    red   = im[3]   # B4
    nir   = im[7]   # B8
    sw1   = im[10]  # B11
    sw2   = im[11]  # B12

    eps = 1e-8
    ndvi  = (nir - red)  / (nir + red + eps)
    evi   = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + eps)
    ndwi  = (green - nir) / (green + nir + eps)
    ndbi  = (sw1 - nir) / (sw1 + nir + eps)
    savi  = 1.5 * (nir - red) / (nir + red + 0.5 + eps)
    nbr   = (nir - sw2) / (nir + sw2 + eps)
    evi2  = 2.5 * (nir - red) / (nir + 2.4*red + 1 + eps)
    msavi = (2*nir + 1 - np.sqrt((2*nir + 1)**2 - 8*(nir - red))) / 2
    nmdi  = (nir - (sw1 - sw2)) / (nir + (sw1 - sw2) + eps)
    ndi45 = (red - blue) / (red + blue + eps)
    si    = (blue + green + red) / 3

    return np.stack([ndvi, evi, ndwi, ndbi, savi, nbr, evi2, msavi, nmdi, ndi45, si])


# In[5]:


#function to load the images
path = '../../../../data/BaseDatos_Sentinel2A/'
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
        #spectral = spectral_indices(img)
        #img = np.concatenate([img, spectral], axis=0)

        #swap image channels
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 0, 1)
        
        x_imgs.append(img)
        
    return np.array(x_imgs)


# In[6]:


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


# In[8]:


#load images
x_train = load_images(ind_train)
x_val = load_images(ind_val)

print(np.shape(x_train))
print(np.shape(x_val))


# In[10]:


#poverty values
y_train = np.array(y_ref[ind_train])
y_val = np.array(y_ref[ind_val])

print(np.shape(y_train), np.shape(y_val))


# In[11]:


# ------------------------------------------------------------------------------
# 4. Modelo EfficientNetB3 adaptado a 12 canales
# ------------------------------------------------------------------------------
def build_custom_efficientnet_b3_partial(input_shape=(224, 224, 12), l2_penalty=0.01):
    cnn_source = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    conv_base = EfficientNetB3(weights=None, include_top=False, input_shape=input_shape)

    transfer_count = 0
    for i in range(5, len(conv_base.layers)):
        try:
            layer_source = cnn_source.get_layer(index=i + 1)
            layer_target = conv_base.get_layer(index=i)
            ws, wt = layer_source.get_weights(), layer_target.get_weights()
            if ws and wt and all(w1.shape == w2.shape for w1, w2 in zip(ws, wt)):
                layer_target.set_weights(ws)
                transfer_count += 1
        except Exception as e:
            print(f"⚠️  Skipped layer {i} ({layer_target.name if 'layer_target' in locals() else '?'}) — {e}")

   
    print(f"Weights transferred to {transfer_count} compatible layers.")


    model = Sequential([
        conv_base,
        GlobalAveragePooling2D(),
        Dropout(0.6),
        Dense(1, activation='sigmoid')
    ])
    return model

model = build_custom_efficientnet_b3_partial(input_shape=(224, 224, 12))


# In[12]:


batch_size = 32

def resize_img(img, shape):
    return cv2.resize(img, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_CUBIC)


height = 224
width = 224
channels = 12 


# In[13]:


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
                                    albu.RandomRotate90(p=0.5),
                                    albu.RandomBrightnessContrast(p=0.5),
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


x_sample, y_sample = train_gen[0]
print(f"Input batch size: {x_sample.shape}")
print(f"Target batch size: {y_sample.shape}")


# In[18]:


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

#compile the model
learning_rate = 1.0e-4
mse = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#early stopping to monitor the validation loss and avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)

#reducing learning rate on plateau
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr= 1e-7, verbose=1)

checkpoint_best = ModelCheckpoint(
    filepath='model_best_val_loss_12bands_efficientnetB3.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

model.compile(optimizer=opt, loss=mse)


# In[19]:


#fit the model
epochs = 1
res = model.fit(train_gen,
                validation_data=val_gen,
                callbacks = [early_stop, rlrop, checkpoint_best],
                epochs=epochs, verbose=1)


# In[20]:


# ------------------------------------------------------------------------------
# 7. Save history and model
# ------------------------------------------------------------------------------
pd.DataFrame({
    'train_loss': res.history['loss'],
    'val_loss': res.history['val_loss']
}).to_csv('training_history_efficientnetb3_12bands.csv', index=False)

model.save('model_efficientnetb3_12bands.h5')


# In[21]:


# ------------------------------------------------------------------------------
# 11. Evaluate the model and generate predictions on the test set
# ------------------------------------------------------------------------------
print("\n Loading trained model...")
model = tf.keras.models.load_model(
    "model_best_val_loss_12bands_efficientnetB3.h5"
)

# Cargar datos de test
x_test = load_images(ind_test)
y_test = np.array(y_ref[ind_test])
code_test = code[ind_test].reset_index(drop=True)

# Crear generador de test
test_gen = DataGenerator(x_test, y_test, augment=False)

# Predictions
print("Generating predictions...")
y_pred = model.predict(test_gen, verbose=1).flatten()
y_true = y_test

# ------------------------------------------------------------------------------
# 12. METRICS (sklearn + manual)
# ------------------------------------------------------------------------------
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# --- sklearn ---
mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)

# --- Manual R² (verification) ---
sigma2 = np.mean((y_true - np.mean(y_true)) ** 2.0)
r2_manual = 1.0 - mse / sigma2

# Display results
print("\n EVALUATION METRICS (TEST)")
print(f"► R² (sklearn): {r2:.4f}")
print(f"► R² (manual) : {r2_manual:.4f}")
print(f"► MAE         : {mae:.4f}")
print(f"► RMSE        : {rmse:.4f}")
print(f"► N           : {len(y_true)}")

# ------------------------------------------------------------------------------
# 13. VISUALIZATION
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.5, s=10)
plt.plot(
    [y_true.min(), y_true.max()],
    [y_true.min(), y_true.max()],
    'r--'
)
plt.xlabel("Ground truth")
plt.ylabel("Prediction")
plt.title("True vs. predicted values (TEST)")
plt.grid(True)
plt.tight_layout()
plt.savefig("scatter_plot_predictions_vs_true.png", dpi=300)
plt.show()

# ------------------------------------------------------------------------------
# 14. EXPORT CSV
# ------------------------------------------------------------------------------
df_pred = pd.DataFrame({
    "codigo": code_test,
    "true": y_true,
    "prediction": y_pred
})

df_pred.to_csv(
    "predictions_test_efficientnet_b3_12bands_sentinel2a.csv",
    index=False
)

print("✅ Predictions exported successfully")


# In[ ]:




