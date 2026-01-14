#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"


# In[4]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import random
import rasterio
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from keras.saving import register_keras_serializable


# In[5]:


# ----------------------
# Utils
# ----------------------
def window_partition(x, window_size):
    B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    x = tf.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, [-1, window_size, window_size, C])
    return windows

def window_reverse(windows, window_size, H, W):
    B = tf.shape(windows)[0] // (H * W // window_size // window_size)
    x = tf.reshape(windows, [B, H // window_size, W // window_size, window_size, window_size, -1])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [B, H, W, -1])
    return x


# In[6]:


# ----------------------
# Custom Layers
# ----------------------
@register_keras_serializable()
class MLPBlock(layers.Layer):
    def __init__(self, hidden_units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.mlp_layers = []
        for units in hidden_units:
            self.mlp_layers.append(layers.Dense(units, activation=tf.nn.gelu))
            self.mlp_layers.append(layers.Dropout(dropout_rate))

    def call(self, inputs):
        x = inputs
        for layer in self.mlp_layers:
            x = layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "dropout_rate": self.dropout_rate
        })
        return config


# In[7]:


@register_keras_serializable()
class WindowAttention(layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)

    def call(self, x):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = tf.unstack(qkv, num=3)

        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "window_size": self.window_size,
            "num_heads": self.num_heads
        })
        return config


# In[8]:


@register_keras_serializable()
class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = layers.Dropout(drop)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPBlock([mlp_hidden_dim, dim], dropout_rate=drop)

    def call(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "input_resolution": self.input_resolution,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "shift_size": self.shift_size,
            "mlp_ratio": self.mlp_ratio
        })
        return config


# In[9]:


# ----------------------
# Swin Transformer Regression Model
# ----------------------
def build_swin_regression_model(input_shape=(96, 96, 23), embed_dim=48, window_size=6, num_heads=4, num_blocks=2):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(embed_dim, kernel_size=4, strides=4)(inputs)
    x = layers.Reshape((input_shape[0] // 4 * input_shape[1] // 4, embed_dim))(x)

    for i in range(num_blocks):
        x = SwinTransformerBlock(
            dim=embed_dim,
            input_resolution=(input_shape[0] // 4, input_shape[1] // 4),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0 if i % 2 == 0 else window_size // 2,
            drop=0.0,
            attn_drop=0.0
        )(x)

    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    output = layers.Dense(1)(x)

    return keras.Model(inputs, output)


# In[10]:


# ----------------------
# Function to compute spectral indices
# ----------------------
def spectral_indices(im):
    blue  = im[1]
    green = im[2]
    red   = im[3]
    nir   = im[7]
    sw1   = im[10]
    sw2   = im[11]

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


# In[11]:


# ----------------------
# Load and Prepare Data
# ----------------------
df = pd.read_csv('../data/ensemble_inferences_calidad_vivienda_2020.csv')
y_ref = df[[f"prediction_{i:02d}" for i in range(1, 31)]].mean(axis=1)
code = df['codigo']

path = '../data/sentinel_images/BaseDatos_Sentinel2A/'
scale = 0.00005

def load_images(inds):
    x_imgs = []
    for idx in inds:
        filename = f'{path}{code[idx]}.tif'
        with rasterio.open(filename) as src:
            img = src.read() * scale
        indices = spectral_indices(img)
        img_stack = np.concatenate([img, indices], axis=0)
        img_stack = np.swapaxes(img_stack, 0, 2)
        img_stack = np.swapaxes(img_stack, 0, 1)
        x_imgs.append(img_stack)
    return np.array(x_imgs)

random.seed(0)
nrec = len(y_ref)
ntrain = int(0.5 * nrec)
nval = int(0.2 * nrec)
ind = np.arange(0, nrec)
random.shuffle(ind)
ind_train = np.repeat(ind[0:ntrain], 1)
ind_val = ind[ntrain:ntrain+nval]
ind_test = ind[ntrain+nval:nrec]

x_train = load_images(ind_train)
x_val = load_images(ind_val)
y_train = np.array(y_ref[ind_train], dtype=np.float32)
y_val = np.array(y_ref[ind_val], dtype=np.float32)


# In[12]:


print(x_train.shape)  # Debería ser (n, 96, 96, 12)


# In[11]:


import tensorflow as tf
import math
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --------------------------
# OneCycleLR Callback
# --------------------------
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
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.Variable):
            tf.keras.backend.set_value(lr, self.initial_lr)
        else:
            self.model.optimizer.learning_rate = self.initial_lr

    def on_train_batch_end(self, batch, logs=None):
        self.step_num += 1

        if self.step_num <= self.phase_one_steps:
            lr_value = self.initial_lr + (self.max_lr - self.initial_lr) * (self.step_num / self.phase_one_steps)
        else:
            lr_value = self.max_lr - (self.max_lr - self.min_lr_phase_two) * \
                       ((self.step_num - self.phase_one_steps) / self.phase_two_steps)

        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.Variable):
            tf.keras.backend.set_value(lr, lr_value)
        else:
            self.model.optimizer.learning_rate = lr_value

# --------------------------
# Hiperparámetros
# --------------------------
max_lr = 1e-3
epochs = 100
batch_size = 32
steps_per_epoch = len(x_train) // batch_size
total_steps = steps_per_epoch * epochs

# --------------------------
# Callbacks
# --------------------------
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)

onecycle_callback = OneCycleLR(
    max_lr=max_lr,
    total_steps=total_steps,
    div_factor=25,
    pct_start=0.3,
    final_div_factor=1e4
)

checkpoint_best = ModelCheckpoint(
    filepath='model_best_val_loss_24bands_swin.keras',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

# --------------------------
# Modelo y entrenamiento
# --------------------------
model = build_swin_regression_model(input_shape=(96, 96, 23))  # asegúrate que esta función está definida
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

res = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stop, onecycle_callback, checkpoint_best],
    verbose=1
)


# In[14]:


# ----------------------
# Save Model and History
# ----------------------
pd.DataFrame({
    'train_loss': res.history['loss'],
    'val_loss': res.history['val_loss']
}).to_csv('training_history_swin.csv', index=False)
model.save('model_final_swing_24band.keras')


# In[16]:


# ----------------------
# Evaluate Model and Predict
# ----------------------
print("\n Loading trained model...")
model = tf.keras.models.load_model(
    "model_best_val_loss_24bands_swin.keras",
    custom_objects={
        'SwinTransformerBlock': SwinTransformerBlock,
        'WindowAttention': WindowAttention,
        'MLPBlock': MLPBlock
    }
)

x_test = load_images(ind_test)
y_test = np.array(y_ref[ind_test])
code_test = code[ind_test].reset_index(drop=True)

# Predictions
print("Generating predictions...")
y_pred = model.predict(x_test, verbose=1).flatten()
y_true = y_test

# Evaluation with sklearn
from sklearn.metrics import mean_squared_error, r2_score
mse_sklearn = mean_squared_error(y_true, y_pred)
r2_sklearn = r2_score(y_true, y_pred)

# Manual evaluation
mse = np.mean((y_pred - y_true) ** 2.0)
sigma2 = np.mean((y_true - np.mean(y_true)) ** 2.0)
r2_manual = 1.0 - mse / sigma2

# Display results
print(f"\n MSE (manual): {mse:.4f}")
print(f"Sigma² (variance of y_true): {sigma2:.4f}")
print(f"R² (manual): {r2_manual:.4f}")
print(f"\n Verification with sklearn:")
print(f"   MSE (sklearn): {mse_sklearn:.4f}")
print(f"   R²  (sklearn): {r2_sklearn:.4f}")

plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.5, s=10)
plt.plot(
    [y_true.min(), y_true.max()],
    [y_true.min(), y_true.max()],
    'r--'
)
plt.xlabel("Ground truth")
plt.ylabel("Prediction")
plt.title("True vs. predicted values")
plt.grid(True)
plt.tight_layout()
plt.savefig("scatter_plot_predictions_vs_true.png", dpi=300)
plt.show()

df_pred = pd.DataFrame({
    "codigo": code_test,
    "true": y_true,
    "prediction": y_pred
})
df_pred.to_csv("predictions_test_set_swin_24bands.csv", index=False)

print("Predictions exported to predictions_test_set_swin_24bands.csv")


# In[ ]:




