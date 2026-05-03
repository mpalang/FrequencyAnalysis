# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 14:47:32 2025

@author: Mo
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
models = keras.models
optimizers = keras.optimizers
callbacks = keras.callbacks
from matplotlib import pyplot as plt
from pathlib import Path

exp_data = np.genfromtxt(Path(Path(__file__).parent,'test_data.csv'),delimiter=',')
wl = np.genfromtxt(Path(Path(__file__).parent,'test_data_wl.csv'),delimiter=',')
t = np.genfromtxt(Path(Path(__file__).parent,'test_data_t.csv'),delimiter=',')



#%

# ---- generate dataset with varying t grids ----
def make_signal(t, A, f, phi,noise=0.1):
    y = A * np.sin(2*np.pi*f*t + phi)
    
    y += np.random.normal(scale=noise, size=t.shape)
    return y*np.heaviside(t,0)

def sample_dataset(t,n_signals=20000):
    X = np.zeros((n_signals, len(t)), dtype=np.float32)
    Y = np.zeros((n_signals, 3), dtype=np.float32)
    
    for i in range(n_signals):
        
        # Define Ranges:
        A = np.random.uniform(0.1, 3)
        f = np.random.uniform(0.02, 0.08)
        phi = np.random.uniform(0, 2*np.pi)
          
        y = make_signal(t, A, f, phi, noise=0.1)
        
        X[i] = y
        Y[i] = [A, f, phi]
        
    # Split
    n_train = int(0.8 * n_signals)
    return (X[:n_train], X[n_train:],
            Y[:n_train], Y[n_train:])

#% Train Model
N = 500
t_int=np.linspace(min(t),max(t),N)
X_train, X_val, Y_train, Y_val = sample_dataset(t_int)

#%
model = tf.keras.Sequential([
    layers.Input((N,)),
    layers.Dense(300, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(200, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(100, activation='relu', kernel_initializer='he_normal'),
    # layers.Dense(100, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(3, activation='linear')   
    ])
model.summary()
#%
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    factor=0.5, patience=5, verbose=1
)   
model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
          epochs=50, batch_size=100,callbacks=[lr_scheduler],
          shuffle=True)


#% Test Model

def predict_params(model, y, t):
    y = np.array(y, dtype=np.float32).reshape(1, -1)
    A, f, phi = model.predict(y)[0]

    y_fit = make_signal(t, A, f, phi, noise=0)

    plt.figure()
    plt.plot(t, y.flatten(), 'o', ms=3, label='data')
    plt.plot(t, y_fit, lw=2, label='fit')
    plt.legend()
    plt.show()
    
    return A,f,phi

A,f, phi = predict_params(model, make_signal(t_int,1,0.05,0), t_int)
























