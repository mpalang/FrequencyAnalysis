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
from scipy import fft


class FrequencyAnalysis:
    def __init__(self):
        self.load_data()
        
        
        # self.make_model()
        
        # self.A,self.f, self.phi = self.predict_params()


# =============================================================================
#     #%functions
# =============================================================================
    def load_data(self):
        self.exp_data = np.genfromtxt(Path(Path(__file__).parent,'test_data.csv'),delimiter=',')
        self.wl = np.genfromtxt(Path(Path(__file__).parent,'test_data_wl.csv'),delimiter=',')
        self.t = np.genfromtxt(Path(Path(__file__).parent,'test_data_t.csv'),delimiter=',')
        
        self.N = 5000
        self.t_int=np.linspace(min(self.t),max(self.t),self.N)
    
    # ---- generate dataset with varying t grids ----
    def make_signal(self,t, A, f, phi,noise=0.1):
        y = A * np.sin(2*np.pi*f*t + phi)
        
        y += np.random.normal(scale=noise, size=t.shape)
        return y*np.heaviside(t,0)
    
    def sample_dataset(self,t,n_signals=20000):
        X = np.zeros((n_signals, len(t)), dtype=np.float32)
        Y = np.zeros((n_signals, 3), dtype=np.float32)
        
        for i in range(n_signals):
            
            # Define Ranges:
            A = np.random.uniform(0.1, 3)
            f = np.random.uniform(0.02, 0.08)
            phi = np.random.uniform(0, 2*np.pi)
              
            y = self.make_signal(t, A, f, phi, noise=0.1)
            
            X[i] = y
            Y[i] = [A, f, phi]
            
        # Split
        n_train = int(0.8 * n_signals)
        return (X[:n_train], X[n_train:],
                Y[:n_train], Y[n_train:])
    

    
    #%
    def make_model(self):
        #% Train Model

        X_train, X_val, Y_train, Y_val = self.sample_dataset(self.t_int)
        
        self.model = tf.keras.Sequential([
                layers.Input((self.N,)),
                layers.Dense(300, activation='relu', kernel_initializer='he_normal'),
                layers.Dense(200, activation='relu', kernel_initializer='he_normal'),
                layers.Dense(100, activation='relu', kernel_initializer='he_normal'),
                # layers.Dense(100, activation='relu', kernel_initializer='he_normal'),
                layers.Dense(3, activation='linear')   
                ])
        self.model.summary()
        #%
        self.model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
        
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=5, verbose=1
        )   
        self.model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                  epochs=1, batch_size=100,callbacks=[lr_scheduler],
                  shuffle=True)
    
    
    #% Test Model
    
    def predict_params(self):
        y = self.make_signal(self.t_int,1,0.05,0)
        y = np.array(y, dtype=np.float32).reshape(1, -1)
        A, f, phi = self.model.predict(y)[0]
    
        y_fit = self.make_signal(self.t_int, A, f, phi, noise=0)
    
        plt.figure()
        plt.plot(self.t_int, y.flatten(), 'o', ms=3, label='data')
        plt.plot(self.t_int, y_fit, lw=2, label='fit')
        plt.legend()
        plt.show()
        
        return A,f,phi


# =============================================================================
# Execute:
# =============================================================================

FA = FrequencyAnalysis()

#%% =============================================================================
# Test :
# =============================================================================
from scipy.signal import savgol_filter
t = FA.t
t_int = np.linspace(min(t),max(t),500)
y = FA.exp_data[1000,:]
y_int = np.interp(t_int,t,y)

# y = y_int
# t = t_int

fromx = 0
tox = None

t = t[fromx:tox]
y = y[fromx:tox]
# 

est_T = 18
N_est_T = int(np.ceil(est_T/(1*min(np.diff(t_int)))))



window = N_est_T
fig,axs = plt.subplots(2,1)
axs[0].plot(t,y,'x-')

axs[0].plot(t_int,savgol_filter(y_int,window,3))
# axs[1].plot(t_int,y_int-savgol_filter(y_int,window,3))

y_red = y_int-savgol_filter(y_int,window,3)

Fy = fft.fft(y_red)
f = fft.fftfreq(len(t_int), np.mean(np.diff(t_int)))[:len(t_int)//2]
Fy = 2/len(t_int)*np.abs(Fy)[:len(t_int)//2]

axs[1].plot(f,Fy)

#%% Diff Test

t = FA.t
t_int = np.linspace(min(t),max(t),5000)
y = FA.exp_data[1000,:]
y_int = np.interp(t_int,t,y)

y_diff = np.diff(y_int)   

fig,ax=plt.subplots(2,1)
ax[0].plot(t,y)
ax[0].plot(t_int,y_int)
ax[1].plot(t_int[:-1],y_diff)

condition = (y_diff[:-1] > 0) & (y_diff[1:] < 0)

# Get indices where the condition is True
indices = np.where(condition)[0]

diff_pos = np.diff(t_int[indices])

ax[1].plot(t_int[indices],np.array([0.01]*len(indices)),'x')

condition = (y_diff[:-1] < 0) & (y_diff[1:] > 0)

# Get indices where the condition is True
indices = np.where(condition)[0]

diff_neg = np.diff(t_int[indices])

ax[1].plot(t_int[indices],np.array([-0.01]*len(indices)),'x')

ax[1].set_ylim([-0.02,0.02])
ax[1].axhline(color='k')









