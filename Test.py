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



#%%

# ---- generate dataset with varying t grids ----
def make_signal(t, A, f, phi, 
                A_rise1=1,tau_rise1=0.05,A_rise2=0.5,tau_rise2=20,
                A_decay1=0.3,tau_decay1=80,A_decay2=0.1,tau_decay2=100000,
                noise=0.1):
    y = A * np.sin(2*np.pi*f*t + phi)
    y += A_rise1*(1-np.exp(-t/tau_rise1))
    y += A_rise2*(1-np.exp(-t/tau_rise2))
    y += A_decay1*np.exp(-t/tau_decay1)
    y += A_decay2*np.exp(-t/tau_decay2)
    
    y+= -A_decay1-A_decay2 
    
    y += np.random.normal(scale=noise, size=t.shape)
    return y*np.heaviside(t,0)

def sample_dataset(t,n_signals=20000):
    X = []
    Y = []
    for _ in range(n_signals):
        
        # Define Ranges:
        A = np.random.uniform(0.01, 0.5)
        f = np.random.uniform(0.02, 0.4)
        phi = np.random.uniform(0, 2*np.pi)
        A_rise1 = np.random.uniform(0, 2)
        tau_rise1 = np.random.uniform(0.1,1)
        A_rise2 = np.random.uniform(0, 2)
        tau_rise2 = np.random.uniform(2,20)
        A_decay1 = np.random.uniform(0, 2)
        tau_decay1 = np.random.uniform(40,150)
        A_decay2 = np.random.uniform(0, 2)
        tau_decay2 = np.random.uniform(500,2000)
        
        
        y = make_signal(t, A, f, phi, noise=0.1,
                        A_rise1=A_rise1, tau_rise1=tau_rise1, A_rise2=A_rise2, tau_rise2=tau_rise2, 
                        A_decay1=A_decay1, tau_decay1=tau_decay1, A_decay2=A_decay2, tau_decay2=tau_decay2)
        # concatenate t and y as 2D input [N,2]
        xy = np.stack([t, y], axis=1)
        X.append(xy)
        Y.append([A, f, phi, A_rise1, tau_rise1, A_rise2, tau_rise2,
                  A_decay1, tau_decay1,A_decay2, tau_decay2])
    # pad sequences to same length for batching
    X = keras.preprocessing.sequence.pad_sequences(X, padding='post', dtype='float32')
    Y = np.array(Y, dtype=np.float32)
    return X, Y

def make_model():
    inp = layers.Input((None, 2))
    # x = layers.Dense(64, activation='relu')(inp)
    # x = layers.Conv1D(64, 5, activation='relu', padding='same')(inp)
    # x = layers.MaxPool1D(2)(x)
    # x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(inp)
    # x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(11, activation='linear')(x)  # A, f, sin(phi), cos(phi), A_exp, tau_rise, tau_decay 
    return keras.Model(inp, out)

def predict_params(model, t, y):
    xy = np.stack([t, y], axis=1)
    xy = keras.preprocessing.sequence.pad_sequences([xy], padding='post', dtype='float32')
    pred = model.predict(xy)[0]
    A, f, phi, A_rise1, tau_rise1, A_rise2, tau_rise2, A_decay1, tau_decay1,A_decay2, tau_decay2 = pred
        
    fig,ax=plt.subplots()
    ax.plot(t,y,'o')
    ax.plot(t,make_signal(t, A, f, phi, noise=0,
                    A_rise1=A_rise1, tau_rise1=tau_rise1, A_rise2=A_rise2, tau_rise2=tau_rise2, 
                    A_decay1=A_decay1, tau_decay1=tau_decay1, A_decay2=A_decay2, tau_decay2=tau_decay2))
    
    return A, f, phi



# fig,ax=plt.subplots()
# ax.plot(t,exp_data[1000,:])
# ax.plot(t,make_signal(t,0.1,0.091,0.9,
#                       A_rise1=1,tau_rise1=0.05,A_rise2=0.5,tau_rise2=20,
#                       A_decay1=0.3,tau_decay1=80,A_decay2=0.1,tau_decay2=100000,
#                       noise=0
#                       ))


#% Train Model

X, Y = sample_dataset(t)
n_train = int(0.8*len(X))
X_train, X_val = X[:n_train], X[n_train:]
Y_train, Y_val = Y[:n_train], Y[n_train:]

model = make_model()
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
          epochs=100, batch_size=128)


#%% Test Model

t_int=np.linspace(min(t),max(t),1500)

A,f, phi = predict_params(model, t_int, np.interp(t_int,t,exp_data[100,:]))
























