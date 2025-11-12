import numpy as np
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
models = keras.models
optimizers = keras.optimizers
callbacks = keras.callbacks
from matplotlib import pyplot as plt

# ---- generate dataset with varying t grids ----
def make_signal(t, A, f, phi, noise_std=0.1):
    y = A * np.sin(2*np.pi*f*t + phi)
    y += np.random.normal(scale=noise_std, size=t.shape)
    return y

def sample_dataset_variable_t(n_signals=20000):
    X = []
    Y = []
    for _ in range(n_signals):
        # random time sampling
        seq_len = np.random.randint(128, 256)
        fs = np.random.uniform(100, 300)  # random sampling freq
        t = np.sort(np.random.uniform(0, seq_len/fs, seq_len))
        A = np.random.uniform(0.2, 2.0)
        f = np.random.uniform(0.5, 10.0)
        phi = np.random.uniform(0, 2*np.pi)
        y = make_signal(t, A, f, phi, noise_std=0.1)
        # concatenate t and y as 2D input [N,2]
        xy = np.stack([t, y], axis=1)
        X.append(xy)
        Y.append([A, f, np.sin(phi), np.cos(phi)])
    # pad sequences to same length for batching
    X = keras.preprocessing.sequence.pad_sequences(X, padding='post', dtype='float32')
    Y = np.array(Y, dtype=np.float32)
    return X, Y

def make_model_variable_t(seq_len):
    inp = layers.Input((seq_len, 2))
    x = layers.Conv1D(64, 5, activation='relu', padding='same')(inp)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(4, activation='linear')(x)  # A, f, sinφ, cosφ
    return keras.Model(inp, out)

def predict_params(model, t, y):
    xy = np.stack([t, y], axis=1)
    xy = keras.preprocessing.sequence.pad_sequences([xy], padding='post', dtype='float32')
    pred = model.predict(xy)[0]
    A, f, sinp, cosp = pred
    phi = np.arctan2(sinp, cosp)
        
    fig,ax=plt.subplots()
    ax.plot(t,y)
    ax.plot(t,A*np.sin(2*np.pi*f*t+phi))
    
    return A, f, phi

X, Y = sample_dataset_variable_t()
n_train = int(0.8*len(X))
X_train, X_val = X[:n_train], X[n_train:]
Y_train, Y_val = Y[:n_train], Y[n_train:]

model = make_model_variable_t(seq_len=X.shape[1])
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
          epochs=4, batch_size=128)


#%%

t=np.linspace(0,10,1000)

A_true = 2
f_true = 0.1
phi_true = np.pi

y = A_true*np.sin(2*np.pi*f_true*t-phi_true)

A, f, phi = predict_params(model, t, y)
























#%%
# Keras example: predict A, f, phi from noisy sine waves
import numpy as np
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
models = keras.models
optimizers = keras.optimizers
callbacks = keras.callbacks
from matplotlib import pyplot as plt

# --- Data generation ---
def make_signal(t, A, f, phi, noise_std=0.1, drift=0.0, damp=0.0):
    s = A * np.sin(2*np.pi*f*t + phi) * np.exp(-damp*t) + drift
    s += np.random.normal(scale=noise_std, size=t.shape)
    return s

def sample_dataset(n_signals=20000, seq_len=256, fs=100.0,
                   A_range=(0.2, 2.0), f_range=(0.5, 10.0), phi_range=(0, 2*np.pi),
                   noise_std_range=(0.01, 0.5), damp_range=(0.0, 0.5), drift_range=(-0.2,0.2)):
    t = np.arange(seq_len) / fs
    X = np.zeros((n_signals, seq_len), dtype=np.float32)
    Y = np.zeros((n_signals, 3), dtype=np.float32)  # A, f, sin(phi), cos(phi)
    for i in range(n_signals):
        A = np.random.uniform(*A_range)
        f = np.random.uniform(*f_range)
        phi = np.random.uniform(*phi_range)
        noise = np.random.uniform(*noise_std_range)
        damp = np.random.uniform(*damp_range)
        drift = np.random.uniform(*drift_range)
        X[i] = make_signal(t, A, f, phi, noise_std=noise, drift=drift, damp=damp)
        Y[i,0] = A
        Y[i,1] = f
        Y[i,2] = np.sin(phi)
        Y[i,3-1] = np.cos(phi)  # careful indexing if you want 4 cols; here we used 3, so adjust
    return X, Y, t

# Fix small indexing bug: use 4 outputs (A, f, sinφ, cosφ)
def sample_dataset_fixed(n_signals=20000, seq_len=256, fs=100.0,
                   A_range=(0.2, 2.0), f_range=(0.5, 10.0), phi_range=(0, 2*np.pi),
                   noise_std_range=(0.01, 0.5), damp_range=(0.0, 0.5), drift_range=(-0.2,0.2)):
    t = np.arange(seq_len) / fs
    X = np.zeros((n_signals, seq_len), dtype=np.float32)
    Y = np.zeros((n_signals, 4), dtype=np.float32)  # A, f, sin(phi), cos(phi)
    for i in range(n_signals):
        A = np.random.uniform(*A_range)
        f = np.random.uniform(*f_range)
        phi = np.random.uniform(*phi_range)
        noise = np.random.uniform(*noise_std_range)
        damp = np.random.uniform(*damp_range)
        drift = np.random.uniform(*drift_range)
        X[i] = make_signal(t, A, f, phi, noise_std=noise, drift=drift, damp=damp)
        Y[i,0] = A
        Y[i,1] = f
        Y[i,2] = np.sin(phi)
        Y[i,3] = np.cos(phi)
    return X, Y, t

# Create data
sample_rate=200.0
X, Y, t = sample_dataset_fixed(n_signals=20000, seq_len=256, fs=sample_rate)

# Normalize inputs (per-signal mean removal and global std)
X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
X = X[..., np.newaxis]  # shape (N, seq_len, 1)

# Train/test split
n_train = int(0.8*X.shape[0])
X_train, X_val = X[:n_train], X[n_train:]
Y_train, Y_val = Y[:n_train], Y[n_train:]

# --- Model (1D CNN) ---
def make_model(seq_len):
    inp = layers.Input((seq_len,1))
    x = layers.Conv1D(32, 7, activation='relu', padding='same')(inp)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(4, activation='linear')(x)  # A, f, sinφ, cosφ
    model = models.Model(inp, out)
    return model

model = make_model(seq_len=256)
model.compile(optimizer=optimizers.Adam(1e-3), loss='mse')
model.summary()

# --- Training ---
es = callbacks.EarlyStopping(patience=8, restore_best_weights=True)
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                    epochs=60, batch_size=128, callbacks=[es])

# --- Usage: predict and reconstruct phi ---
def predict_params(model, raw_signal, fs=200.0):
    # raw_signal: 1D array length seq_len
    x = (raw_signal - raw_signal.mean()) / (raw_signal.std() + 1e-8)
    x = x[np.newaxis,:,np.newaxis]
    pred = model.predict(x)[0]
    A_pred = pred[0]
    f_pred = pred[1]
    sinp, cosp = pred[2], pred[3]
    phi_pred = np.arctan2(sinp, cosp)
    
    t=np.arange(len(raw_signal))/sample_rate
    
    fig,ax=plt.subplots()
    ax.plot(t,raw_signal)
    ax.plot(t,A_pred*np.sin(2*np.pi*f_pred*t+phi_pred))
    
    return A_pred, f_pred, phi_pred

#%% Example test
i = n_train + 5
A_p, f_p, phi_p = predict_params(model, X_val[i-n_train,:,0])


#%%

t=np.linspace(0,100,200)
A_true=2
f_true=0.1
phi_true=2
y=A_true*np.sin(2*np.pi*f_true*t-phi_true)

A_p, f_p, phi_p = predict_params(model, y)
# print("Predicted:", A_p, f_p, phi_p)
# print("True     :", Y_val[i-n_train,0], Y_val[i-n_train,1], np.arctan2(Y_val[i-n_train,2], Y_val[i-n_train,3]))