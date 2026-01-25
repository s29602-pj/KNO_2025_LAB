import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
DT = 0.05
T_MAX = 60 * np.pi
WINDOW = 80  #
HORIZON_N = 10
BATCH_SIZE = 64
EPOCHS = 40
NOISE_STD = 0.0


t = np.arange(0, T_MAX, DT, dtype=np.float32)

y = np.sin(t).astype(np.float32)
if NOISE_STD > 0:
    y = (y + NOISE_STD * np.random.randn(len(y))).astype(np.float32)


plt.figure()
plt.plot(t, y)
plt.title("Funkcja: y = sin(t)")
plt.xlabel("t")
plt.ylabel("y")
plt.grid(True)
plt.show()


def make_supervised(t, y, window, horizon):
    X, Y, T_y = [], [], []
    L = len(y)
    last_start = L - window - horizon + 1
    for i in range(last_start):
        X.append(y[i : i + window])
        target_idx = i + window + horizon - 1
        Y.append(y[target_idx])
        T_y.append(t[target_idx])

    X = np.array(X, dtype=np.float32)[..., None]
    Y = np.array(Y, dtype=np.float32)[..., None]
    T_y = np.array(T_y, dtype=np.float32)
    return X, Y, T_y


X, Y, T_y = make_supervised(t, y, WINDOW, HORIZON_N)
n = len(X)


train_end = int(0.70 * n)
val_end = int(0.85 * n)

X_train, Y_train = X[:train_end], Y[:train_end]
X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
X_test, Y_test = X[val_end:], Y[val_end:]
T_test = T_y[val_end:]

print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
print("X_val:  ", X_val.shape, "Y_val:  ", Y_val.shape)
print("X_test: ", X_test.shape, "Y_test: ", Y_test.shape)


model = keras.Sequential(
    [
        keras.Input(shape=(WINDOW, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="mse",
    metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
)

# model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    )
]

history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)


plt.figure()
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Zbieżność uczenia (MSE loss)")
plt.xlabel("Epoka")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()


# PREDYKCJA I WYKRES

Y_pred_test = model.predict(X_test, batch_size=BATCH_SIZE).reshape(-1)

plt.figure(figsize=(12, 5))
plt.plot(t, y, alpha=0.35, label="prawdziwe y = sin(t)")
plt.plot(T_test, Y_pred_test, label=f"predykcja y(t + {HORIZON_N} kroków)", linewidth=2)
plt.plot(
    T_test,
    Y_test.reshape(-1),
    label="prawdziwe y w chwilach targetów",
    linewidth=2,
    alpha=0.8,
)
plt.axvline(T_y[val_end], linestyle="--", label="start test (w czasach targetów)")
plt.title("Kontynuacja wykresu funkcji: prawda vs predykcja (N kroków do przodu)")
plt.xlabel("t")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()


# ZOOM

zoom_from = int(0.90 * len(t))
plt.figure(figsize=(12, 5))
plt.plot(t[zoom_from:], y[zoom_from:], label="prawdziwe y = sin(t)", linewidth=2)
plt.plot(T_test, Y_pred_test, label="predykcja (test)", linewidth=2)
plt.title("Zoom: końcowy fragment (kontynuacja)")
plt.xlabel("t")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()
