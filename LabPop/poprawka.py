import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

DT = 0.01
T_MAX = 3 * np.pi
WINDOW = 80
HORIZON = 1
TAIL = 600
FUTURE_STEPS = 800
BATCH_SIZE = 64
EPOCHS = 10

t = np.arange(0, T_MAX, DT, dtype=np.float32)
y = np.sin(t).astype(np.float32)

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

    X = np.array(X, dtype=np.float32)[..., None]   # (samples, window, 1)
    Y = np.array(Y, dtype=np.float32)[..., None]   # (samples, 1)
    T_y = np.array(T_y, dtype=np.float32)
    return X, Y, T_y

X, Y, T_y = make_supervised(t, y, WINDOW, HORIZON)
print("X:", X.shape, "Y:", Y.shape)

model = keras.Sequential(
    [
        keras.Input(shape=(WINDOW, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ]
)

model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

model.fit(
    X,
    Y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
)

def forecast_future(model, last_window, steps):
    w = last_window.reshape(1, WINDOW, 1).astype(np.float32)
    preds = []

    for _ in range(steps):
        y_next = model.predict(w, verbose=0)[0, 0]
        preds.append(y_next)
        w = np.concatenate([w[:, 1:, :], np.array([[[y_next]]], dtype=np.float32)], axis=1)

    return np.array(preds, dtype=np.float32)

last_window = y[-WINDOW:]
y_future = forecast_future(model, last_window, FUTURE_STEPS)
t_future = t[-1] + DT * np.arange(1, FUTURE_STEPS + 1, dtype=np.float32)

# opcjonalnie do porównania
y_true_future = np.sin(t_future).astype(np.float32)

plt.figure(figsize=(12, 5))
plt.plot(t[-TAIL:], y[-TAIL:], label="prawdziwe y (koniec danych)")
plt.plot(t_future, y_future, label="kontynuacja (model)", linewidth=2)
plt.plot(t_future, y_true_future, linestyle="--", label="prawdziwe y (poza zakresem)")
plt.axvline(t[-1], linestyle="--", label="koniec danych")
plt.title("Kontynuacja wykresu funkcji: predykcja poza zakresem danych")
plt.xlabel("t")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()