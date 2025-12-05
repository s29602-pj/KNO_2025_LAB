import os
import argparse
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Normalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

import keras_tuner as kt  # Keras Tuner
from sklearn.metrics import confusion_matrix  # NOWE: macierz pomyłek

DATA_PATH = "wine.data"
BEST_MODEL_PATH = "wine_best_model_norm.keras"  # najlepszy model z normalizacją

column_names = [
    "class",
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315",
    "proline",
]

# Zmienne globalne dla tunera (żeby build_model_hp miał dostęp do wymiarów i danych)
X_TRAIN_FOR_NORM_TUNER = None
INPUT_DIM_TUNER = None
NUM_CLASSES_TUNER = None


def load_and_prepare_data(test_ratio: float = 0.2, seed: int = 42):

    df = pd.read_csv(DATA_PATH, header=None, names=column_names)

    print("Rozkład klas:")
    print(df["class"].value_counts(), "\n")

    X = df.drop(columns=["class"]).to_numpy(dtype=np.float32)
    y = df["class"].to_numpy(dtype=np.int64)

    rng = np.random.default_rng(seed=seed)
    indices = rng.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]

    print("Kształt X:", X.shape, " Kształt y:", y.shape)

    num_classes = len(np.unique(y))
    y_onehot = tf.keras.utils.to_categorical(y - 1, num_classes=num_classes)

    n_samples = X.shape[0]
    n_test = int(test_ratio * n_samples)

    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y_onehot[:-n_test], y_onehot[-n_test:]

    print("X_train:", X_train.shape, " X_test:", X_test.shape)
    print("y_train:", y_train.shape, " y_test:", y_test.shape)

    return X_train, X_test, y_train, y_test


def build_model_1(
    input_dim: int, num_classes: int, lr: float, X_train: np.ndarray
) -> tf.keras.Model:
    """
    Model 1 z warstwą Normalization na wejściu.
    """
    norm = Normalization(axis=-1, input_shape=(input_dim,), name="input_norm_1")
    norm.adapt(X_train)

    model = Sequential(name="wine_model_1_norm")
    model.add(norm)
    model.add(Dense(32, activation="relu", name="hidden_1"))
    model.add(Dense(16, activation="relu", name="hidden_2"))
    model.add(Dense(num_classes, activation="softmax", name="output"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model_2(
    input_dim: int, num_classes: int, lr: float, X_train: np.ndarray
) -> tf.keras.Model:
    """
    Model 2 (głębszy) z warstwą Normalization na wejściu.
    """
    norm = Normalization(axis=-1, input_shape=(input_dim,), name="input_norm_2")
    norm.adapt(X_train)

    model = Sequential(name="wine_model_2_norm")
    model.add(norm)
    model.add(
        Dense(
            64,
            activation="selu",
            kernel_initializer="he_normal",
            name="hidden_1_selu",
        )
    )
    model.add(Dropout(0.2, name="dropout_1"))
    model.add(
        Dense(
            32,
            activation="relu",
            kernel_initializer="glorot_uniform",
            name="hidden_2_relu",
        )
    )
    model.add(
        Dense(
            16,
            activation="tanh",
            kernel_initializer="glorot_uniform",
            name="hidden_3_tanh",
        )
    )
    model.add(Dense(num_classes, activation="softmax", name="output"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ============================
#  Funkcja do macierzy pomyłek
# ============================


def print_and_save_confusion_matrix(model, X, y_onehot, filename: str):
    """
    Liczy macierz pomyłek dla danych testowych (y w postaci one-hot),
    wypisuje ją na konsolę i zapisuje do pliku CSV.
    """

    y_true = np.argmax(y_onehot, axis=1)

    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)

    cm = confusion_matrix(y_true, y_pred)
    print(f"\nMacierz pomyłek ({filename}):")
    print(cm)

    np.savetxt(filename, cm, fmt="%d", delimiter=",")
    return cm


# ============================
#  Hiperparametry – Keras Tuner
# ============================


def build_model_hp(hp: kt.HyperParameters) -> tf.keras.Model:
    """
    Funkcja budująca model dla Keras Tunera.

    Stroimy 3 hiperparametry:
      - learning_rate (float, logarytmicznie)
      - units_1      (liczba neuronów w 1. warstwie ukrytej)
      - dropout_rate (intensywność Dropout)
    """
    if (
        INPUT_DIM_TUNER is None
        or NUM_CLASSES_TUNER is None
        or X_TRAIN_FOR_NORM_TUNER is None
    ):
        raise RuntimeError(
            "Globalne parametry tunera nie zostały ustawione. "
            "Upewnij się, że build_model_hp jest używane tylko z run_tuner()."
        )

    lr = hp.Float(
        "learning_rate",
        min_value=1e-4,
        max_value=1e-2,
        sampling="log",
    )

    units_1 = hp.Int(
        "units_1",
        min_value=16,
        max_value=128,
        step=16,
    )

    dropout_rate = hp.Float(
        "dropout_rate",
        min_value=0.0,
        max_value=0.5,
        step=0.1,
    )

    norm = Normalization(
        axis=-1, input_shape=(INPUT_DIM_TUNER,), name="input_norm_tuner"
    )
    norm.adapt(X_TRAIN_FOR_NORM_TUNER)

    model = Sequential(name="wine_model_tuned")
    model.add(norm)
    model.add(Dense(units_1, activation="relu", name="hidden_1"))
    model.add(Dropout(dropout_rate, name="dropout_1"))
    model.add(Dense(NUM_CLASSES_TUNER, activation="softmax", name="output"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_tuner(max_trials: int = 20, executions_per_trial: int = 1):
    """
    Uruchamia Keras Tuner (RandomSearch) dla modelu z normalizacją.
    Zapisuje najlepszy model do wine_best_model_tuned.keras
    oraz wyniki do tuner_results.json i confusion_matrix_tuned.csv.
    """
    gpus = tf.config.list_physical_devices("GPU")
    print("Dostępne GPU:", gpus)

    X_train, X_test, y_train, y_test = load_and_prepare_data()

    global X_TRAIN_FOR_NORM_TUNER, INPUT_DIM_TUNER, NUM_CLASSES_TUNER
    X_TRAIN_FOR_NORM_TUNER = X_train
    INPUT_DIM_TUNER = X_train.shape[1]
    NUM_CLASSES_TUNER = y_train.shape[1]

    tuner = kt.RandomSearch(
        build_model_hp,
        objective="val_accuracy",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory="kt_logs",
        project_name="wine_norm_randomsearch",
        overwrite=True,
    )

    print(
        f"\n=== Start tuningu (RandomSearch) ===\n"
        f"max_trials={max_trials}, executions_per_trial={executions_per_trial}\n"
    )

    tuner.search(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=16,
        verbose=1,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]

    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
    print(
        f"\n=== Najlepszy model z tunera ===\n"
        f"learning_rate={best_hp.get('learning_rate'):.6f}, "
        f"units_1={best_hp.get('units_1')}, "
        f"dropout_rate={best_hp.get('dropout_rate'):.2f}"
    )
    print(f"Tuner – test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

    tuned_model_path = "wine_best_model_tuned.keras"
    best_model.save(tuned_model_path)
    print(f"Najlepszy model z tunera zapisano do pliku: {tuned_model_path}")

    tuner_results = {
        "best_hyperparameters": {
            "learning_rate": float(best_hp.get("learning_rate")),
            "units_1": int(best_hp.get("units_1")),
            "dropout_rate": float(best_hp.get("dropout_rate")),
        },
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
    }
    with open("tuner_results.json", "w") as f:
        json.dump(tuner_results, f, indent=4)
    print("Wyniki tuningu zapisano do tuner_results.json")

    print_and_save_confusion_matrix(
        best_model,
        X_test,
        y_test,
        "confusion_matrix_tuned.csv",
    )


# ============================
#  Trening dwóch modeli (norm)
# ============================


def train_and_save_best():
    """Trening dwóch modeli z normalizacją + zapis najlepszych wyników do plików."""
    gpus = tf.config.list_physical_devices("GPU")
    print("Dostępne GPU:", gpus)

    X_train, X_test, y_train, y_test = load_and_prepare_data()
    input_dim = X_train.shape[1]
    num_classes = y_train.shape[1]

    EPOCHS_1, BATCH_1, LR_1 = 50, 16, 0.001
    EPOCHS_2, BATCH_2, LR_2 = 70, 8, 0.0005

    print(
        f"\n=== Model 1 (norm) === epochs={EPOCHS_1}, batch_size={BATCH_1}, lr={LR_1}"
    )
    model1 = build_model_1(input_dim, num_classes, LR_1, X_train)
    model1.summary()

    log_dir_1 = os.path.join("logs_norm", "model1_norm")
    tb1 = TensorBoard(log_dir=log_dir_1, histogram_freq=1)

    history1 = model1.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS_1,
        batch_size=BATCH_1,
        callbacks=[tb1],
        verbose=1,
    )

    test_loss_1, test_acc_1 = model1.evaluate(X_test, y_test, verbose=0)
    print(f"Model 1 (norm) – test_loss={test_loss_1:.4f}, test_acc={test_acc_1:.4f}")

    # Zapis wyników Modelu 1 (norm)
    val_acc_1 = history1.history["val_accuracy"]
    best_val_1 = float(np.max(val_acc_1))
    last_val_1 = float(val_acc_1[-1])

    norm_results_1 = {
        "model": "Model 1 (z normalizacją)",
        "epochs": EPOCHS_1,
        "batch_size": BATCH_1,
        "learning_rate": LR_1,
        "val_acc_last": last_val_1,
        "val_acc_best": best_val_1,
        "test_acc": float(test_acc_1),
    }

    with open("norm_results_model1.json", "w") as f:
        json.dump(norm_results_1, f, indent=4)

    print(
        f"Wyniki Modelu 1 (norm) zapisane do norm_results_model1.json: "
        f"best val_acc={best_val_1:.4f}, test_acc={test_acc_1:.4f}"
    )

    print(
        f"\n=== Model 2 (norm) === epochs={EPOCHS_2}, batch_size={BATCH_2}, lr={LR_2}"
    )
    model2 = build_model_2(input_dim, num_classes, LR_2, X_train)
    model2.summary()

    log_dir_2 = os.path.join("logs_norm", "model2_norm")
    tb2 = TensorBoard(log_dir=log_dir_2, histogram_freq=1)

    history2 = model2.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS_2,
        batch_size=BATCH_2,
        callbacks=[tb2],
        verbose=1,
    )

    test_loss_2, test_acc_2 = model2.evaluate(X_test, y_test, verbose=0)
    print(f"Model 2 (norm) – test_loss={test_loss_2:.4f}, test_acc={test_acc_2:.4f}")

    val_acc_2 = history2.history["val_accuracy"]
    best_val_2 = float(np.max(val_acc_2))
    last_val_2 = float(val_acc_2[-1])

    norm_results_2 = {
        "model": "Model 2 (z normalizacją)",
        "epochs": EPOCHS_2,
        "batch_size": BATCH_2,
        "learning_rate": LR_2,
        "val_acc_last": last_val_2,
        "val_acc_best": best_val_2,
        "test_acc": float(test_acc_2),
    }

    with open("norm_results_model2.json", "w") as f:
        json.dump(norm_results_2, f, indent=4)

    print(
        f"Wyniki Modelu 2 (norm) zapisane do norm_results_model2.json: "
        f"best val_acc={best_val_2:.4f}, test_acc={test_acc_2:.4f}"
    )

    if test_acc_2 > test_acc_1:
        best_model = model2
        best_name = "Model 2 (norm)"
        best_acc = test_acc_2
    else:
        best_model = model1
        best_name = "Model 1 (norm)"
        best_acc = test_acc_1

    best_model.save(BEST_MODEL_PATH)
    print(
        f"\n>>> Najlepszy (z normalizacją) okazał się {best_name} (test_acc={best_acc:.4f})"
    )
    print(f"Model zapisano do pliku: {BEST_MODEL_PATH}")

    print_and_save_confusion_matrix(
        best_model,
        X_test,
        y_test,
        "confusion_matrix_norm_best.csv",
    )


def predict_from_features(features):
    """Predykcja klasy wina przy użyciu najlepszego modelu z normalizacją."""
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Nie znaleziono {BEST_MODEL_PATH}. "
            f"Najpierw uruchom trening: python Wina2_norm.py train"
        )

    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    x = np.array(features, dtype=np.float32).reshape(1, -1)

    probs = model.predict(x, verbose=0)[0]
    predicted_class = int(np.argmax(probs) + 1)

    print("Prawdopodobieństwa klas:", probs)
    print("Przewidywana klasa wina:", predicted_class)


def main():
    parser = argparse.ArgumentParser(
        description="Klasyfikacja win – modele z normalizacją (trening, tuning i predykcja)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "train", help="Wytrenuj dwa modele z normalizacją i zapisz najlepszy do pliku"
    )

    tune_parser = subparsers.add_parser(
        "tune",
        help="Uruchom Keras Tuner (RandomSearch) dla modelu z normalizacją",
    )
    tune_parser.add_argument(
        "--max_trials",
        type=int,
        default=20,
        help="Liczba prób (różnych konfiguracji hiperparametrów) w tunerze",
    )
    tune_parser.add_argument(
        "--executions_per_trial",
        type=int,
        default=1,
        help="Ile razy trenować ten sam zestaw hiperparametrów (uśrednianie)",
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Przewiduj klasę wina na podstawie 13 parametrów",
    )
    predict_parser.add_argument(
        "features",
        nargs=13,
        type=float,
        metavar=(
            "ALCOHOL",
            "MALIC_ACID",
            "ASH",
            "ALCALINITY_ASH",
            "MAGNESIUM",
            "TOTAL_PHENOLS",
            "FLAVANOIDS",
            "NONFLAVANOID_PHENOLS",
            "PROANTHOCYANINS",
            "COLOR_INTENSITY",
            "HUE",
            "OD280_OD315",
            "PROLINE",
        ),
        help="13 cech wina w kolejności jak w zbiorze Wine",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_and_save_best()
    elif args.command == "predict":
        predict_from_features(args.features)
    elif args.command == "tune":
        run_tuner(
            max_trials=args.max_trials, executions_per_trial=args.executions_per_trial
        )


if __name__ == "__main__":
    main()
