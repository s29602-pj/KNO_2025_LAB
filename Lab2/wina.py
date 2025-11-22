import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard


DATA_PATH = "wine.data"
BEST_MODEL_PATH = "wine_best_model.keras"


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


def build_model_1(input_dim: int, num_classes: int, lr: float) -> tf.keras.Model:

    model = Sequential(name="wine_model_1_simple")
    model.add(Dense(32, activation="relu", input_shape=(input_dim,), name="hidden_1"))
    model.add(Dense(16, activation="relu", name="hidden_2"))
    model.add(Dense(num_classes, activation="softmax", name="output"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model_2(input_dim: int, num_classes: int, lr: float) -> tf.keras.Model:

    model = Sequential(name="wine_model_2_deeper")
    model.add(
        Dense(
            64,
            activation="selu",
            input_shape=(input_dim,),
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


def train_and_save_best():
    gpus = tf.config.list_physical_devices("GPU")
    print("Dostępne GPU:", gpus)

    X_train, X_test, y_train, y_test = load_and_prepare_data()
    input_dim = X_train.shape[1]
    num_classes = y_train.shape[1]


    EPOCHS_1, BATCH_1, LR_1 = 50, 16, 0.001
    EPOCHS_2, BATCH_2, LR_2 = 70, 8, 0.0005


    print(f"\n=== Model 1 === epochs={EPOCHS_1}, batch_size={BATCH_1}, lr={LR_1}")
    model1 = build_model_1(input_dim, num_classes, LR_1)
    model1.summary()

    log_dir_1 = os.path.join("logs", "model1")
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
    print(f"Model 1 – test_loss={test_loss_1:.4f}, test_acc={test_acc_1:.4f}")


    print(f"\n=== Model 2 === epochs={EPOCHS_2}, batch_size={BATCH_2}, lr={LR_2}")
    model2 = build_model_2(input_dim, num_classes, LR_2)
    model2.summary()

    log_dir_2 = os.path.join("logs", "model2")
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
    print(f"Model 2 – test_loss={test_loss_2:.4f}, test_acc={test_acc_2:.4f}")


    if test_acc_2 > test_acc_1:
        best_model = model2
        best_name = "Model 2"
        best_acc = test_acc_2
    else:
        best_model = model1
        best_name = "Model 1"
        best_acc = test_acc_1

    best_model.save(BEST_MODEL_PATH)
    print(f"\n>>> Najlepszy okazał się {best_name} (test_acc={best_acc:.4f})")
    print(f"Model zapisano do pliku: {BEST_MODEL_PATH}")


def predict_from_features(features):
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Nie znaleziono {BEST_MODEL_PATH}. "
            f"Najpierw uruchom trening: python wina.py train"
        )

    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    x = np.array(features, dtype=np.float32).reshape(1, -1)

    probs = model.predict(x, verbose=0)[0]
    predicted_class = int(np.argmax(probs) + 1)

    print("Prawdopodobieństwa klas:", probs)
    print("Przewidywana klasa wina:", predicted_class)


def main():
    parser = argparse.ArgumentParser(
        description="Klasyfikacja win – trening dwóch modeli i predykcja."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)


    subparsers.add_parser(
        "train", help="Wytrenuj dwa modele i zapisz najlepszy do pliku"
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


if __name__ == "__main__":
    main()
