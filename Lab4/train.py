from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from common import load_fashion_mnist

NUM_CLASSES = 10


def build_dense_model(input_shape) -> keras.Model:
    """Wersja oparta o warstwy w pełni połączone."""
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_model(input_shape, use_augmentation: bool = False) -> keras.Model:
    """Wersja oparta o warstwy splotowe, opcjonalna augmentacja."""
    inputs = keras.Input(shape=input_shape)

    x = inputs
    if use_augmentation:
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),
                layers.RandomZoom(0.1),
            ]
        )
        x = data_augmentation(x)

    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_learning_curves(history: keras.callbacks.History, out_path: Path) -> None:
    """Rysowanie krzywej uczenia (loss + val_loss)."""
    plt.figure()
    plt.plot(history.history["loss"], label="loss (train)")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="loss (val)")
    plt.xlabel("Epoka")
    plt.ylabel("Loss")
    plt.title("Krzywa uczenia – loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_confusion_matrix(model: keras.Model, x_test, y_test):
    """Macierz pomyłek na zbiorze testowym."""
    probas = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probas, axis=1)
    cm = tf.math.confusion_matrix(
        y_test, y_pred, num_classes=NUM_CLASSES
    ).numpy()
    return cm.tolist()  # do zapisu w JSON


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trenowanie modelu Fashion-MNIST."
    )
    parser.add_argument(
        "--arch",
        choices=["dense", "cnn"],
        default="dense",
        help="Wybór architektury: gęsta lub splotowa.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Katalog wyjściowy na modele i metryki.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Użyj augmentacji danych (tylko dla CNN).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for_cnn = args.arch == "cnn"
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist(for_cnn=for_cnn)

    if args.arch == "dense":
        model = build_dense_model(input_shape=x_train.shape[1:])
        model_name = "fashion_dense"
    else:
        model = build_cnn_model(
            input_shape=x_train.shape[1:], use_augmentation=args.augment
        )
        model_name = "fashion_cnn_aug" if args.augment else "fashion_cnn"

    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}  |  test acc: {test_acc:.4f}")

    cm = compute_confusion_matrix(model, x_test, y_test)

    # zapis modelu
    model_path = out_dir / f"{model_name}.keras"
    model.save(model_path)
    print(f"Model zapisany do: {model_path}")

    # zapis metryk + historii + macierzy pomyłek
    metrics = {
        "arch": args.arch,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "history": history.history,
        "confusion_matrix": cm,
    }
    metrics_path = out_dir / f"{model_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metryki zapisane do: {metrics_path}")

    # krzywa uczenia
    plot_path = out_dir / f"{model_name}_learning_curve.png"
    plot_learning_curves(history, plot_path)
    print(f"Krzywa uczenia zapisana do: {plot_path}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())