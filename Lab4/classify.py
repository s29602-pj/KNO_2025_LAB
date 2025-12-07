from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

from common import CLASS_NAMES


def preprocess_image(path: Path, target_size=(28, 28)) -> np.ndarray:
    """
    Wczytuje obrazek, skaluje do 28x28, konwertuje do skali szarości
    i stosuje negatyw. Zwraca tablicę (28,28).
    """
    img = Image.open(path).convert("L")  # grayscale
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = 1.0 - arr  # negatyw
    return arr


def prepare_for_model(model, img: np.ndarray) -> np.ndarray:
    """
    Dopasowuje kształt wejścia do modelu (.keras).
    Obsługuje modele gęste (flatten) i CNN (28x28x1).
    """
    input_shape = model.input_shape  # np. (None, 784) albo (None, 28, 28, 1)

    if len(input_shape) == 2:
        # (None, 784) – sieć gęsta
        x = img.reshape(1, -1)
    elif len(input_shape) == 3:
        # (None, 28, 28)
        x = img.reshape(1, 28, 28)
    else:
        # (None, 28, 28, 1)
        x = img.reshape(1, 28, 28, 1)
    return x


def parse_args():
    parser = argparse.ArgumentParser(
        description="Klasyfikacja obrazka modelem Fashion-MNIST."
    )
    parser.add_argument(
        "image_path",
        help="Ścieżka do obrazka do sklasyfikowania.",
    )
    parser.add_argument(
        "--model",
        default="models/fashion_cnn.keras",
        help="Ścieżka do pliku .keras z wytrenowanym modelem.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image_path)
    model_path = Path(args.model)

    if not image_path.is_file():
        print(f"Nie znaleziono obrazka: {image_path}")
        return 1
    if not model_path.is_file():
        print(f"Nie znaleziono modelu: {model_path}")
        return 1

    print(f"Ładuję model z: {model_path}")
    model = keras.models.load_model(model_path)

    img = preprocess_image(image_path)
    x = prepare_for_model(model, img)

    probas = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probas))
    confidence = float(probas[pred_idx])

    class_name = CLASS_NAMES[pred_idx]

    print(f"Plik: {image_path}")
    print(f"Przewidziana klasa: {pred_idx} ({class_name})")
    print(f"Pewność predykcji: {confidence:.4f}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())