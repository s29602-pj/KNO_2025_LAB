from __future__ import annotations
from typing import Tuple

import numpy as np
from tensorflow import keras

# Nazwy klas z Fashion-MNIST (0–9)
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_fashion_mnist(for_cnn: bool) -> Tuple[tuple, tuple]:
    """
    Ładuje dane Fashion-MNIST i przygotowuje je dla:
    - sieci gęstej  (flatten)  gdy for_cnn=False
    - sieci splotowej (28,28,1) gdy for_cnn=True
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    if for_cnn:
        x_train = np.expand_dims(x_train, -1)  # (N,28,28,1)
        x_test = np.expand_dims(x_test, -1)
    else:
        x_train = x_train.reshape((-1, 28 * 28))
        x_test = x_test.reshape((-1, 28 * 28))

    return (x_train, y_train), (x_test, y_test)