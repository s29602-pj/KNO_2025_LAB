import tensorflow as tf
from PIL import Image
import numpy as np


def wczytaj_i_przygotuj(image_path: str) -> np.ndarray:

    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def main() -> int:

    image_path = sys.argv[1]

    model = tf.keras.models.load_model("mnist_model.keras")

    img_array = wczytaj_i_przygotuj(image_path)
    predictions = model.predict(img_array)
    predicted_digit = np.argmax(predictions)

    print(f"➡️  Model rozpoznał cyfrę na {image_path}: {predicted_digit}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
