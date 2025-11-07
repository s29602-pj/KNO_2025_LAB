import tensorflow as tf
import matplotlib.pyplot as plt


def main() -> int:

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=1
    )

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Strata treningowa")
    plt.plot(history.history["val_loss"], label="Strata walidacyjna")
    plt.plot(history.history["accuracy"], label="Dokładność treningowa")
    plt.plot(history.history["val_accuracy"], label="Dokładność walidacyjna")
    plt.title("Krzywa uczenia modelu MNIST")
    plt.xlabel("Epoka")
    plt.ylabel("Wartość")
    plt.legend()
    plt.grid(True)
    plt.show()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
