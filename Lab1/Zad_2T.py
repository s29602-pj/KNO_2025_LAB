import numpy as np
import tensorflow as tf


def rotacja_punktów_tf( punkty, kat_rad):
    """
     punkty = tf.convert_to_tensor( punkty, dtype=tf.float32)
    kat_rad = tf.cast(kat_rad,  punkty.dtype)

    """

    c = tf.cos(kat_rad)
    s = tf.sin(kat_rad)
    rotacja_matrix = tf.reshape(tf.stack([c, -s, s, c]), (2, 2))

    if len( punkty.shape) == 1:
        return tf.linalg.matvec(rotacja_matrix, punkty)
    else:
        return tf.transpose(rotacja_matrix @ tf.transpose( punkty))


punkty = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
kat_rad = tf.constant(np.pi / 2, dtype=tf.float32)
zrotowane_punkty = rotacja_punktów_tf(punkty, kat_rad)
print("TensorFlow zrotowane punkty:\n", zrotowane_punkty.numpy())
print(tf.round(zrotowane_punkty * 1e6) / 1e6)


oczekiwane_punkty = np.array([[0.0, 1.0], [-1.0, 0.0]])
assert np.allclose(
    zrotowane_punkty.numpy(), oczekiwane_punkty, atol=1e-7
), "Test TensorFlow nie przeszedł!"
print("Test TensorFlow dla wielu punktów przeszedł pomyślnie!")
