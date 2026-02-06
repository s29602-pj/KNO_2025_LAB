import numpy as np
import tensorflow as tf


def rotacja_punktu_tf(punkt, kat_rad):
    """
    punkt = tf.convert_to_tensor(punkt, dtype=tf.float32)
    kat_rad = tf.convert_to_tensor(kat_rad, dtype=point.dtype)
    """

    c = tf.cos(kat_rad)
    s = tf.sin(kat_rad)
    rotacja = tf.stack([[c, -s], [s, c]])
#Tensor to wielowymiarowa tablica liczb
    zrotowany_punkt = tf.linalg.matvec(rotacja, punkt)
    return zrotowany_punkt


punkt = tf.constant([1.0, 0.0], dtype=tf.float32)
kat_rad = tf.constant(np.pi / 2, dtype=tf.float32) # 90 stopni

rotacja_punktu_tf = rotacja_punktu_tf(punkt, kat_rad)
print(tf.round(rotacja_punktu_tf * 1e6) / 1e6)
#print("TensorFlow zrotowany punkt:", rotacja_punktów_tf.numpy())


oczekiwany_punkt = np.array([0.0, 1.0])
assert np.allclose(
    rotacja_punktu_tf.numpy(), oczekiwany_punkt, atol=1e-7
), "Test TensorFlow nie przeszedł!"
print("Test przeszedł pomyślnie!")
