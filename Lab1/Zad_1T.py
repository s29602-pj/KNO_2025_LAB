import numpy as np
import tensorflow as tf


def rotate_point_tf(point, angle_rad):

    c = tf.cos(angle_rad)
    s = tf.sin(angle_rad)
    rotation_matrix = tf.stack([[c, -s], [s, c]])

    rotated_point = tf.linalg.matvec(rotation_matrix, point)
    return rotated_point


point = tf.constant([1.0, 0.0], dtype=tf.float32)
angle = np.pi / 2  # 90 stopni

rotated_point_tf = rotate_point_tf(point, angle)
print("TensorFlow rotated point:", rotated_point_tf.numpy())


expected_point = np.array([0.0, 1.0])
assert np.allclose(
    rotated_point_tf.numpy(), expected_point, atol=1e-7
), "Test TensorFlow nie przeszedł!"
print("Test przeszedł pomyślnie!")
