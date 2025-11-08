import numpy as np
import tensorflow as tf


def rotate_points_tf(points, angle_rad):

    c = tf.cos(angle_rad)
    s = tf.sin(angle_rad)
    rotation_matrix = tf.reshape(tf.stack([c, -s, s, c]), (2, 2))

    if len(points.shape) == 1:
        return tf.linalg.matvec(rotation_matrix, points)
    else:
        return tf.transpose(rotation_matrix @ tf.transpose(points))


points = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
angle = np.pi / 2
rotated_points = rotate_points_tf(points, angle)
print("TensorFlow rotated points:\n", rotated_points.numpy())


expected_points = np.array([[0.0, 1.0], [-1.0, 0.0]])
assert np.allclose(
    rotated_points.numpy(), expected_points, atol=1e-7
), "Test TensorFlow nie przeszedł!"
print("Test TensorFlow dla wielu punktów przeszedł pomyślnie!")
