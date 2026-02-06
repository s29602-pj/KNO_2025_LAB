import tensorflow as tf


def rownanie_liniowe(A, b):

    A = tf.convert_to_tensor(A, dtype=tf.float32)
    b = tf.convert_to_tensor(b, dtype=tf.float32)

    x = tf.linalg.solve(A, tf.reshape(b, (-1, 1)))

    return tf.reshape(x, (-1,))


if __name__ == "__main__":
    A = [[3, 1], [1, 2]]
    b = [9, 8]

    rozwiazanie = rownanie_liniowe(A, b)
    print("Rozwiązanie układu:", rozwiazanie.numpy())
