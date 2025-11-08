import argparse
import math

import numpy as np
import tensorflow as tf


@tf.function
def solve_linear_system(A, b):
    x = tf.linalg.solve(A, tf.reshape(b, (-1, 1)))
    return tf.reshape(x, (-1,))


def main():
    parser = argparse.ArgumentParser(
        description="Rozwiązywanie układów równań liniowych metodą macierzową."
    )
    parser.add_argument(
        "--A",
        nargs="+",
        required=True,
        help="Współczynniki macierzy A w kolejności wierszami, np. 3 1 1 2 dla macierzy 2x2",
    )
    parser.add_argument(
        "--b", nargs="+", required=True, help="Wektor prawej strony, np. 9 8"
    )

    args = parser.parse_args()

    A_flat = np.array(list(map(float, args.A)))
    b = np.array(list(map(float, args.b)))

    n = int(math.sqrt(len(A_flat)))
    if n * n != len(A_flat):
        raise ValueError(
            "Liczba elementów w A nie pozwala utworzyć macierzy kwadratowej."
        )

    if len(b) != n:
        raise ValueError(
            f"Liczba elementów wektora b ({len(b)}) nie pasuje do rozmiaru macierzy ({n})."
        )

    A_tf = tf.convert_to_tensor(A_flat.reshape((n, n)), dtype=tf.float32)
    b_tf = tf.convert_to_tensor(b, dtype=tf.float32)

    if tf.linalg.det(A_tf) == 0:
        raise ValueError("Macierz A jest osobliwa, brak unikalnego rozwiązania.")

    result = solve_linear_system(A_tf, b_tf)

    print("Rozwiązanie układu równań:", result.numpy())


if __name__ == "__main__":
    main()

    # python zad_4.py --A 3 1 1 2 --b 9 8
