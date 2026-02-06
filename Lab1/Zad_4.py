import argparse
import math
import numpy as np
import tensorflow as tf


@tf.function
def solve_linear_system_unique(A: tf.Tensor, b: tf.Tensor) -> tf.Tensor:#liniowe
    """Rozwiązuje Ax=b zakładając jednoznaczne rozwiązanie (A odwracalna)."""
    x = tf.linalg.solve(A, tf.reshape(b, (-1, 1)))
    return tf.reshape(x, (-1,))


@tf.function
def solve_linear_system_lstsq(A: tf.Tensor, b: tf.Tensor) -> tf.Tensor:#metoda najmniejszych kwadratów
    x = tf.linalg.lstsq(A, tf.reshape(b, (-1, 1)), fast=False)
    return tf.reshape(x, (-1,))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rozwiązywanie układów równań liniowych Ax=b (TensorFlow)."
    )
    parser.add_argument(
        "--A",
        nargs="+",
        required=True,
        help="Współczynniki macierzy A w kolejności wierszami, np. 3 1 1 2 dla 2x2",
    )
    parser.add_argument(
        "--b",
        nargs="+",
        required=True,
        help="Wektor prawej strony, np. 9 8",
    )
    args = parser.parse_args()

    A_flat = np.array(list(map(float, args.A)), dtype=np.float32)
    b = np.array(list(map(float, args.b)), dtype=np.float32)

    n = int(math.isqrt(len(A_flat)))
    if n * n != len(A_flat):
        raise ValueError("Liczba elementów w A nie pozwala utworzyć macierzy kwadratowej n×n.")

    if len(b) != n:
        raise ValueError(f"Długość b ({len(b)}) nie pasuje do rozmiaru macierzy ({n}).")

    A_tf = tf.convert_to_tensor(A_flat.reshape((n, n)), dtype=tf.float32)
    b_tf = tf.convert_to_tensor(b, dtype=tf.float32)

    # Sprawdzenie "czy istnieje rozwiązanie" przez rangi
    b_col = tf.reshape(b_tf, (-1, 1))
    aug = tf.concat([A_tf, b_col], axis=1)

    rank_A = int(tf.linalg.matrix_rank(A_tf).numpy())
    rank_aug = int(tf.linalg.matrix_rank(aug).numpy())

    if rank_A < rank_aug:
        raise ValueError("Brak rozwiązania: układ jest sprzeczny (rank(A) < rank([A|b])).")

    if rank_A == n:
        # Jednoznaczne rozwiązanie
        x = solve_linear_system_unique(A_tf, b_tf)
        print("Jednoznaczne rozwiązanie:", x.numpy())
    else:
        # Nieskończenie wiele rozwiązań — zwróć jedno (np. najmniejszych kwadratów)
        x = solve_linear_system_lstsq(A_tf, b_tf)
        print("Nieskończenie wiele rozwiązań (pokazuję jedno z nich):", x.numpy())


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(e)

    # python zad_4.py --A 3 1 1 2 --b 9 8
