import numpy as np


def rotate_point_np(point, angle_rad):

    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    return rotation_matrix @ point


point = np.array([1.0, 0.0])
angle = np.pi / 2

rotated_point_np = rotate_point_np(point, angle)
print("NumPy rotated point:", rotated_point_np)


expected_point = np.array([0.0, 1.0])
assert np.allclose(
    rotated_point_np, expected_point, atol=1e-7
), "Test NumPy nie przeszedł!"
print("Test przeszedł pomyślnie!")
