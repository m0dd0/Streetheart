import numpy as np


def poly_length(line):
    return np.sum(np.linalg.norm(np.diff(np.append(line, line[0]), axis=0), axis=1))


def create_rotation_matrix(a):
    a = a * np.pi / 180
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def rotate_around_point(points, center, phi):
    return (points - center) @ create_rotation_matrix(phi) + center


def bbox_center(points):
    x_extent, y_extent = np.ptp(points, axis=0)
    return np.array(
        [points[:, 0].min() + x_extent * 0.5, points[:, 1].min() + y_extent * 0.5]
    )
