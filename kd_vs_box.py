import time
import numpy as np
from scipy.spatial import KDTree
from matplotlib import pyplot as plt


def query_rect(points, min_x, min_y, max_x, max_y):
    points[
        np.all(
            np.logical_and(
                np.array([min_x, min_y]) <= points,
                points <= np.array([max_x, max_y]),
            ),
            axis=1,
        )
    ]


def query_kdtree(tree, center, r):
    tree.query_ball_point(center, r)


if __name__ == "__main__":
    D_P = 1000
    D_Q = 0.05 * D_P

    CENTER = (D_P * 0.5, D_P * 0.5)
    R = 0.5 * D_Q
    MIN_X = D_P * 0.5 - R
    MIN_Y = D_P * 0.5 - R
    MAX_X = D_P * 0.5 + R
    MAX_Y = D_P * 0.5 + R

    NS = [int(n) for n in np.linspace(10_000, 100_000, 20)]
    # NS = [int(n) for n in np.linspace(100, 1000, 20)]
    POINT_SETS = [np.random.randint(low=0, high=D_P, size=(n, 2)) for n in NS]

    times_kd = []
    times_np = []

    for points in POINT_SETS:
        tree = KDTree(points)
        start = time.perf_counter()
        query_kdtree(tree, CENTER, R)
        times_kd.append(time.perf_counter() - start)

        start = time.perf_counter()
        query_rect(points, MIN_X, MIN_Y, MAX_X, MAX_Y)
        times_np.append(time.perf_counter() - start)

    plt.plot(NS, times_kd, label="kd")
    plt.plot(NS, times_np, label="np")
    # plt.plot(NS, np.array(times_np) / np.array(times_kd))
    plt.legend(loc="upper left")
    plt.show()
    # 0.0001 = 10000 * 1/s
