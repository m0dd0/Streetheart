import time
from scipy.spatial import KDTree
import numpy as np
from numba import jit


def get_pir_kdtree(tree, center, r):
    return tree.query_ball_point(center, r)


def get_pir_np(points, bounds):
    min_x, min_y, max_x, max_y = bounds
    return points[
        np.all(
            np.logical_and(
                np.array([min_x, min_y]) <= points,
                points <= np.array([max_x, max_y]),
            ),
            axis=1,
        )
    ]
    # equivalent to but little bit slower:
    # return point_set[
    #     np.logical_and(
    #         np.logical_and(point_set[:, 0] > min_x, point_set[:, 0] < max_x),
    #         np.logical_and(point_set[:, 1] > min_y, point_set[:, 1] < max_x),
    #     )
    # ]


@jit(nopython=True)
def query_np(points, bounds):
    # for bounds in query_bounds:
    points[
        np.logical_and(
            np.logical_and(points[:, 0] > bounds[0], points[:, 0] < bounds[2]),
            np.logical_and(points[:, 1] > bounds[1], points[:, 1] < bounds[3]),
        )
    ]
    # points[
    #     np.all(
    #         np.logical_and(
    #             bounds[:2] <= points,
    #             points <= bounds[2:],
    #         ),
    #         axis=1,
    #     )
    # ]

    # min_x, min_y, max_x, max_y = bounds
    # points[
    #     np.all(
    #         np.logical_and(
    #             np.array([min_x, min_y]) <= points,
    #             points <= np.array([max_x, max_y]),
    #         ),
    #         axis=1,
    #     )
    # ]


@jit
def query_kdtree(tree, centers, r):
    for c in centers:
        tree.query_ball_point(c, r)


if __name__ == "__main__":
    X_P = 10
    Y_P = 10
    N_P = 1000
    POINTS = np.array([np.random.rand(N_P) * X_P, np.random.rand(N_P) * Y_P]).T
    TREE = KDTree(POINTS)

    # number of "shape queries" independent of usage of grid or not
    # only point set size changes when using the grid
    # n_dx*n_dy*n_dp*n_approx
    N_QUERIES = 20 * 20 * 30 * 20

    QUERY_BOUNDS_MIN_X = np.random.randint(0, 9, size=N_QUERIES)
    QUERY_BOUNDS_MIN_Y = np.random.randint(0, 9, size=N_QUERIES)
    QUERY_BOUNDS_MAX_X = QUERY_BOUNDS_MIN_X + 1
    QUERY_BOUNDS_MAX_Y = QUERY_BOUNDS_MIN_Y + 1
    QUERY_BOUNDS = np.array(
        [QUERY_BOUNDS_MIN_X, QUERY_BOUNDS_MIN_Y, QUERY_BOUNDS_MAX_X, QUERY_BOUNDS_MAX_Y]
    ).T

    start = time.perf_counter()
    for bounds in QUERY_BOUNDS:
        query_np(POINTS, bounds)
    print(time.perf_counter() - start)

    start = time.perf_counter()
    for bounds in QUERY_BOUNDS:
        query_np(POINTS, bounds)
    # query_np(POINTS, QUERY_BOUNDS)
    print(time.perf_counter() - start)

    # QUERY_CENTERS = np.random.randint(0, 9, size=(N_QUERIES, 2))

    # start = time.perf_counter()
    # query_kdtree(TREE, QUERY_CENTERS, 0.5)
    # print(time.perf_counter() - start)

    # start = time.perf_counter()
    # query_kdtree(TREE, QUERY_CENTERS, 0.5)
    # print(time.perf_counter() - start)
