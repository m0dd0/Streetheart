from scipy.spatial import KDTree
import numpy as np
import time

X_P = 100
Y_P = 40
N_P = 10_000

X_S = 10
Y_S = 10

D_X = 1
D_Y = 1
D_PHI = 5

POINTS = np.array([np.random.rand(N_P) * X_P, np.random.rand(N_P) * Y_P]).T
SHAPE = [(0, 0), (0, Y_S / 2), (X_S, 0)]


def rectangle_subset_np(point_set, min_x, min_y, max_x, max_y):
    return point_set[
        np.all(
            np.logical_and(
                np.array([min_x, min_y]) <= point_set,
                point_set <= np.array([max_x, max_y]),
            ),
            axis=1,
        )
    ]
    # equivalent to but little bit faster:
    # return point_set[
    #     np.logical_and(
    #         np.logical_and(point_set[:, 0] > min_x, point_set[:, 0] < max_x),
    #         np.logical_and(point_set[:, 1] > min_y, point_set[:, 1] < max_x),
    #     )
    # ]


def get_subsets_primitive_np(points, x_min, y_min, x_max, y_max, d_x, d_y, x_s, y_s):
    x_offsets = np.linspace(x_min, x_max, d_x)
    y_offsets = np.linspace(y_min, y_max, d_y)

    for x_off in x_offsets:
        for y_off in y_offsets:
            yield rectangle_subset_np(points, x_off, y_off, x_off + x_s, y_off + y_s)


def get_subset_primitive_sply(points, d_x, d_y):
    x_offsets = np.linspace(x_min, x_max, d_x)
    y_offsets = np.linspace(y_min, y_max, d_y)

    for x_off in x_offsets:
        for y_off in y_offsets:
            pass


def get_subsets_optimized_np(points):
    x_offsets = np.linspace(x_min, x_max, d_x)
    y_offsets = np.linspace(y_min, y_max, d_y)

    # offsets_first, offsets_second = x_offsets, y_offsets
    # if len(x_offsets) > len(y_offsets):  # TODO check with claculations
    #     offsets_first, offsets_second = offsets_second, offsets_first

    for y_off in y_offsets:
        rectangle_subset_np()


# print(points)

# tree = KDTree(points)

# shape_points = np.random.rand(100, 2) * 2 + 5
# start = time.perf_counter()
# # shape_tree = KDTree(shape_points)
# # tree.query_ball_tree(shape_tree, 0.5)
# for p in shape_points:
#     tree.query_ball_point(p, 0.5)
# print((time.perf_counter() - start) * n_trafo)




def get_subsets(points, d_x, d_y, s_x, s_y, grid=True, pir_1="np", pir_2="np"):
    min_x, min_y = points.T.min()
    min_x, max_x = points.T.max()
    
    x_offsets = np.linspace(x_min, x_max, d_x)
    y_offsets = np.linspace(y_min, y_max, d_y)

    if not grid:
        for x_off, y_off in combinations(x_offsets, y_offsets):

        # if pir_1 == "np":


#  = get_extents(points):
#     x_offsets,
