from shapely import geometry as sply_geometry
from shapely import affinity as sply_affinity
from shapely import prepared as sply_prepared
from scipy.spatial import KDTree
import numpy as np
import time
from itertools import combinations


# PIR = points in rectangle


def get_pir_np(point_set, min_x, min_y, max_x, max_y):
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


# helper
def get_pir_sply(point_set, rect_poly):
    return list(filter(rect_poly.contains, point_set))


def get_circles_along_line(line, circle_dist, include_end=False):
    # TODO use approximation covering tolerance factor to calculate circle_dist
    center_points = [line.interpolate(d) for d in range(0, line.length, circle_dist)]
    if include_end:
        center_points.append(line.boundary[1])
    return center_points


def get_pir_kdtree(tree, min_x, min_y, max_x, max_y):
    if max_x - min_x > max_y - min_y:  # horizontal rectangle
        radius = 0.5 * (max_y - min_y)
        central_line = sply_geometry.LineString(
            [(min_x + radius, min_y + radius), (max_x - radius, min_y + radius)]
        )
    else:  # vertical rectangle
        radius = 0.5 * (max_x - min_x)
        central_line = sply_geometry.LineString(
            [(min_x + radius, min_y + radius), (min_x + radius, max_y - radius)]
        )

    ball_centers = get_circles_along_line(central_line, radius)

    contained_points = np.unique(
        np.concatenate(
            [tree.query_ball_point(center) for center in ball_centers], axis=0
        ),
        axis=0,
    )

    return contained_points


# helper
def bounds2poly(bounds):
    min_x, min_y, max_x, max_y = bounds
    return sply_geometry.Polygon(
        [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
    )


def get_grid_subsets(
    points, n_dx, n_dy, s_x, s_y, use_subgrid=True, pir_1="np", pir_2="np"
):
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)

    x_offsets = np.linspace(min_x, max_x, n_dx)
    y_offsets = np.linspace(min_y, max_y, n_dy)

    pir_functions = {"np": get_pir_np, "sply": get_pir_sply, "kdtree": get_pir_kdtree}

    pir_1_func = pir_functions[pir_1]
    pir_2_func = pir_functions[pir_2]

    # np: points_np --> points_np
    # sply: multipoints --> list[sply.points]
    # kdtree: tree --> points_np

    conversions_point_set = {
        (None, "np"): lambda points_np: points_np,
        (None, "sply"): sply_geometry.MultiPoint,
        (None, "kdtree"): KDTree,
        ("np", "np"): lambda points_np: points_np,
        ("np", "sply"): sply_geometry.MultiPoint,
        ("np", "kdtree"): KDTree,
        ("sply", "np"): lambda sply_points: np.array([(p.x, p.y) for p in sply_points]),
        ("sply", "sply"): sply_geometry.MultiPoint,
        ("sply", "kdtree"): lambda sply_points: KDTree(
            np.array([(p.x, p.y) for p in sply_points])
        ),
        ("kdtree", "np"): lambda points_np: points_np,
        ("kdtree", "sply"): sply_geometry.MultiPoint,
        ("kdtree", "kdtree"): KDTree,
        ("np", None): lambda points_np: points_np,
        ("scply", None): lambda sply_points: np.array(
            [(p.x, p.y) for p in sply_points]
        ),
        ("kdtree", None): lambda points_np: points_np,
    }

    conversions_rect = {
        "np": lambda bounds: bounds,
        "scply": lambda bounds: (bounds2poly,),
        "kdtree": lambda bounds: bounds,
    }

    points = conversions_point_set[(None, pir_1)](points)

    if not use_subgrid:
        for x_off, y_off in combinations(x_offsets, y_offsets):
            yield pir_1_func(
                points,
                conversions_rect[pir_1](
                    (
                        min_x + x_off,
                        min_y + y_off,
                        max_x + x_off + s_x,
                        max_y + y_off + s_y,
                    )
                ),
            )

    else:

        if max_x - min_x > max_y - min_y:  # TODO check which >< (see calculations)
            ax_order = ("x", "y")
            offsets_1 = x_offsets
            offsets_2 = y_offsets
        else:
            ax_order = ("y", "x")
            offsets_1 = y_offsets
            offsets_2 = x_offsets

        for off_ax_1 in offsets_1:

            if ax_order == ("x", "y"):
                bounds = (
                    min_x,
                    min_y + off_ax_1,
                    max_x,
                    max_y + off_ax_1 + s_y,
                )
            elif ax_order == ("y", "x"):
                bounds = (
                    min_x + off_ax_1,
                    min_y,
                    max_x + off_ax_1 + s_x,
                    max_y,
                )

            subset_ax_1 = pir_1_func(points, *conversions_rect[pir_1](bounds))
            subset_ax_1 = conversions_point_set[(pir_1, pir_2)](subset_ax_1)

            for off_ax_2 in offsets_2:

                if ax_order == ("x", "y"):
                    bounds = (
                        min_x + off_ax_2,
                        min_y + off_ax_1,
                        max_x + off_ax_2 + s_x,
                        max_y + off_ax_1 + s_y,
                    )
                elif ax_order == ("y", "x"):
                    bounds = (
                        min_x + off_ax_1,
                        min_y + off_ax_2,
                        max_x + off_ax_1 + s_x,
                        max_y + off_ax_2 + s_y,
                    )

                yield pir_2_func(subset_ax_1, *conversions_rect[pir_2](bounds))


if __name__ == "__main__":

    # create point set
    X_P = 100
    Y_P = 40
    N_P = 10_000
    POINTS = np.array([np.random.rand(N_P) * X_P, np.random.rand(N_P) * Y_P]).T

    # create shape (triangle)
    X_S = 10
    Y_S = 10
    SHAPE = [(0, 0), (0, Y_S / 2), (X_S, 0)]

    # RANDOM_GRID_COORS = np.random.rand(
    #     100, 4
    # )  # TODO account for point extents X_P Y_P !!!

    # start = time.perf_counter()
    # for rect_pos
    # get_pir_np(
    #     points,
    # )
