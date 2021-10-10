from shapely import geometry as sply_geometry
from scipy.spatial import KDTree
import numpy as np
import time
from itertools import combinations


### helper ###
def get_points_along_line(line, point_dist, include_end=False):
    # TODO use approximation covering tolerance factor to calculate circle_dist
    points = [line.interpolate(d) for d in range(0, line.length, point_dist)]
    if include_end:
        points.append(line.boundary[1])
    return points


def bounds2poly(bounds):
    min_x, min_y, max_x, max_y = bounds
    return sply_geometry.Polygon(
        [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
    )


### PIR = points in rectangle ###
# all PIR functions are written assuming that the passed arguments do not need
# any preprocessing. This is done to ensure comparability while profiling but has
# the downside of loosing a common interface
def get_pic_kdtree(tree, bounds):
    min_x, min_y, max_x, max_y = bounds
    return tree.query_ball_point(
        [min_x + 0.5 * (max_x - min_x), min_y + 0.5 * (max_y - min_y)]
    )


def get_pir_np(point_set, bounds):
    min_x, min_y, max_x, max_y = bounds
    return point_set[
        np.all(
            np.logical_and(
                np.array([min_x, min_y]) <= point_set,
                point_set <= np.array([max_x, max_y]),
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


def get_pir_sply(point_set, rect_poly):
    return list(filter(rect_poly.contains, point_set))


def get_pir_kdtree(tree, bounds):
    min_x, min_y, max_x, max_y = bounds

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

    ball_centers = get_points_along_line(central_line, radius)

    contained_points = np.unique(
        np.concatenate(
            [tree.query_ball_point(center) for center in ball_centers], axis=0
        ),
        axis=0,
    )

    return contained_points


def get_grid_subsets(
    points,
    n_dx,
    n_dy,
    x_s,
    y_s,
    use_subgrid=True,
    pir_1="np",
    pir_2="np",
    profiling=None,
):
    start = time.perf_counter()

    profiling["preprocessing"] = 0
    profiling["pir_1"] = 0
    profiling["pir_2"] = 0
    profiling["conversion_points"] = 0
    profiling["conversion_rect"] = 0
    profiling["translation"] = 0

    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)

    x_offsets = np.linspace(min_x, max_x, n_dx)
    y_offsets = np.linspace(min_y, max_y, n_dy)

    pir_functions = {
        "np": get_pir_np,
        "sply": get_pir_sply,
        "kdtree": get_pir_kdtree,
        "kdtree_pic": get_pic_kdtree,
    }

    if pir_1 == "kdtree_pic" and use_subgrid:
        raise ValueError("PIC only allowed for second level")

    pir_1_func = pir_functions[pir_1]
    pir_2_func = pir_functions[pir_2]

    if pir_1 == "kdtree_pic":
        pir_1 = "kdtree"
    if pir_2 == "kdtree_pic":
        pir_2 = "kdtree"

    # np: points_np --> points_np
    # sply: multipoints --> list[sply.points]
    # kdtree: tree --> points_np
    # kdtree_pic: tree -> points_np

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

    profiling["preprocessing"] += time.perf_counter() - start

    start = time.perf_counter()
    points = conversions_point_set[(None, pir_1)](points)
    profiling["conversion_points"] += time.perf_counter() - start

    if not use_subgrid:

        for x_off, y_off in combinations(x_offsets, y_offsets):

            start = time.perf_counter()
            rect = conversions_rect[pir_1](
                (
                    min_x + x_off,
                    min_y + y_off,
                    max_x + x_off + x_s,
                    max_y + y_off + y_s,
                )
            )
            profiling["conversion_points"] += time.perf_counter() - start

            start = time.perf_counter()
            final_subset = pir_1_func(points, rect)
            profiling["pir_1"] += time.perf_counter() - start

            yield final_subset

    else:

        start = time.perf_counter()
        if max_x - min_x > max_y - min_y:  # TODO check which >< (see calculations)
            ax_order = ("x", "y")
            offsets_1 = x_offsets
            offsets_2 = y_offsets
        else:
            ax_order = ("y", "x")
            offsets_1 = y_offsets
            offsets_2 = x_offsets
        profiling["preprocessing"] += time.perf_counter() - start

        for off_ax_1 in offsets_1:

            start = time.perf_counter()
            if ax_order == ("x", "y"):
                bounds = (
                    min_x,
                    min_y + off_ax_1,
                    max_x,
                    max_y + off_ax_1 + y_s,
                )
            elif ax_order == ("y", "x"):
                bounds = (
                    min_x + off_ax_1,
                    min_y,
                    max_x + off_ax_1 + x_s,
                    max_y,
                )
            profiling["translation"] += time.perf_counter() - start

            start = time.perf_counter()
            rect = conversions_rect[pir_1](bounds)
            profiling["conversion_rect"] += time.perf_counter() - start

            start = time.perf_counter()
            subset_ax_1 = pir_1_func(points, rect)
            profiling["pir_1"] += time.perf_counter() - start

            start = time.perf_counter()
            subset_ax_1 = conversions_point_set[(pir_1, pir_2)](subset_ax_1)
            profiling["conversion_points"] += time.perf_counter() - start

            for off_ax_2 in offsets_2:

                start = time.perf_counter()
                if ax_order == ("x", "y"):
                    bounds = (
                        min_x + off_ax_2,
                        min_y + off_ax_1,
                        max_x + off_ax_2 + x_s,
                        max_y + off_ax_1 + y_s,
                    )
                elif ax_order == ("y", "x"):
                    bounds = (
                        min_x + off_ax_1,
                        min_y + off_ax_2,
                        max_x + off_ax_1 + x_s,
                        max_y + off_ax_2 + y_s,
                    )
                profiling["translation"] += time.perf_counter() - start

                start = time.perf_counter()
                rect = conversions_rect[pir_2](bounds)
                profiling["conversion_rect"] += time.perf_counter() - start

                start = time.perf_counter()
                final_subset = pir_2_func(subset_ax_1, rect)
                profiling["pir_2"] += time.perf_counter() - start

                yield final_subset


if __name__ == "__main__":

    # create point set
    X_P = 100
    Y_P = 40
    N_P = 10_000
    POINTS = np.array([np.random.rand(N_P) * X_P, np.random.rand(N_P) * Y_P]).T

    # create shape (triangle)
    X_S = 10
    Y_S = 10

    N_DX = 20
    N_DY = 20

    profiling = {}

    result = list(
        get_grid_subsets(
            POINTS,
            N_DX,
            N_DY,
            X_S,
            Y_S,
            use_subgrid=True,
            pir_1="np",
            pir_2="np",
            profiling=profiling,
        )
    )
    print(len(result))
    print([len(ss) for ss in result[:10]])
    print(profiling)
