import json
from pathlib import Path
import time
from itertools import product

import numpy as np
from numpy.lib import meshgrid
from shapely import geometry as sply_geometry
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import KDTree


def _poly_length(line):
    return np.sum(np.linalg.norm(np.diff(np.append(line, line[0]), axis=0), axis=1))


def _create_rotation_matrix(a):
    a = a * np.pi / 180
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _rotate_around_point(points, center, phi):
    return (points - center) @ _create_rotation_matrix(phi) + center


def _bbox_center(points):
    x_extent, y_extent = np.ptp(points, axis=0)
    return np.array(
        [points[:, 0].min() + x_extent * 0.5, points[:, 1].min() + y_extent * 0.5]
    )


def split_osm_data(osm_data):
    """[summary]

    Args:
        osm_data ([type]): [description]
    """
    nodes_osm = {}
    streets_osm = {}
    for elem in osm_data["elements"]:
        elem_id = elem.pop("id")
        elem_type = elem.pop("type")
        {"node": nodes_osm, "way": streets_osm}[elem_type][elem_id] = elem
    return nodes_osm, streets_osm


def osm_streets_to_lines(streets_osm, nodes_osm):
    """get all streets as list of coords

    Args:
        osm_streets ([type]): [description]
        osm_nodes ([type]): [description]
    """
    streets = []
    for s_osm in streets_osm.values():
        s = []
        for n_osm in s_osm["nodes"]:
            n = nodes_osm[n_osm]
            s.append((n["lon"], n["lat"]))
        streets.append(s)
    return streets


def osm_nodes_to_coordinates(nodes_osm):
    """get all the coords of the nodes

    Args:
        nodes_osm ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.array([(n["lon"], n["lat"]) for n in nodes_osm.values()])


def resample_line(line, delta):
    # array of vecotors between points in line
    delta_vectors = np.diff(line, axis=0)
    # distances between points in line
    distances = np.linalg.norm(delta_vectors, axis=1)
    # the delta vectors with a length of one
    delta_vectors_normed = delta_vectors / distances[:, None]
    # the accumulated distances from potin to point in line
    distances_acc = np.insert(np.cumsum(distances), 0, 0)

    sample_distances = np.arange(0, distances_acc[-1], delta)
    previous_indices = (
        np.searchsorted(distances_acc, sample_distances, side="right") - 1
    )

    resampled_line = []
    for d, i in zip(sample_distances, previous_indices):
        remaining_dist = d - distances_acc[i]
        p = line[i] + delta_vectors_normed[i] * remaining_dist
        resampled_line.append(p)

    return np.array(resampled_line)


def shape_to_initial_position(
    shape,
    point_set,
    relative_shape_size,
):
    shape = shape - np.min(shape, axis=0)
    shape = (
        shape
        * np.ptp(point_set, axis=0).min()
        / np.ptp(shape, axis=0).max()
        * relative_shape_size
    )
    shape = shape + np.min(point_set, axis=0)

    return shape


def get_query_circle_diameter(shape, query_circle_diameter_factor):
    return np.ptp(shape, axis=0).max() * query_circle_diameter_factor


def resample_shape(
    shape,
    query_circle_diameter,
    query_circle_relative_distance,
):
    query_circle_distance = query_circle_relative_distance * query_circle_diameter

    if np.linalg.norm(shape[0] - shape[-1]) > query_circle_distance:
        shape = np.append(shape, [shape[0]], axis=0)

    shape = resample_line(shape, query_circle_distance)

    return shape


def get_centers(point_set, shape_relative_size, n_x, n_y):
    shape_size_max = np.ptp(point_set, axis=0).min() * shape_relative_size

    min_x = point_set[:, 0].min() + shape_size_max * 0.5
    max_x = point_set[:, 0].max() - shape_size_max * 0.5
    min_y = point_set[:, 1].min() + shape_size_max * 0.5
    max_y = point_set[:, 1].max() - shape_size_max * 0.5

    xs = np.linspace(min_x, max_x, n_x)
    ys = np.linspace(min_y, max_y, n_y)

    centers = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)

    return centers


def get_rotations(shape, n_phi):
    shape_center = _bbox_center(shape)
    rotated_shapes = [
        _rotate_around_point(shape, shape_center, phi)
        for phi in np.linspace(0, 359, n_phi)
    ]

    return rotated_shapes


if __name__ == "__main__":
    with open(Path(__file__).parent / "overpass_sample_data.json", "r") as f:
        osm_data = json.load(f)

    nodes_osm, streets_osm = split_osm_data(osm_data)

    streets = osm_streets_to_lines(streets_osm, nodes_osm)
    nodes = osm_nodes_to_coordinates(nodes_osm)

    RESAMPLE_FACTOR = 0.02
    resample_delta = np.ptp(nodes, axis=0).min() * RESAMPLE_FACTOR

    # streets have differnt length so dont convert to np.array
    streets_resampled = [resample_line(s, resample_delta) for s in streets]
    nodes_resampled = np.unique(
        np.array([p for s in streets_resampled for p in s]), axis=0
    )

    shape = [(1, 1), (2, 2), (3, 1)]  # triangle
    SHAPE_RELATIVE_SIZE = 0.3
    shape = shape_to_initial_position(shape, nodes_resampled, SHAPE_RELATIVE_SIZE)
    # TODO calculate circle properties from given corridor width
    QUERY_CIRCLE_DIAMETER_FACTOR = 0.2
    query_circle_diameter = get_query_circle_diameter(
        shape, QUERY_CIRCLE_DIAMETER_FACTOR
    )
    QUERY_CIRCLE_RELATIVE_DIST = 0.8
    shape = resample_shape(shape, query_circle_diameter, QUERY_CIRCLE_RELATIVE_DIST)

    N_X = 20
    N_Y = 20
    N_PHI = 20
    centers = get_centers(nodes_resampled, SHAPE_RELATIVE_SIZE, N_X, N_Y)
    shape_rotations = get_rotations(shape, N_PHI)

    fig, ax = plt.subplots()
    ax.add_collection(mpl.collections.LineCollection(streets))
    ax.scatter(nodes_resampled[:, 0], nodes_resampled[:, 1], s=10, c="red")
    ax.scatter(shape[:, 0], shape[:, 1], s=10, c="green")
    ax.scatter(centers[:, 0], centers[:, 1], s=10, c="green")
    for c in shape:
        ax.add_artist(
            plt.Circle(c, query_circle_diameter / 2, color="green", fill=False)
        )
    ax.autoscale()
    ax.set_aspect("equal")
    plt.show()

    start = time.perf_counter()
    tree = KDTree(nodes_resampled)
    possible_routes = []
    for c in centers:
        for points in shape_rotations:
            points = points + c
            for p in points:
                contained_nodes = tree.query_ball_point(p, query_circle_diameter)
                print(len(contained_nodes))
                if len(contained_nodes) == 0:
                    break
                possible_routes.append(points)
    print(len(possible_routes))
    print(time.perf_counter() - start)
    # list of (center, rotated shape tuples)
    # using np.meshgrid wont work directly since the elements to build the product
    # from are not 1d
    # c_r_combs = np.array(list(product(centers, rotated_shapes)))
    # query_point_sets = c_r_combs[:, 1] + c_r_combs[:, 0]

    # query_point_sets = []
    # for c in centers:
    #     for shape_points in rotated_shapes:
    #         for p in shape_points:

    #         query_point_sets.append(shape_points + c)

    # it is incredibly fast to query even very large point sets with a kdtree
    # therefore no subbox approach is used
    # see kd_vs_box.py for details
