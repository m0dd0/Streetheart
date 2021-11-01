import json
from pathlib import Path
import time
from itertools import product

import numpy as np
from numpy.lib import meshgrid
from shapely import geometry as sply_geometry
import matplotlib.pyplot as plt
import matplotlib as mpl


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


def query_transform_shape(
    shape,
    point_set,
    relative_shape_size,
    shape_circle_diameter_factor,
    shape_circle_overlap,
):
    shape = shape - np.min(shape, axis=0)
    shape = (
        shape
        * np.ptp(point_set, axis=0).min()
        / np.ptp(shape, axis=0).max()
        * relative_shape_size
    )
    shape = shape + np.min(point_set, axis=0)

    shape_circle_diameter = np.ptp(shape, axis=0).max() * shape_circle_diameter_factor
    distances = np.linalg.norm(np.diff(shape + shap, axis=0), axis=1)

    shape_length = _poly_length(shape)

    shape = resample_line(shape, shape_circle_overlap * shape_circle_diameter)

    return shape


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


def get_centers_and_rotations(
    shape_points_orig, point_set, shape_relative_size, n_x=10, n_y=10, n_phi=10
):
    # put the shape on the botton left corner of the point set
    shape_points_initial = _get_inital_shape(
        shape_points_orig, point_set, shape_relative_size
    )

    # rotate the shape at this position
    shape_center = _bbox_center(shape_points_initial)
    rotated_shapes = [
        _rotate_around_point(shape_points_initial, shape_center, phi)
        for phi in np.linspace(0, 359, n_phi)
    ]

    x_extent_shape, y_extent_shape = np.ptp(shape_points_initial, axis=0)
    min_x = point_set[:, 0].min() + x_extent_shape * 0.5
    max_x = point_set[:, 0].max() - x_extent_shape * 0.5
    min_y = point_set[:, 1].min() + y_extent_shape * 0.5
    max_y = point_set[:, 1].max() - y_extent_shape * 0.5
    xs = np.linspace(min_x, max_x, n_x)
    ys = np.linspace(min_y, max_y, n_y)
    centers = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)

    return centers, rotated_shapes
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

    # return query_point_sets


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

    shape = [(0, 0), (1, 1), (2, 0)]  # triangle
    shape_circle_diameter = np.ptp(shape, axis=0).max() * SHAPE_CIRCLE_DIAMETER_FACTOR
    n_shape_queries = f(shape_circle_diameter, SHAPE_CIRCLE_OVERLAP, shape_length)

    # fig, ax = plt.subplots()
    # ax.add_collection(mpl.collections.LineCollection(streets))
    # ax.scatter(nodes_resampled[:, 0], nodes_resampled[:, 1], s=10, c="red")
    # # ax.scatter(shape[:, 0], shape[:, 1], s=10, c="red")
    # ax.scatter(centers[:, 0], centers[:, 1], s=10, c="green")
    # ax.autoscale()
    # ax.set_aspect("equal")
    # plt.show()

    # it is incredibly fast to query even very large point sets with a kdtree
    # therefore no subbox approach is used
    # see kd_vs_box.py for details
