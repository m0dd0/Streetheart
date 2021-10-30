import json
from pathlib import Path

import numpy as np
from shapely import geometry as sply_geometry
from plotly import graph_objects as go
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
    return np.array(streets)


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


if __name__ == "__main__":
    with open(Path(__file__).parent / "overpass_sample_data.json", "r") as f:
        osm_data = json.load(f)

    nodes_osm, streets_osm = split_osm_data(osm_data)

    streets = osm_streets_to_lines(streets_osm, nodes_osm)
    nodes = osm_nodes_to_coordinates(nodes_osm)

    RESAMPLE_FACTOR = 0.01
    resample_delta = np.ptp(nodes, axis=0).min() * RESAMPLE_FACTOR

    # streets have differnt length so dont convert to np.array
    streets_resampled = [resample_line(s, resample_delta) for s in streets]
    nodes_resampled = np.unique(
        np.array([p for s in streets_resampled for p in s]), axis=0
    )

    fig, ax = plt.subplots()
    ax.add_collection(mpl.collections.LineCollection(streets))
    ax.scatter(nodes_resampled[:, 0], nodes_resampled[:, 1], s=10, c="red")
    ax.autoscale()
    ax.set_aspect("equal")
    plt.show()

    # it is incredibly fast to query even very large point sets with a kdtree
    # therefore no subbox approach is used
    # see kd_vs_box.py for details