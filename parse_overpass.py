from pathlib import Path
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from shapely import geometry as sply_geometry
from shapely import affinity as sply_affinity
from shapely import prepared as sply_prepared
import numpy as np
import logging
import time
import random


def nodes_ways_from_osm(osm_json):
    """sort elemts by type and map their attributes to their id

    Args:
        osm_json ([type]): [description]

    Returns:
        [type]: [description]
    """
    nodes = {}
    ways = {}
    for elem in osm_json["elements"]:
        elem_id = elem.pop("id")
        elem_type = elem.pop("type")
        {"node": nodes, "way": ways}[elem_type][elem_id] = elem
    return (nodes, ways)


def create_street_graph(nodes, ways):
    """create graph which conatins all the data

    Args:
        nodes ([type]): [description]
        ways ([type]): [description]

    Returns:
        [type]: [description]
    """
    g = nx.Graph()
    g.add_nodes_from(nodes.items())
    for way_id, way_attrs in ways.items():
        way_nodes = way_attrs.pop("nodes")
        edges = list(zip(way_nodes[1:], way_nodes[:-1]))
        g.add_edges_from(edges, **way_attrs, way_id=way_id)
    return g


def get_edge_lines(g):
    """create a list of edges by their coordinate

    Args:
        g ([type]): [description]

    Returns:
        [type]: [description]
    """
    lines = []
    for start_node_id, end_node_id in g.edges:
        start_node = g.nodes[start_node_id]
        end_node = g.nodes[end_node_id]
        lines.append(
            [
                (start_node["lon"], start_node["lat"]),
                (end_node["lon"], end_node["lat"]),
            ]
        )
    return lines


def get_point_set(g, use_multipoint=True):
    """get all points

    Args:
        g ([type]): [description]

    Returns:
        [type]: [description]
    """
    points = [[n_attrs["lon"], n_attrs["lat"]] for _, n_attrs in g.nodes(data=True)]
    if use_multipoint:
        points = sply_geometry.MultiPoint(points)
    return points


# helper functions to get the x,y-extents og an geometry object
def get_extents(o):
    return (o.bounds[2] - o.bounds[0], o.bounds[3] - o.bounds[1])


def scale_polygon(point_set, polygon, size_factor=0.3):
    points_extents = get_extents(point_set)
    poly_extents = get_extents(polygon)

    if (points_extents[0] / points_extents[1]) < (poly_extents[0] / poly_extents[1]):
        scaling_axis = 0
    else:
        scaling_axis = 1
    scaling_factor = (
        points_extents[scaling_axis] * size_factor / poly_extents[scaling_axis]
    )

    return sply_affinity.scale(
        polygon, xfact=scaling_factor, yfact=scaling_factor, origin="center"
    )


def translate_to_start_position(point_set, polygon, position="nw"):
    if position == "nw":
        return sply_affinity.translate(
            polygon,
            xoff=point_set.bounds[0] - polygon.bounds[0],
            yoff=point_set.bounds[3] - polygon.bounds[3],
        )
    else:
        raise NotImplementedError("only north west start position implemented")


def dilate_polygon_absolute(polygon, thickness):
    """create the polygon to match with tolerance

    Args:
        polygon ([type]): [description]
        thickness ([type]): [description]

    Returns:
        [type]: [description]
    """
    poly_outer = polygon.buffer(0.5 * thickness)
    poly_inner = polygon.buffer(-0.5 * thickness)
    return poly_inner.symmetric_difference(poly_outer)


def dilate_polygon_relative(polygon, dilatation_factor=0.05):
    poly_extents = get_extents(polygon)
    diagonal_length = (poly_extents[0] ** 2 + poly_extents[1] ** 2) ** 0.5
    dilatation_thickness = dilatation_factor * diagonal_length
    return dilate_polygon_absolute(polygon, dilatation_thickness)


def get_dilated_polygon_borders(polygon):
    return (polygon.boundary.geoms[0], polygon.boundary.geoms[1])


def get_transformed_polygons(
    polygon,
    point_set,
    delta_x,
    delta_y,
    delta_phi=10,
):
    point_extents = get_extents(point_set)
    x_offsets = np.linspace(0, point_extents[0], delta_x)
    y_offsets = np.linspace(0, point_extents[1], delta_y)
    phi_offsets = np.linspace(0, 359, delta_phi)

    n = len(x_offsets) * len(y_offsets) * len(phi_offsets)
    i = 0

    for y_off in y_offsets:
        for x_off in x_offsets:
            for phi_off in phi_offsets:
                i += 1
                progress = i / n
                # TODO do transformation stepwise
                transformed_polygon = sply_affinity.rotate(
                    sply_affinity.translate(polygon, xoff=x_off, yoff=-y_off),
                    phi_off,
                )
                yield progress, transformed_polygon


def get_transformed_polygons_with_subsets(
    polygon, point_set, delta_x, delta_y, delta_phi=10
):
    point_extents = get_extents(point_set)
    x_offsets = np.linspace(0, point_extents[0], delta_x)
    y_offsets = np.linspace(0, point_extents[1], delta_y)
    phi_offsets = np.linspace(0, 359, delta_phi)

    n_yields = len(x_offsets) * len(y_offsets) * len(phi_offsets)
    i = 0

    polygon_extents = get_extents(polygon)
    grid_size = max(polygon_extents)
    east_abs = point_set.bounds[2]
    west_abs = point_set.bounds[0]

    for y_off in y_offsets:
        # build the subset along the horizontal x-axis
        north_y = point_set.bounds[3] - y_off
        south_y = north_y - grid_size
        filter_rect = sply_geometry.Polygon(
            [
                (west_abs, north_y),
                (east_abs, north_y),
                (east_abs, south_y),
                (west_abs, south_y),
            ]
        )
        # filter_rect = sply_prepared.prep(filter_rect)
        point_subset_x = sply_geometry.MultiPoint(
            list(filter(filter_rect.contains, point_set))
        )

        for x_off in x_offsets:
            # build the grid subset in which the polygon rotates
            west_x = west_abs + x_off
            east_x = west_abs + grid_size
            filter_rect = sply_geometry.Polygon(
                [
                    (west_x, north_y),
                    (east_x, north_y),
                    (east_x, south_y),
                    (west_x, south_y),
                ]
            )
            # filter_rect = sply_prepared.prep(filter_rect)
            point_subset_xy = sply_geometry.MultiPoint(
                list(filter(filter_rect.contains, point_subset_x))
            )

            for phi_off in phi_offsets:
                i += 1
                progress = i / n_yields
                # TODO do transforms stepwise
                transformed_polygon = sply_affinity.rotate(
                    sply_affinity.translate(polygon, xoff=x_off, yoff=-y_off),
                    phi_off,
                )
                yield progress, point_subset_xy, transformed_polygon


def draw_dilated_polygon(polygon, ax):
    border_1, border_2 = get_dilated_polygon_borders(polygon)
    border_1_mpl = ax.plot(*border_1.xy, color="red")[0]
    border_2_mpl = ax.plot(*border_2.xy, color="red")[0]

    return border_1_mpl, border_2_mpl


def draw_polygon(polygon, ax):
    return ax.plot(*polygon.exterior.xy, color="red")


def draw_streets(g, ax):
    ax.add_collection(mpl.collections.LineCollection(get_edge_lines(g)))
    ax.autoscale()
    ax.axis("equal")


def draw_point_set(point_set, ax):
    return ax.scatter(*zip(*[(p.x, p.y) for p in point_set]))


if __name__ == "__main__":
    # TODO setup logger

    # load the data
    with open(Path(__file__).parent / "overpass_sample_data.json", "r") as f:
        data = json.load(f)

    nodes, ways = nodes_ways_from_osm(data)
    g = create_street_graph(nodes, ways)
    point_set = get_point_set(g)

    # create the polygon to match
    polygon = sply_geometry.Polygon([(1, 1), (2, 3), (3, 1)])
    polygon = scale_polygon(point_set, polygon)
    polygon = dilate_polygon_relative(polygon)
    polygon = translate_to_start_position(point_set, polygon)

    start = time.perf_counter()
    transformed_polygons_with_subsets = [
        (polygon, subset)
        for _, subset, polygon in get_transformed_polygons_with_subsets(
            polygon, point_set, 5, 5, 5
        )
    ]
    print(time.perf_counter() - start)

    start = time.perf_counter()
    contained_points_sizes = [
        len(list(filter(p.contains, ss))) for p, ss in transformed_polygons_with_subsets
    ]
    print(time.perf_counter() - start)

    start = time.perf_counter()
    transformed_polygons = [
        p for _, p in get_transformed_polygons(polygon, point_set, 5, 5, 5)
    ]
    print(time.perf_counter() - start)

    start = time.perf_counter()
    contained_points_sizes = [
        len(list(filter(p.contains, point_set))) for p in transformed_polygons
    ]
    print(time.perf_counter() - start)

    # fig, ax = plt.subplots()
    # draw_streets(g, ax)
