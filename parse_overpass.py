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


def get_point_set(g):
    """get all points

    Args:
        g ([type]): [description]

    Returns:
        [type]: [description]
    """
    points = [[n_attrs["lon"], n_attrs["lat"]] for _, n_attrs in g.nodes(data=True)]
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


# def check_bounds(polygon, point_set, axis="maxx"):
#     if axis == "maxx":
#         return polygon.bounds[2] < point_set.bounds[2]
#     elif axis == "miny":
#         return polygon.bounds[1] > point_set.bounds[1]
#     else:
#         raise NotImplementedError("this axis is not implemented")


def transformed_polygons(
    polygon,
    point_set,
    delta_x_factor=0.1,
    delta_y_factor=0.1,
    delta_phi=5,
):
    # TODO rotate

    polygon_extents = get_extents(polygon)
    x_delta = polygon_extents[0] * delta_x_factor
    y_delta = polygon_extents[1] * delta_y_factor
    point_extents = get_extents(point_set)
    x_deltas = np.arange(0, point_extents[0], x_delta)
    y_deltas = np.arange(0, point_extents[1], y_delta)

    # polygon_y = sply_geometry.Polygon(polygon)
    for y_delta in y_deltas:
        for x_delta in x_deltas:
            yield sply_affinity.translate(polygon, xoff=x_delta, yoff=-y_delta)
        # polygon_y = sply_affinity.translate(polygon, yoff=-y_delta)

    # if method == "while":
    #     while check_bounds(polygon, point_set, "miny"):
    #         x_off_acc = 0
    #         while check_bounds(polygon, point_set, "maxx"):
    #             x_off_acc += x_delta
    #             polygon = sply_affinity.translate(polygon, xoff=x_delta)
    #             yield polygon
    #         polygon = sply_affinity.translate(polygon, yoff=-y_delta, xoff=-x_off_acc)


if __name__ == "__main__":
    # TODo setup logger

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

    fig, ax = plt.subplots()
    ax.add_collection(mpl.collections.LineCollection(get_edge_lines(g)))
    ax.autoscale()
    ax.axis("equal")
    plt.ion()
    for polygon_transformed in transformed_polygons(polygon, point_set):
        border_1, border_2 = get_dilated_polygon_borders(polygon)
        border_1_mpl = ax.plot(*border_1.xy, color="red")[0]
        border_2_mpl = ax.plot(*border_2.xy, color="red")[0]

        fig.canvas.draw()
        fig.canvas.flush_events()

        border_1_mpl.remove()
        border_2_mpl.remove()

#         prepared_polygon = sply_prepared.prep(transformed_polygon)
#         contained_points = [
#             (p.x, p.y) for p in filter(prepared_polygon.contains, point_set)
#         ]
#         contained_points_mpl = ax.scatter(*zip(*contained_points), color="green")
