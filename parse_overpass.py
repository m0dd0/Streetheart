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
import math
import copy

# pylint:disable=redefined-outer-name


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
    nodes = copy.deepcopy(nodes)
    ways = copy.deepcopy(ways)

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
        lines.append(
            [
                pos_from_node_id(start_node_id, g.nodes),
                pos_from_node_id(end_node_id, g.nodes),
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
    # TODO rewrite with outer bounding box because of rotation
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
    # return sply_geometry.Polygon(poly_outer, holes=poly_inner)


def dilate_polygon_relative(polygon, dilatation_factor=0.1):
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
    polygon, point_set, delta_x, delta_y, delta_phi=10, debug_dict=None
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

    if debug_dict is not None:
        debug_dict["n_checked_points"] = 0

    for y_off in y_offsets:
        # build the subset along the horizontal x-axis
        north_y = point_set.bounds[3] - y_off
        south_y = north_y - grid_size
        # TODO maybe use custom filter is faster
        filter_rect = sply_geometry.Polygon(
            [
                (west_abs, north_y),
                (east_abs, north_y),
                (east_abs, south_y),
                (west_abs, south_y),
            ]
        )
        filter_rect = sply_prepared.prep(filter_rect)
        point_subset_x = sply_geometry.MultiPoint(
            list(filter(filter_rect.contains, point_set))
        )
        if debug_dict is not None:
            debug_dict["n_checked_points"] += len(point_subset_x)

        for x_off in x_offsets:
            # build the grid subset in which the polygon rotates
            west_x = west_abs + x_off
            east_x = west_abs + grid_size
            # TODO maybe use custom filter is faster
            filter_rect = sply_geometry.Polygon(
                [
                    (west_x, north_y),
                    (east_x, north_y),
                    (east_x, south_y),
                    (west_x, south_y),
                ]
            )
            filter_rect = sply_prepared.prep(filter_rect)
            point_subset_xy = sply_geometry.MultiPoint(
                list(filter(filter_rect.contains, point_subset_x))
            )
            if debug_dict is not None:
                debug_dict["n_checked_points"] += len(point_subset_x)

            for phi_off in phi_offsets:
                i += 1
                progress = i / n_yields
                # TODO do transforms stepwise
                transformed_polygon = sply_affinity.rotate(
                    sply_affinity.translate(polygon, xoff=x_off, yoff=-y_off),
                    phi_off,
                )
                yield progress, point_subset_xy, transformed_polygon


def draw_dilated_polygon(polygon, ax, **kwargs):
    border_1, border_2 = get_dilated_polygon_borders(polygon)
    border_1_mpl = ax.plot(*border_1.xy, **kwargs)[0]
    border_2_mpl = ax.plot(*border_2.xy, **kwargs)[0]

    return border_1_mpl, border_2_mpl


def draw_polygon(polygon, ax, **kwargs):
    return ax.plot(*polygon.exterior.xy, **kwargs)


def draw_streets(g, ax, **kwargs):
    ax.add_collection(mpl.collections.LineCollection(get_edge_lines(g), **kwargs))
    ax.autoscale()
    ax.axis("equal")


def draw_point_set(point_set, ax, **kwargs):
    return ax.scatter(*zip(*[(p.x, p.y) for p in point_set]), **kwargs)


def pos_from_node_id(node_id, nodes):
    node = nodes[node_id]
    return (node["lon"], node["lat"])


def resample_street_data(nodes, ways, min_dist, max_dist, dist_func="euclidean"):
    # TODO try shapely interpolate

    if dist_func == "euclidean":
        # TODO use numpy
        dist_func = lambda a, b: ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    elif isinstance(dist_func, str):
        raise ValueError(f"'{dist_func}' is no supported distance metric.")

    nodes = copy.deepcopy(nodes)
    ways = copy.deepcopy(ways)

    resampled_ways = {}
    resampled_nodes = {}

    for way_id, way_attrs in ways.items():
        way_node_ids = way_attrs.pop("nodes")

        current_node_id = way_node_ids.pop(0)
        current_node_pos = pos_from_node_id(current_node_id, nodes)

        resampled_way_node_ids = [current_node_id]

        while len(way_node_ids) > 1:
            next_node_id = way_node_ids.pop(0)
            next_node_pos = pos_from_node_id(next_node_id, nodes)

            # TODO insert point if distance to next point is to long
            # if dist_func(next_node, current_node) > max_dist:
            #     pass

            if dist_func(next_node_pos, current_node_pos) > min_dist:
                resampled_way_node_ids.append(next_node_id)
                current_node_id = next_node_id
                current_node_pos = next_node_pos
        resampled_way_node_ids.append(way_node_ids.pop(0))
        assert len(way_node_ids) == 0

        resampled_ways[way_id] = {**way_attrs, **{"nodes": resampled_way_node_ids}}

        for n_id in resampled_way_node_ids:
            resampled_nodes[n_id] = nodes[n_id]

    return resampled_nodes, resampled_ways


def get_points_in_rectangle(point_set, min_x, min_y, max_x, max_y):
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


def divide_square(square):
    cx, cy = square.centroid.x, square.centroid.y
    d = max(get_extents(square)) * 0.5

    p1 = sply_geometry.Polygon([(cx - d, cy + d), (cx, cy + d), (cx, cy), (cx - d, cy)])
    p2 = sply_geometry.Polygon([(cx, cy + d), (cx + d, cy + d), (cx + d, cy), (cx, cy)])
    p3 = sply_geometry.Polygon([(cx, cy), (cx + d, cy), (cx + d, cy - d), (cx, cy - d)])
    p4 = sply_geometry.Polygon([(cx - d, cy), (cx, cy), (cx, cy - d), (cx - d, cy - d)])

    return (p1, p2, p3, p4)


def squarify_polygon(polygon, min_quad_size):
    def quadtree(square):
        if max(get_extents(square)) < min_quad_size:
            return []

        # rect fully inside --> return square and stop recursion
        if polygon.contains(square):
            return [square]

        # divide current square and collect results from subdivision recursively
        else:
            squares = []
            for sub_square in divide_square(square):
                squares.extend(quadtree(sub_square))
            return squares

    # TODO fix wrong bounding box
    cx, cy = polygon.centroid.x, polygon.centroid.y
    d = max(get_extents(polygon)) * 0.5  # + 2
    bounding_square = sply_geometry.Polygon(
        [(cx - d, cy), (cx + d, cy + d), (cx + d, cy - d), (cx - d, cy - d)]
    )

    fully_contained_squares = quadtree(bounding_square)

    return fully_contained_squares


if __name__ == "__main__":
    # TODO setup logger

    # fig, ax = plt.subplots()
    # square = sply_geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    # draw_polygon(square, ax)
    # # draw_polygon(divide_square(square)[0], ax)
    # for sub_square in divide_square(square):
    #     draw_polygon(sub_square, ax)
    # plt.show()

    polygon = sply_geometry.Polygon([(1, 1), (2, 3), (3, 1)])
    polygon = dilate_polygon_absolute(polygon, 0.5)

    # polygon_squares = squarify_polygon(polygon, 0.03)

    # fig, ax = plt.subplots()
    # draw_dilated_polygon(polygon, ax)
    # for contained_square in polygon_squares:
    #     draw_polygon(contained_square, ax)
    # ax.autoscale()
    # ax.axis("equal")
    # plt.show()

    # load the data
    # with open(Path(__file__).parent / "overpass_sample_data.json", "r") as f:
    #     data = json.load(f)

    # nodes, ways = nodes_ways_from_osm(data)
    # g_orig = create_street_graph(nodes, ways)
    # point_set_orig = get_point_set(g_orig)

    # start = time.perf_counter()
    # point_Set_orig_list = [(p.x, p.y) for p in point_set_orig]
    # print(time.perf_counter() - start)

    rectangle = sply_geometry.Polygon([(5, 5), (5, 6), (6, 6), (6, 5)])
    points_np = np.random.rand(10_000, 2) * 10
    points_sply = sply_geometry.MultiPoint(points_np)

    start = time.perf_counter()
    contained_points_sply = list(filter(rectangle.contains, points_sply))
    print(time.perf_counter() - start)
    print(len(contained_points_sply))

    start = time.perf_counter()
    contained_points_np = get_points_in_rectangle(points_np, *rectangle.bounds)
    print(time.perf_counter() - start)
    print(len(contained_points_np))

    # min_dist = min(get_extents(point_set_orig)) / 30
    # g_resampled = create_street_graph(
    #     *resample_street_data(nodes, ways, min_dist, math.inf)
    # )
    # point_set_resampled = get_point_set(g_resampled)

    # fig, ax = plt.subplots()
    # draw_streets(g_orig, ax, colors="green")
    # draw_point_set(point_set_orig, ax, s=10, c="green")
    # draw_streets(g_resampled, ax, colors="red")
    # draw_point_set(point_set_resampled, ax, s=10, c="red")
    # plt.show()
    # print(len(nodes))

    # for way_length_fraction in range(5, 50, 5):
    #     min_dist = min(get_extents(point_set_orig)) / way_length_fraction
    #     resampled_nodes, resampled_ways = resample_street_data(
    #         nodes, ways, min_dist, math.inf
    #     )
    #     print(way_length_fraction, len(resampled_nodes))

    # point_set = get_point_set(g)

    # create the polygon to match
    # polygon = sply_geometry.Polygon([(1, 1), (2, 3), (3, 1)])
    # polygon = scale_polygon(point_set, polygon)
    # polygon = dilate_polygon_relative(polygon)
    # polygon = translate_to_start_position(point_set, polygon)

    # start = time.perf_counter()
    # transformed_polygons_with_subsets = [
    #     (polygon, subset)
    #     for _, subset, polygon in get_transformed_polygons_with_subsets(
    #         polygon, point_set, 10, 10, 10
    #     )
    # ]
    # print(time.perf_counter() - start)

    # start = time.perf_counter()
    # contained_points_sets = [
    #     list(filter(sply_prepared.prep(p).contains, ss))
    #     for p, ss in transformed_polygons_with_subsets
    # ]
    # print(time.perf_counter() - start)

    # start = time.perf_counter()
    # transformed_polygons = [
    #     p for _, p in get_transformed_polygons(polygon, point_set, 10, 10, 10)
    # ]
    # print(time.perf_counter() - start)

    # start = time.perf_counter()
    # contained_points_sizes = [
    #     len(list(filter(sply_prepared.prep(p).contains, point_set)))
    #     for p in transformed_polygons
    # ]
    # print(time.perf_counter() - start)

    # i_show = np.argsort([len(point_set) for point_set in contained_points_sets])[-10:]

    # fig, ax = plt.subplots()
    # draw_streets(g, ax)
    # for i in i_show:
    #     draw_dilated_polygon(transformed_polygons_with_subsets[i][0], ax)
    #     draw_point_set(contained_points_sets[i], ax)
    # plt.show()

    # fig, ax = plt.subplots()
    # draw_streets(g, ax)
    # draw_point_set(point_set, ax)
    # plt.show()
