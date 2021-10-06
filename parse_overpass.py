from pathlib import Path
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from shapely import geometry as sply_geometry
from shapely import affinity as sply_affinity
from shapely import prepared as sply_prepared
import logging
import time

# setup logger
# TODO


# load the data
with open(Path(__file__).parent / "overpass_sample_data.json", "r") as f:
    data = json.load(f)


# sort elemts by type and map their attributes to their id
nodes = {}
ways = {}
for elem in data["elements"]:
    elem_id = elem.pop("id")
    elem_type = elem.pop("type")
    {"node": nodes, "way": ways}[elem_type][elem_id] = elem


# create graph which conatins all the data
g = nx.Graph()
g.add_nodes_from(nodes.items())
for way_id, way_attrs in ways.items():
    way_nodes = way_attrs.pop("nodes")
    edges = list(zip(way_nodes[1:], way_nodes[:-1]))
    g.add_edges_from(edges, **way_attrs, way_id=way_id)


# create a list of edges by their coordinate
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


# plot the list of coordinates representing all edges
plt.ion()
fig, ax = plt.subplots()
ax.add_collection(mpl.collections.LineCollection(lines))
ax.autoscale()
ax.axis("equal")


# get all points
points = [[n_attrs["lon"], n_attrs["lat"]] for _, n_attrs in g.nodes(data=True)]
points = sply_geometry.MultiPoint(points)


# create the polygon to match
poly = sply_geometry.Polygon([(1, 1), (2, 3), (3, 1)])


# helper functions to get the x,y-extents og an geometry object
def get_extents(o):
    return (o.bounds[2] - o.bounds[0], o.bounds[3] - o.bounds[1])


# scale and translate the polygon to fir in the upper left corner of the point set
SIZE_FACTOR = 0.3
points_extents = get_extents(points)
poly_extents = get_extents(poly)

if (points_extents[0] / points_extents[1]) < (poly_extents[0] / poly_extents[1]):
    scaling_axis = 0
else:
    scaling_axis = 1
scaling_factor = points_extents[scaling_axis] * SIZE_FACTOR / poly_extents[scaling_axis]

poly = sply_affinity.scale(
    poly, xfact=scaling_factor, yfact=scaling_factor, origin="center"
)
poly = sply_affinity.translate(
    poly, xoff=points.bounds[0] - poly.bounds[0], yoff=points.bounds[3] - poly.bounds[3]
)


# create the polygon to match with tolerance
MATCH_TOLERANCE = 0.05
poly_extents = get_extents(poly)
diagonal_length = (poly_extents[0] ** 2 + poly_extents[1] ** 2) ** 0.5
buffer_value = MATCH_TOLERANCE * diagonal_length
poly_outer = poly.buffer(buffer_value)
poly_inner = poly.buffer(-buffer_value)
poly_tolerance = poly_inner.symmetric_difference(poly_outer)


DELTA_X_FACTOR = 0.1
DELTA_Y_FACTOR = 0.1
DELTA_ROTATE = 5

x_delta = poly_extents[0] * DELTA_X_FACTOR
y_delta = poly_extents[1] * DELTA_Y_FACTOR
rotate_delta = DELTA_ROTATE

# b1 = ax.plot(*poly_tolerance.boundary.geoms[0].xy, color="red")[0]
# b2 = ax.plot(*poly_tolerance.boundary.geoms[1].xy, color="red")[0]
# b1.remove()
# b2.remove()
# b1 = ax.plot(*poly_tolerance.boundary.geoms[0].xy, color="green")[0]
# b2 = ax.plot(*poly_tolerance.boundary.geoms[1].xy, color="green")[0]


# 1=miny: as long as miny border of polygon is higher than miny border of poin set
while poly_tolerance.bounds[1] > points.bounds[1]:
    x_off_acc = 0
    # 2=maxx: as long as maxx border of polygon is scmaller than maxx border of point set
    while poly_tolerance.bounds[2] < points.bounds[2]:
        # it is not possible to transform a prepared polygon so it must be rebuid here
        poly_tolerance_prep = sply_prepared.prep(poly_tolerance)
        # TODO rotate
        #### MATCHING
        # contained_points = [
        #     (p.x, p.y) for p in filter(poly_tolerance_prep.contains, points)
        # ]
        b1 = ax.plot(*poly_tolerance.boundary.geoms[0].xy, color="red")[0]
        b2 = ax.plot(*poly_tolerance.boundary.geoms[1].xy, color="red")[0]

        # ax.scatter(*zip(*contained_points), color="green")
        fig.canvas.draw()
        fig.canvas.flush_events()
        # time.sleep(1)
        b1.remove()
        b2.remove()

        # move to the right
        poly_tolerance = sply_affinity.translate(poly_tolerance, xoff=x_delta)
        x_off_acc += x_delta
    # move down and reset x offset
    poly_tolerance = sply_affinity.translate(
        poly_tolerance, yoff=-y_delta, xoff=-x_off_acc
    )
# x_steps = list(range(0, point_extents[0], poly_extents[0] * DELTA_X_FACTOR))
# y_steps = list(range(0, point_extents[1], poly_extents[1] * DELTA_Y_FACTOR))
# rotate_steps = list(range(0, 359, DELTA_ROTATE))
# for x_off in x_steps:
#     for y_off in y_steps:
#         for phi in rotate_steps:
#             pass

# get the contained points in the tolerance area


# plot it
# ax.plot(*poly_outer.exterior.xy, color="red")
# ax.plot(*poly_inner.exterior.xy, color="red")
# ax.scatter(*zip(*contained_points), color="green")
# ax.scatter(*zip(*[(p.x, p.y) for p in points]), color="green")
# plt.show()