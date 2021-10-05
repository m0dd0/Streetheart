from pathlib import Path
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from shapely import geometry as sply_geometry
from shapely import affinity as sply_affinity
from shapely import prepared as sply_prepared
import logging
import numpy as np

# logging.set_logg

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
fig, ax = plt.subplots()
ax.add_collection(mpl.collections.LineCollection(lines))
ax.autoscale()
ax.axis("equal")
# plt.show()


def get_extents(o):
    return (o.bounds[2] - o.bounds[0], o.bounds[3] - o.bounds[1])


# get all points
points = [[n_attrs["lon"], n_attrs["lat"]] for _, n_attrs in g.nodes(data=True)]
# points = np.array(points)
points = sply_geometry.MultiPoint(points)
points_extents = get_extents(points)

poly = sply_geometry.Polygon([(1, 1), (2, 3), (3, 1)])
poly_extents = get_extents(poly)

SIZE_FACTOR = 0.3
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

ax.plot(*poly.exterior.xy, color="red")

poly_prep = sply_prepared.prep(poly)

contained_points = [(p.x, p.y) for p in filter(poly_prep.contains, points)]
# print(contained_points[0])
ax.scatter(*zip(*contained_points), color="green")

plt.show()