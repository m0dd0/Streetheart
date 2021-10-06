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

from descartes import PolygonPatch


def get_extents(o):
    return (o.bounds[2] - o.bounds[0], o.bounds[3] - o.bounds[1])


line = sply_geometry.LineString([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])

circle = sply_geometry.Point(0, 0).buffer(10)

outer_circle = circle.buffer(1)
inner_circle = circle.buffer(-1)

ring = inner_circle.symmetric_difference(outer_circle)
# print(poly.contains(sply_geometry.Point(0, 0)))
# print(poly.contains(sply_geometry.Point(10, 0)))
# print(poly.contains(sply_geometry.Point(10.5, 0)))
# print(poly.contains(sply_geometry.Point(9.5, 0)))
# print(poly.contains(sply_geometry.Point(12, 0)))

# print(len(ring.boundary.geoms))
boundary_1 = ring.boundary.geoms[0]
boundary_2 = ring.boundary.geoms[1]

# ring_prep = sply_prepared.prep(ring)
# ring_prep = sply_affinity.translate(ring_prep, xoff=1)


# fig, ax = plt.subplots()
# ax.plot(*buff_outer.exterior.xy)
# ax.plot(*buff_inner.exterior.xy)

# plt.show()

print("end")
