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

ring = sply_geometry.Point(0, 0).buffer(10)

buff_outer = ring.buffer(1)
buff_inner = ring.buffer(-1)

# print(buff_inner.contains(sply_geometry.Point(0, 0)))
# print(buff_inner.contains(sply_geometry.Point(10, 0)))
# print(buff_outer.contains(sply_geometry.Point(10, 0)))
# print(buff_outer.contains(sply_geometry.Point(0, 0)))
# print(buff_outer.contains(sply_geometry.Point(10.5, 0)))
# print(buff_outer.contains(sply_geometry.Point(12, 0)))


poly = buff_inner.symmetric_difference(buff_outer)
print(poly.contains(sply_geometry.Point(0, 0)))
print(poly.contains(sply_geometry.Point(10, 0)))
print(poly.contains(sply_geometry.Point(10.5, 0)))
print(poly.contains(sply_geometry.Point(9.5, 0)))
print(poly.contains(sply_geometry.Point(12, 0)))

# fig, ax = plt.subplots()
# ax.plot(*buff_outer.exterior.xy)
# ax.plot(*buff_inner.exterior.xy)

# plt.show()

print("end")
