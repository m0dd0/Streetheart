from pathlib import Path
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

nodes = {}
ways = {}

with open(Path(__file__).parent / "overpass_sample_data.json", "r") as f:
    data = json.load(f)

for elem in data["elements"]:
    elem_id = elem.pop("id")
    elem_type = elem.pop("type")
    {"node": nodes, "way": ways}[elem_type][elem_id] = elem

g = nx.Graph()

g.add_nodes_from(nodes.items())

for way in ways.values():
    way_nodes = way.pop("nodes")
    edges = list(zip(way_nodes[1:], way_nodes[:-1]))
    g.add_edges_from(edges, **way)

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

fig, ax = plt.subplots()
ax.add_collection(mpl.collections.LineCollection(lines))
ax.autoscale()
ax.axis("equal")

plt.show()
