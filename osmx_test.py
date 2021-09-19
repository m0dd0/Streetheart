import osmnx as ox
import pandas

G = ox.graph_from_place(
    "Piedmont, California, USA",
    network_type="drive",
)

# print(list(G.nodes))
print(list(G.nodes(data=True))[0])
# print(type(G))
# fig, ax = ox.plot_graph(G)
