import time
import random
import networkx as nx
import matplotlib.pyplot as plt

NUMBER_OF_NODES = 250
CONNECTIONS_PER_NODE = 3

node_ids = list(range(1, NUMBER_OF_NODES+1))
edges = []
for node_id in node_ids:
    for _ in range(CONNECTIONS_PER_NODE):
        edges.append((str(node_id), str(node_ids[random.randrange(0, len(node_ids))])))

G = nx.MultiDiGraph()
G.add_edges_from(edges)

color_map = [ "red" if node in [ '1', str(NUMBER_OF_NODES) ] else "green" for node in G ]
plt.figure(figsize=(20, 20))
t1 = time.time()
nx.draw(G, node_color=color_map, with_labels=True, connectionstyle="arc3, rad=0.05")
t2 = time.time()
print(f"Rendering time: {t2-t1:.2f} seconds")

plt.savefig(f"nodes_{NUMBER_OF_NODES}_conn_{CONNECTIONS_PER_NODE}")
# plt.show()