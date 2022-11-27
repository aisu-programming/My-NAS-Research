import time
import random
from graphviz import Digraph

NUMBER_OF_NODES = 150
CONNECTIONS_PER_NODE = 3

node_ids = list(range(1, NUMBER_OF_NODES+1))
edges = []
for node_id in node_ids:
    for _ in range(CONNECTIONS_PER_NODE):
        edges.append((str(node_id), str(node_ids[random.randrange(0, len(node_ids))])))

dot = Digraph()
for node_id in node_ids: dot.node(str(node_id))
dot.edges(edges)

# print(dot.source)

t1 = time.time()
dot.render(f"nodes_{NUMBER_OF_NODES}_conn_{CONNECTIONS_PER_NODE}", view=True, format="png")
t2 = time.time()

print(f"Rendering time: {t2-t1} seconds")