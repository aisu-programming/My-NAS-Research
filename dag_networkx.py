''' Libraries '''
import time
import random
import networkx as nx
import matplotlib.pyplot as plt


''' Functions '''
def plot_DAG(edges, isolated_nodes, head_node, tail_node, test=False):

    G = nx.MultiDiGraph()
    G.add_edges_from(edges)
    G.add_nodes_from(isolated_nodes)

    color_map = [ "red" if node in [ head_node, tail_node ] else "green" for node in G ]
    plt.figure(figsize=(20, 20))
    if test: t1 = time.time()
    nx.draw(G, node_color=color_map, with_labels=True, connectionstyle="arc3, rad=0.05")
    if test:
        t2 = time.time()
        print(f"Rendering time: {t2-t1:.2f} seconds")

    filename = "test" if test else "DAG"
    plt.savefig(filename)
    # plt.show()
    return


''' Execution '''
if __name__ == "__main__":

    NUMBER_OF_NODES = 250
    CONNECTIONS_PER_NODE = 3

    node_ids = list(range(1, NUMBER_OF_NODES+1))
    edges = []
    for node_id in node_ids:
        for _ in range(CONNECTIONS_PER_NODE):
            edges.append((node_id, node_ids[random.randrange(0, len(node_ids))]))

    plot_DAG(edges, [], node_ids[0], node_ids[-1], test=True)