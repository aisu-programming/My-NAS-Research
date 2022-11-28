''' Libraries '''
import time
import random
import networkx as nx
import matplotlib.pyplot as plt


''' Functions '''
def plot_DAG(edges, number_of_nodes, filename=None, test=False):

    G = nx.MultiDiGraph()
    G.add_nodes_from(list(range(number_of_nodes)))
    G.add_edges_from(edges)

    sources  = [ node for node,  in_degree in G.in_degree()  if  in_degree == 0 ]  # type: ignore
    sinks    = [ node for node, out_degree in G.out_degree() if out_degree == 0 ]  # type: ignore
    isolated = list(filter(lambda node: node in sinks, sources))
    for node in isolated:
        sources.remove(node)
        sinks.remove(node)

    color_map = [
        "yellow" if node in sources else
            "cyan" if node in sinks else
                "white" if node in isolated else "green" for node in G ]
    plt.figure(figsize=(10, 10))
    t1 = time.time()
    nx.draw(G,
        pos=nx.kamada_kawai_layout(G, scale=2),
        # pos=nx.circular_layout(G),
        # pos=nx.spring_layout(G, scale=2),
        node_color=color_map,
        edgecolors="black",
        with_labels=True,
        connectionstyle="arc3, rad=0.3"
    )
    t2 = time.time()
    print(f"Rendering time: {t2-t1:.2f} seconds")

    filename = "test" if test else ("DAG" if filename is None else filename)
    plt.savefig(filename)
    # plt.show()
    plt.close()
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