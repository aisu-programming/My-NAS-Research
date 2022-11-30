""" Libraries """
import sys
import time
import math
import random
import networkx as nx


""" Functions """
def plot_DAG(graph_data):

    node_num           = graph_data["node_num"]
    edges              = graph_data["edges"]
    isolated_node_ids  = graph_data["isolated_node_ids"]
    source_ids         = graph_data["source_ids"]
    removed_source_ids = graph_data["removed_source_ids"]
    sink_ids           = graph_data["sink_ids"]
    removed_sink_ids   = graph_data["removed_sink_ids"]

    G = nx.MultiDiGraph()
    G.add_nodes_from(list(range(node_num)))
    for edge in edges: G.add_edge(edge[0], edge[1], weight=edge[2])

    color_map = []
    for node_id in G:
        if   node_id in source_ids        : color_map.append("greenyellow")
        elif node_id in removed_source_ids: color_map.append("limegreen")
        elif node_id in sink_ids          : color_map.append("aqua")
        elif node_id in removed_sink_ids  : color_map.append("dodgerblue")
        elif node_id in isolated_node_ids : color_map.append("red")
        else                              : color_map.append("white")

    node_positions = {}
    for node_id in G:
        position_tuple = [math.sin(node_id/node_num*2*math.pi),
                          math.cos(node_id/node_num*-2*math.pi)]
        if node_id in removed_source_ids+removed_sink_ids+isolated_node_ids:
            position_tuple[0] *= 1.1
            position_tuple[1] *= 1.1
        node_positions[node_id] = position_tuple

    edge_weights = list(nx.get_edge_attributes(G, "weight").values())
    ews = edge_weights
    edge_colors = [ (ew-min(ews))/(max(ews)-min(ews)) for ew in ews ]
    white_limit = 0.7
    edge_colors = [ str((1-ec)*white_limit) for ec in edge_colors ]

    t1 = time.time()
    print("Rendering... ", end='')
    sys.stdout.flush()
    nx.draw(G,
        node_color=color_map,
        pos=node_positions,
        edge_color=edge_colors,
        # width=2,
        edgecolors="black",
        with_labels=True,
        connectionstyle="arc3, rad=0.3"
    )
    t2 = time.time()
    print(f"cost time: {t2-t1:.2f} seconds")

    return


""" Execution """
if __name__ == "__main__":

    NUMBER_OF_NODES = 64
    CONNECTIONS_PER_NODE = 3

    node_ids = list(range(1, NUMBER_OF_NODES+1))
    edges = []
    for node_id in node_ids:
        for _ in range(CONNECTIONS_PER_NODE):
            edges.append((node_id, node_ids[random.randrange(0, len(node_ids))]))

    raise NotImplementedError
    # plot_DAG(edges, [], node_ids[0], node_ids[-1], test=True)