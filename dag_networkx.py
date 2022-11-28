''' Libraries '''
import sys
import time
import math
import random
import networkx as nx
import matplotlib.pyplot as plt


''' Functions '''
def plot_DAG(graph_data, filename=None, test=False):

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

    # sources  = [ node for node,  in_degree in G.in_degree()  if  in_degree == 0 ]  # type: ignore
    # sinks    = [ node for node, out_degree in G.out_degree() if out_degree == 0 ]  # type: ignore
    # isolated = list(filter(lambda node: node in sinks, sources))
    # for node in isolated:
    #     sources.remove(node)
    #     sinks.remove(node)

    # all_removed_sources = []
    # while sources != [ 0 ]:
    #     remove_sources = sources.copy()
    #     remove_sources.remove(0)
    #     G.remove_nodes_from(remove_sources)
    #     all_removed_sources += remove_sources
    #     sources  = [ node for node, in_degree in G.in_degree()  if  in_degree == 0 ]  # type: ignore
    # G.add_nodes_from(all_removed_sources)

    # all_removed_sinks = []
    # while sinks != [ node_num-1 ]:
    #     remove_sinks = sinks.copy()
    #     remove_sinks.remove(node_num-1)
    #     G.remove_nodes_from(remove_sinks)
    #     all_removed_sinks += remove_sinks
    #     sinks = [ node for node, out_degree in G.out_degree() if out_degree == 0 ]  # type: ignore
    # G.add_nodes_from(all_removed_sinks)

    color_map = []
    for node_id in G:
        if   node_id in source_ids        : color_map.append("lime")
        elif node_id in removed_source_ids: color_map.append("green")
        elif node_id in sink_ids          : color_map.append("cyan")
        elif node_id in removed_sink_ids  : color_map.append("blue")
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
    edge_colors = [ str((1-ec)*(4/5)) for ec in edge_colors ]

    plt.figure(figsize=(10, 10))
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

    filename = "test" if test else ("DAG" if filename is None else filename)
    plt.savefig(filename)
    # plt.show()
    plt.close()
    return


''' Execution '''
if __name__ == "__main__":

    NUMBER_OF_NODES = 64
    CONNECTIONS_PER_NODE = 3

    node_ids = list(range(1, NUMBER_OF_NODES+1))
    edges = []
    for node_id in node_ids:
        for _ in range(CONNECTIONS_PER_NODE):
            edges.append((node_id, node_ids[random.randrange(0, len(node_ids))]))

    plot_DAG(edges, [], node_ids[0], node_ids[-1], test=True)