''' Libraries '''
import torch
import torch.nn as nn
from tqdm import tqdm


''' Functions '''
def cycle_check(edges, input_amounts):
    input_amounts_copy = input_amounts.copy()
    while 0 in input_amounts_copy:
        source_id = input_amounts_copy.index(0)
        edges_from_source = list(filter(lambda e: e[0]==source_id, edges))
        for edge in edges_from_source: input_amounts_copy[edge[1]] -= 1
        input_amounts_copy[source_id] = -1
    if input_amounts_copy == [-1]*len(input_amounts_copy): return True
    else: return False


class Model(nn.Module):
    def __init__(self, node_num, max_depth, threshold_of_connection,
                 max_number_of_connection, min_number_of_connection):
        super(Model, self).__init__()
        self.node_num = node_num
        self.nodes : list[Node] = [ 
            Node(node_id, node_num, max_depth, threshold_of_connection,
                 max_number_of_connection, min_number_of_connection)
            for node_id in range(node_num) ]
        self.alphas = torch.rand(node_num*node_num)
        for i in range(node_num):
            self.alphas[i*node_num+i] = 0  # prevent node from linking to itself
            self.alphas[i*node_num] = 0  # Make node 0 source
            self.alphas[node_num*(node_num-1)+i] = 0  # Make node 0 source
        self.t_c     = threshold_of_connection
        self.max_n_c = max_number_of_connection
        self.min_n_c = min_number_of_connection
    
    def search_path(self):
        edges:list[tuple[int, int]] = []
        input_amounts:list[int] = [ -1 ] * self.node_num
        alphas_tmp = list(self.alphas.numpy())
        pbar = tqdm(desc="Backward searching forwarding path")
        while True:
            max_alpha = max(alphas_tmp)
            if max_alpha == 0: break
            max_alpha_id = alphas_tmp.index(max_alpha)
            from_node_id = max_alpha_id // self.node_num
            to_node_id   = max_alpha_id  % self.node_num
            connection_limit = self.max_n_c if max_alpha > self.t_c else self.min_n_c

            # Limitation to output
            edges_to_node = list(filter(lambda e: e[1]==to_node_id, edges))
            if len(edges_to_node) >= connection_limit:
                for from_node_id in range(self.node_num):
                    alphas_tmp[from_node_id*self.node_num+to_node_id] = 0
                continue
            # Limitation to input
            edges_from_node = list(filter(lambda e: e[0]==from_node_id, edges))
            if len(edges_from_node) >= connection_limit:
                for to_node_id in range(self.node_num):
                    alphas_tmp[from_node_id*self.node_num+to_node_id] = 0
                continue
            # Only limit to output --> create multiple sinks but only one source
            # Only limit to input --> create multiple sources but only one sink
            # Both --> create multiple sources and multiple sinks
            # Do nothing --> create only one source and only one sink

            edges_tmp, input_amounts_tmp = edges.copy(), input_amounts.copy()
            edges_tmp.append((from_node_id, to_node_id, max_alpha))
            if input_amounts_tmp[from_node_id] == -1: input_amounts_tmp[from_node_id] = 0
            input_amounts_tmp[to_node_id] = input_amounts_tmp[to_node_id] + 1 if input_amounts_tmp[to_node_id] != -1 else 1
            if cycle_check(edges_tmp, input_amounts_tmp):
                edges         = edges_tmp
                input_amounts = input_amounts_tmp
                alphas_tmp[max_alpha_id] = 0
                alphas_tmp[to_node_id*self.node_num+from_node_id] = 0
            else:
                alphas_tmp[max_alpha_id] = 0
            pbar.update(1)
        pbar.close()
        self.edges:list[int] = edges
        self.isolated_node_ids = [ node_id for node_id in range(self.node_num) if node_id not in self.from_node_ids and node_id not in self.to_node_ids ]

    @property
    def from_node_ids(self) -> list[int]:
        return list(set([ edge[0] for edge in self.edges ]))

    @property
    def to_node_ids(self) -> list[int]:
        return list(set([ edge[1] for edge in self.edges ]))

    @property
    def source_ids(self) -> list[int]:
        return list(set([ node_id for node_id in self.from_node_ids
                          if node_id not in self.to_node_ids ]))

    @property
    def sink_ids(self) -> list[int]:
        return list(set([ node_id for node_id in self.to_node_ids
                          if node_id not in self.from_node_ids ]))

    def prune_path(self):

        self.removed_source_ids:list[int] = []
        while self.source_ids != [ 0 ]:
            for source_id in self.source_ids:
                if source_id == 0: continue
                self.edges = list(filter(lambda e: e[0]!=source_id, self.edges))
                self.removed_source_ids.append(source_id)

        self.removed_sink_ids:list[int] = []
        while self.sink_ids != [ self.node_num-1 ]:
            for sink_id in self.sink_ids:
                if sink_id == self.node_num-1: continue
                self.edges = list(filter(lambda e: e[1]!=sink_id, self.edges))
                self.removed_sink_ids.append(sink_id)

    def forward(self, x):
        # for node_id in self.forward_queue:
        #     node_id
        return x


class Node(nn.Module):
    def __init__(self, node_num, max_depth):
        super(Node, self).__init__()
        self.linear = nn.Linear((node_num-1)*max_depth, max_depth)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        self.output = self.activation(self.linear(x))


''' Execution '''
if __name__ == "__main__":

    NODE_NUMBER = 32
    NODE_DEPTH  = 4

    model = Model(NODE_NUMBER, NODE_DEPTH, 0.75, 6, 2)
    model.search_path()
    from dag_networkx import plot_DAG
    plot_DAG(graph_data={
        "node_num"          : model.node_num,
        "edges"             : model.edges,
        "isolated_node_ids" : model.isolated_node_ids,
        "source_ids"        : model.source_ids,
        "removed_source_ids": [],
        "sink_ids"          : model.sink_ids,
        "removed_sink_ids"  : [],
    }, filename="DAG_before_prune")
    model.prune_path()
    plot_DAG(graph_data={
        "node_num"          : model.node_num,
        "edges"             : model.edges,
        "isolated_node_ids" : model.isolated_node_ids,
        "source_ids"        : model.source_ids,
        "removed_source_ids": model.removed_source_ids,
        "sink_ids"          : model.sink_ids,
        "removed_sink_ids"  : model.removed_sink_ids,
    }, filename="DAG_after_prune")