''' Libraries '''
import torch
import torch.nn as nn
from tqdm import tqdm
from dag_networkx import plot_DAG as imported_plot_DAG


''' Functions '''
def cycle_check(edges, input_amounts):
    # Run topological sort -> Success: no cycle / Failure: cycle exists
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

        assert max_number_of_connection >= 1, "max_number_of_connection should >= 1"
        assert min_number_of_connection >= 1, "min_number_of_connection should >= 1"

        self.node_num = node_num
        self.nodes : list[Node] = [ Node(node_num, max_depth)
                                    for _ in range(node_num) ]
        self.alphas = torch.rand(node_num*node_num)
        for i in range(node_num):
            self.alphas[i*node_num + i] = 0  # prevent node from linking to itself
            self.alphas[i*node_num + 0] = 0  # Make first node (node_id=0) source
            self.alphas[node_num*(node_num-1)+i] = 0  # Make last node (node_id=node_num-1) sink
        self.t_c     = threshold_of_connection
        self.max_n_c = max_number_of_connection
        self.min_n_c = min_number_of_connection

    def search_path(self, plot_dag=False):
        self.link_edges()
        if plot_dag: self.plot_DAG()
        try:
            self.prune_edges()
            if plot_dag: self.plot_DAG()
        except:
            self.plot_DAG()
            raise Exception
    
    def link_edges(self):
        self.edges:list[tuple[int, int]] = []
        self.input_amounts:list[int] = [ -1 ] * self.node_num
        alphas_tmp = list(self.alphas.numpy())
        pbar = tqdm(desc="Backward searching forwarding path")
        while True:
            max_alpha = max(alphas_tmp)
            if max_alpha == 0: break
            max_alpha_id = alphas_tmp.index(max_alpha)
            from_node_id = max_alpha_id // self.node_num
            to_node_id   = max_alpha_id  % self.node_num

            
            # Limitation
            connection_limit = self.max_n_c if max_alpha > self.t_c else self.min_n_c
            # Limitation to inputs of "to_node" (node_id=to_node_id)
            edges_to_node = list(filter(lambda e: e[1]==to_node_id, self.edges))
            continue_or_not = False
            if len(edges_to_node) >= connection_limit:
                # Disable new link to this to_node (node_id=to_node_id)
                for from_node_id in range(self.node_num):
                    alphas_tmp[from_node_id*self.node_num+to_node_id] = 0
                continue_or_not = True
            # Limitation to outputs of "from_node" (node_id=from_node_id)
            edges_from_node = list(filter(lambda e: e[0]==from_node_id, self.edges))
            if len(edges_from_node) >= connection_limit:
                # Disable new link from this from_node (node_id=from_node_id)
                for to_node_id in range(self.node_num):
                    alphas_tmp[from_node_id*self.node_num+to_node_id] = 0
                continue_or_not = True
            # Only limit to output --> create multiple sinks but only one source
            # Only limit to input --> create multiple sources but only one sink
            # Both --> create multiple sources and multiple sinks
            # Do nothing --> create only one source and only one sink
            if continue_or_not: continue

            
            # Make a copy for with new edge, and do cycle check
            edges_tmp, input_amounts_tmp = self.edges.copy(), self.input_amounts.copy()
            edges_tmp.append((from_node_id, to_node_id, max_alpha))
            if input_amounts_tmp[from_node_id] == -1: input_amounts_tmp[from_node_id] = 0
            input_amounts_tmp[to_node_id] = input_amounts_tmp[to_node_id] + 1 if input_amounts_tmp[to_node_id] != -1 else 1
            if cycle_check(edges_tmp, input_amounts_tmp):
                self.edges         = edges_tmp
                self.input_amounts = input_amounts_tmp
                alphas_tmp[max_alpha_id] = 0
                alphas_tmp[to_node_id*self.node_num+from_node_id] = 0
            else:
                alphas_tmp[max_alpha_id] = 0
            pbar.update(1)
        pbar.close()
        self.isolated_node_ids = [
            node_id for node_id in range(self.node_num)
            if node_id not in self.get_from_node_ids() and
               node_id not in self.get_to_node_ids() ]
        self.pruned = False

    def get_from_node_ids(self, edges=None) -> "list[int]":
        if edges is None: edges = self.edges
        return list(set([ edge[0] for edge in edges ]))

    def get_to_node_ids(self, edges=None) -> "list[int]":
        if edges is None: edges = self.edges
        return list(set([ edge[1] for edge in edges ]))

    def get_source_ids(self, edges=None) -> "list[int]":
        if edges is None: edges = self.edges
        return list(set([ node_id for node_id in self.get_from_node_ids(edges)
                          if node_id not in self.get_to_node_ids(edges) ]))

    def get_sink_ids(self, edges=None) -> "list[int]":
        if edges is None: edges = self.edges
        return list(set([ node_id for node_id in self.get_to_node_ids(edges)
                          if node_id not in self.get_from_node_ids(edges) ]))

    def prune_edges(self, prune_sources=False, prune_sinks=True):
        assert prune_sources or prune_sinks, "Prune???"
        edges_tmp = self.edges.copy()
        self.removed_source_ids:list[int] = []
        self.removed_sink_ids:list[int] = []
        if prune_sources:
            # print(self.get_source_ids(edges_tmp), end=' ')  # Debug
            pbar = tqdm(desc="Pruning sources")
            while self.get_source_ids(edges_tmp) != [ 0 ] and self.get_source_ids(edges_tmp) != [ 0, self.node_num-1 ]:
                if self.get_source_ids(edges_tmp) == []: raise Exception  # Debug
                for source_id in self.get_source_ids(edges_tmp):
                    if source_id == 0 or source_id == self.node_num-1: continue
                    edges_tmp = list(filter(lambda e: e[0]!=source_id, edges_tmp))
                    self.removed_source_ids.append(source_id)
                    pbar.update(1)
                # print(self.get_source_ids(edges_tmp), end=' ')  # Debug
            pbar.close()
            # print('')  # Debug
        if prune_sinks:
            # print(self.get_sink_ids(edges_tmp), end=' ')  # Debug
            pbar = tqdm(desc="Pruning sinks")
            while self.get_sink_ids(edges_tmp) != [ self.node_num-1 ]:
                if self.get_sink_ids(edges_tmp) == []: raise Exception  # Debug
                for sink_id in self.get_sink_ids(edges_tmp):
                    if sink_id == self.node_num-1: continue
                    edges_tmp = list(filter(lambda e: e[1]!=sink_id, edges_tmp))
                    self.removed_sink_ids.append(sink_id)
                    pbar.update(1)
                # print(self.get_sink_ids(edges_tmp), end=' ')  # Debug
            pbar.close()
            # print('')  # Debug
        self.pruned = True
        self.edges = edges_tmp

    def plot_DAG(self):
        if not self.pruned:
            imported_plot_DAG(graph_data={
                "node_num"          : self.node_num,
                "edges"             : self.edges,
                "isolated_node_ids" : self.isolated_node_ids,
                "source_ids"        : self.get_source_ids(),
                "removed_source_ids": [],
                "sink_ids"          : self.get_sink_ids(),
                "removed_sink_ids"  : [],
            }, filename="DAG_before_prune")
        else:
            imported_plot_DAG(graph_data={
                "node_num"          : self.node_num,
                "edges"             : self.edges,
                "isolated_node_ids" : self.isolated_node_ids,
                "source_ids"        : self.get_source_ids(),
                "removed_source_ids": self.removed_source_ids,
                "sink_ids"          : self.get_sink_ids(),
                "removed_sink_ids"  : self.removed_sink_ids,
            }, filename="DAG_after_prune")

    def forward(self, x):
        input_amounts_copy = self.input_amounts.copy()
        while 0 in input_amounts_copy:
            source_id = input_amounts_copy.index(0)
            edges_from_source = list(filter(lambda e: e[0]==source_id, self.edges))
            for edge in edges_from_source: input_amounts_copy[edge[1]] -= 1
            input_amounts_copy[source_id] = -1
            raise NotImplementedError
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

    NODE_NUMBER = 64
    NODE_DEPTH  = 4
    THRESHOLD_OF_CONNECTION  = 0.75  # t_c
    MAX_NUMBER_OF_CONNECTION = 4     # max_n_c >= 1
    MIN_NUMBER_OF_CONNECTION = 1     # min_n_c >= 1

    # for i in range(1000):
    #     print(f"{i:03}")
    model = Model(NODE_NUMBER, NODE_DEPTH, THRESHOLD_OF_CONNECTION,
                MAX_NUMBER_OF_CONNECTION, MIN_NUMBER_OF_CONNECTION)
    model.search_path(plot_dag=True)