''' Libraries '''
import sys
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
        for i in range(node_num): self.alphas[i*node_num+i] = 0
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

            # Limitation
            edges_to_node = list(filter(lambda e: e[1]==to_node_id, edges))
            if len(edges_to_node) >= connection_limit:
                for from_node_id in range(self.node_num):
                    alphas_tmp[from_node_id*self.node_num+to_node_id] = 0
                continue

            edges_tmp, input_amounts_tmp = edges.copy(), input_amounts.copy()
            edges_tmp.append((from_node_id, to_node_id))
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
        self.edges = edges
        pbar.close()

    def forward(self, x):
        # for node_id in self.forward_queue:
        #     node_id
        return x


class Node(nn.Module):
    def __init__(self, node_id, node_num, max_depth, threshold_of_connection,
                 max_number_of_connection, min_number_of_connection):
        super(Node, self).__init__()
        # self.node_id = node_id
        # self.node_num = node_num
        # self.max_depth = max_depth
        # self.alphas = torch.rand(node_num-1)
        # # self.alphas = torch.ones(node_num-1)
        # self.passed_nodes:list[bool] = [ False ] * (node_num-1)
        # self.t_c     = threshold_of_connection
        # self.max_n_c = max_number_of_connection
        # self.min_n_c = min_number_of_connection
        self.linear = nn.Linear((node_num-1)*max_depth, max_depth)
        self.activation = nn.ReLU()

    # def get_max_alpha_node_ids(self):
    #     max_alphas:dict[int, float] = { alpha_id: alpha for alpha_id, alpha in enumerate(self.alphas.numpy()) }
    #     max_alphas = dict(filter(lambda ma: not self.passed_nodes[ma[0]], max_alphas.items()))
    #     max_alphas = dict(sorted(max_alphas.items(), key=lambda ma: ma[1], reverse=True))
    #     max_alphas_above = dict(filter(lambda ma: ma[1] > self.t_c, max_alphas.items()))
    #     if len(max_alphas_above) > self.min_n_c:
    #         max_alpha_node_ids = list(max_alphas_above.keys())[:self.max_n_c]
    #     else:
    #         max_alpha_node_ids = list(max_alphas.keys())[:self.min_n_c]
    #     self.max_alpha_node_ids_origin:list[int] = max_alpha_node_ids
    #     self.max_alpha_node_ids:list[int] = [ mani+1 if mani>=self.node_id else mani for mani in max_alpha_node_ids ]

    # def forward(self, inputs:dict):
    #     assert set(inputs.keys()) == set(self.max_alpha_node_ids_origin)
    #     inputs = dict((input[0]-1, input[1]) if input[0]>=self.node_id else input for input in inputs.items())
    #     x = torch.zeros((self.node_num-1)*self.max_depth)
    #     for input_id, input_value in inputs.items():
    #         x[input_id*self.max_depth:(input_id+1)*self.max_depth] = input_value*self.alphas[input_id]
    #     self.output = self.activation(self.linear(x))
        
    def forward(self, x):
        self.output = self.activation(self.linear(x))


''' Execution '''
if __name__ == "__main__":

    NUMBER_OF_NODES = 90

    model = Model(NUMBER_OF_NODES, 4, 0.75, 3, 1)
    model.search_path()
    from dag_networkx import plot_DAG
    plot_DAG(model.edges, NUMBER_OF_NODES)