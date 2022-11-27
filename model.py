''' Libraries '''
import torch
import torch.nn as nn
from tqdm import tqdm


''' Functions '''
class Model(nn.Module):
    def __init__(self, node_num, max_depth, threshold_of_connection,
                 max_number_of_connection, min_number_of_connection):
        super(Model, self).__init__()
        self.node_num = node_num
        self.nodes : list[Node] = [ 
            Node(node_id, node_num, max_depth, threshold_of_connection,
                 max_number_of_connection, min_number_of_connection)
            for node_id in range(node_num)
        ]

    def forward_bfs(self):
        raise NotImplementedError

    def backward_bfs(self):
        unsearch_node_ids = [ self.node_num-1 ]
        current_id = 0
        pbar = tqdm(total=self.node_num)
        while current_id < len(unsearch_node_ids):
            searching_node = self.nodes[unsearch_node_ids[current_id]]
            searching_node.get_max_alpha_node_ids()
            new_passed_nodes : list = searching_node.passed_nodes.copy()
            new_passed_nodes.insert(searching_node.node_id, True)
            for input_node_id in searching_node.max_alpha_node_ids:
                new_passed_nodes_tmp = new_passed_nodes.copy()
                new_passed_nodes_tmp.pop(input_node_id)
                self.nodes[input_node_id].passed_nodes = new_passed_nodes_tmp
                if input_node_id not in unsearch_node_ids:
                    unsearch_node_ids.append(input_node_id)
            current_id += 1
            pbar.update(1)
        pbar.close()

    def get_backward_dag(self):
        edges, isolated_nodes = [], []
        for node in self.nodes:
            try:
                for input_node_id in node.max_alpha_node_ids:
                    edges.append((input_node_id, node.node_id))
            except AttributeError:
                isolated_nodes.append(node.node_id)
        return edges, isolated_nodes

    def forward(self, x):
        return x


class Node(nn.Module):
    def __init__(self, node_id, node_num, max_depth, threshold_of_connection,
                 max_number_of_connection, min_number_of_connection):
        super(Node, self).__init__()
        self.node_id = node_id
        self.alphas = torch.rand(node_num-1)
        # self.alphas = torch.ones(node_num-1)
        self.passed_nodes : list = [ False ] * (node_num-1)
        self.t_c     = threshold_of_connection
        self.max_n_c = max_number_of_connection
        self.min_n_c = min_number_of_connection

        self.linear = nn.Linear((node_num-1)*max_depth, max_depth)
        self.activation = nn.ReLU()

    def get_max_alpha_node_ids(self):
        max_alphas = {}
        for passed_node_id, passed in enumerate(self.passed_nodes):
            if passed: continue
            alpha = self.alphas[passed_node_id]
            if len(max_alphas.values()) != 0 and alpha < min(max_alphas.values()): continue
            n_c = self.max_n_c if alpha > self.t_c else self.min_n_c
            if len(max_alphas.values()) >= n_c:
                for max_alpha_node_id, max_alpha in max_alphas.items():
                    if max_alpha == min(max_alphas.values()):
                        delete_max_alpha_node_id = max_alpha_node_id
                        break
                del max_alphas[delete_max_alpha_node_id]
            if passed_node_id >= self.node_id: passed_node_id += 1
            max_alphas[passed_node_id] = alpha
        self.max_alpha_node_ids = list(max_alphas.keys())

    def forward(self, x):
        self.output = x


''' Execution '''
if __name__ == "__main__":

    NUMBER_OF_NODES = 87

    model = Model(NUMBER_OF_NODES, 4, 0.75, 3, 1)
    model.backward_bfs()
    edges, isolated_nodes = model.get_backward_dag()
    from dag_networkx import plot_DAG
    plot_DAG(edges, isolated_nodes, 0, NUMBER_OF_NODES-1)