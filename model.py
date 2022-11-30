""" Libraries """
import torch
import matplotlib.pyplot as plt
plt.switch_backend("agg")

from tqdm import tqdm
from dag_networkx import plot_DAG as imported_plot_DAG


""" Functions """
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


class Model(torch.nn.Module):
    def __init__(self, node_num, max_depth, threshold_of_connection,
                 max_number_of_connection, min_number_of_connection, output_path):
        super(Model, self).__init__()
        assert max_number_of_connection >= 1, "max_number_of_connection should >= 1"
        assert min_number_of_connection >= 1, "min_number_of_connection should >= 1"
        self.node_num = node_num
        self.max_depth = max_depth
        self.nodes = torch.nn.ModuleList([
            LinearNode(node_num, max_depth, first=True) if node_id == 0 else
            LinearNode(node_num, max_depth,  last=True) if node_id == node_num-1 else
            LinearNode(node_num, max_depth) for node_id in range(node_num) ])
        alphas = torch.rand(node_num*node_num) * 0.1 + 0.9
        for i in range(node_num):
            alphas[i*node_num + i] = 0  # prevent node from linking to itself
            alphas[i*node_num + 0] = 0  # Make first node (node_id=0) source
            alphas[node_num*(node_num-1)+i] = 0  # Make last node (node_id=node_num-1) sink
        self.alphas = torch.nn.parameter.Parameter(alphas)
        self.t_c     = threshold_of_connection
        self.max_n_c = max_number_of_connection
        self.min_n_c = min_number_of_connection
        self.epoch = 1
        self.pruned = False
        self.output_path = output_path

    def freeze_alphas(self):
        self.alphas.requires_grad = False
    
    def unfreeze_alphas(self):
        self.alphas.requires_grad = True

    def freeze_nodes(self):
        for node in self.nodes: node.freeze()  # type: ignore

    def unfreeze_nodes(self):
        for node in self.nodes: node.unfreeze()  # type: ignore

    @property
    def input_amounts(self) -> "list[int]":
        output:list[int] = [ -1 ] * self.node_num
        for edge in self.edges:
            output[edge[1]] = 1 if output[edge[1]] == -1 else output[edge[1]] + 1
        for source_id in self.get_source_ids():
            output[source_id] = 0
        return output

    def search_path(self, plot_dag=False):
        # Initialize nodes' output every epoch starts
        for node in self.nodes: node.forwarded = False  # type: ignore
        self.link_edges()
        if plot_dag: self.plot_DAG()
        try:
            self.prune_edges(prune_sources=False, prune_sinks=True)
            if plot_dag: self.plot_DAG()
        except:
            self.plot_DAG()
            raise Exception
        self.epoch += 1
    
    def link_edges(self):
        self.edges:list[tuple[int, int, int]] = []
        input_amounts = self.input_amounts.copy()
        alphas_tmp = list(self.alphas.cpu().detach().numpy())
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
            edges_tmp, input_amounts_tmp = self.edges.copy(), input_amounts.copy()
            edges_tmp.append((from_node_id, to_node_id, max_alpha))
            if input_amounts_tmp[from_node_id] == -1: input_amounts_tmp[from_node_id] = 0
            input_amounts_tmp[to_node_id] = 1 if input_amounts_tmp[to_node_id] == -1 else input_amounts_tmp[to_node_id] + 1
            if cycle_check(edges_tmp, input_amounts_tmp):
                self.edges    = edges_tmp
                input_amounts = input_amounts_tmp
                alphas_tmp[max_alpha_id] = 0
                alphas_tmp[to_node_id*self.node_num+from_node_id] = 0
            else:
                alphas_tmp[max_alpha_id] = 0
            pbar.update(1)
        # pbar.close()
        # print('')
        self.isolated_node_ids = [
            node_id for node_id in range(self.node_num)
            if node_id not in self.get_from_node_ids() and
               node_id not in self.get_to_node_ids() ]
        self.pruned = False

    def get_from_node_ids(self, edges=None) -> "list[int]":
        if edges is None: edges = self.edges
        assert type(edges) == type([(0, 0)])
        return list(set([ edge[0] for edge in edges ]))

    def get_to_node_ids(self, edges=None) -> "list[int]":
        if edges is None: edges = self.edges
        assert type(edges) == type([(0, 0)])
        return list(set([ edge[1] for edge in edges ]))

    def get_source_ids(self, edges=None) -> "list[int]":
        if edges is None: edges = self.edges
        assert type(edges) == type([(0, 0)])
        return list(set([ node_id for node_id in self.get_from_node_ids(edges)
                          if node_id not in self.get_to_node_ids(edges) ]))

    def get_sink_ids(self, edges=None) -> "list[int]":
        if edges is None: edges = self.edges
        assert type(edges) == type([(0, 0)])
        return list(set([ node_id for node_id in self.get_to_node_ids(edges)
                          if node_id not in self.get_from_node_ids(edges) ]))

    def prune_edges(self, prune_sources, prune_sinks):
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
                    self.removed_source_ids.append(source_id)
                    removed_edges = list(filter(lambda e: e[0]==source_id, edges_tmp))
                    # for removed_edge in removed_edges: self.input_amounts[removed_edge[1]] -= 1
                    edges_tmp = list(filter(lambda e: e[0]!=source_id, edges_tmp))
                    pbar.update(1)
                # print(self.get_source_ids(edges_tmp), end=' ')  # Debug
            # pbar.close()
            # print('')
        if prune_sinks:
            # print(self.get_sink_ids(edges_tmp), end=' ')  # Debug
            pbar = tqdm(desc="Pruning sinks")
            while self.get_sink_ids(edges_tmp) != [ self.node_num-1 ]:
                if self.get_sink_ids(edges_tmp) == []: raise Exception  # Debug
                for sink_id in self.get_sink_ids(edges_tmp):
                    if sink_id == self.node_num-1: continue
                    self.removed_sink_ids.append(sink_id)
                    # self.input_amounts[sink_id] = -1
                    edges_tmp = list(filter(lambda e: e[1]!=sink_id, edges_tmp))
                    pbar.update(1)
                # print(self.get_sink_ids(edges_tmp), end=' ')  # Debug
            pbar.set_description("Pruning sinks")
            # pbar.close()
            # print('')
        self.pruned = True
        self.edges = edges_tmp
    
    def plot_DAG(self):
        plt.figure(figsize=(10, 10))
        if not self.pruned:
            plt.figure(figsize=(10, 10))
            plt.title(f"DAG_before_prune (Epoch: {self.epoch})", fontsize=30)
            imported_plot_DAG(graph_data={
                "node_num"          : self.node_num,
                "edges"             : self.edges,
                "isolated_node_ids" : self.isolated_node_ids,
                "source_ids"        : self.get_source_ids(),
                "removed_source_ids": [],
                "sink_ids"          : self.get_sink_ids(),
                "removed_sink_ids"  : [],
            })
            plt.tight_layout()
            plt.savefig(f"{self.output_path}/DAG_before_prune/{self.epoch:04}")
            if self.epoch == 1: plt.savefig(f"DAG_before_prune_{self.epoch}")  # Debug
        else:
            plt.title(f"DAG_after_prune (Epoch: {self.epoch})", fontsize=30)
            imported_plot_DAG(graph_data={
                "node_num"          : self.node_num,
                "edges"             : self.edges,
                "isolated_node_ids" : self.isolated_node_ids,
                "source_ids"        : self.get_source_ids(),
                "removed_source_ids": self.removed_source_ids,
                "sink_ids"          : self.get_sink_ids(),
                "removed_sink_ids"  : self.removed_sink_ids,
            })
            plt.tight_layout()
            plt.savefig(f"{self.output_path}/DAG_after_prune/{self.epoch:04}")
            if self.epoch == 1: plt.savefig(f"DAG_after_prune_{self.epoch}")  # Debug
        plt.close()

    def get_input(self, to_node_id, from_node_ids:"list[int]", batch_size):
        if len(from_node_ids) == 0:
            output = torch.zeros((batch_size, self.max_depth*(self.node_num-1), 32, 32))
            output = output.to(next(self.parameters()).device)
            return output
        input = []
        zeros = torch.zeros((batch_size, self.max_depth, 32, 32))
        zeros = zeros.to(next(self.parameters()).device)
        for node_id, node in enumerate(self.nodes):
            if node_id == to_node_id: continue
            elif node_id not in from_node_ids:
                input.append(zeros)
            else:
                alpha = torch.nn.Sigmoid()(self.alphas[node_id*(self.node_num)+to_node_id])
                assert node.forwarded
                input.append(alpha*node.output)
        input = torch.concat(input, dim=1)
        return input

    def forward(self, input_images:torch.Tensor) -> torch.Tensor:
        assert self.pruned
        input_amounts_copy = self.input_amounts.copy()
        source_id = 0
        while 0 in input_amounts_copy:
            source_id = input_amounts_copy.index(0)
            if source_id == 0:
                self.nodes[source_id](input_images)
            else:
                edges_to_source = list(filter(lambda e: e[1]==source_id, self.edges))
                from_node_ids = self.get_from_node_ids(edges_to_source)
                self.nodes[source_id](self.get_input(source_id, from_node_ids, input_images.shape[0]))
            edges_from_source = list(filter(lambda e: e[0]==source_id, self.edges))
            for edge in edges_from_source: input_amounts_copy[edge[1]] -= 1
            input_amounts_copy[source_id] = -1
            self.nodes[source_id].forwarded = True  # type: ignore
        assert source_id == self.node_num-1
        return self.nodes[source_id].output  # type: ignore


class LinearNode(torch.nn.Module):
    def __init__(self, node_num, max_depth, first=False, last=False):
        super(LinearNode, self).__init__()
        self.last = last
        input_size = 3 if first else (node_num-1)*max_depth
        if not last:
            self.layers = torch.nn.ModuleList([
                torch.nn.Conv2d(input_size, max_depth, kernel_size=1),
                torch.nn.ReLU(),
            ])
        else:
            self.layers = torch.nn.ModuleList([
                torch.nn.Conv2d(input_size, 4, kernel_size=1),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(32*32*4, 10),
                torch.nn.Softmax(dim=-1),
            ])
        self.forwarded = False

    def freeze(self):
        for param in self.parameters(): param.requires_grad = False
        
    def unfreeze(self):
        for param in self.parameters(): param.requires_grad = True

    def forward(self, input:torch.Tensor):
        x = input
        for layer in self.layers: x = layer(x)
        self.output:torch.Tensor = x
        assert self.output.shape[0] == input.shape[0]
        if not self.last:
            assert self.output.shape[2] == input.shape[2]
            assert self.output.shape[3] == input.shape[3]


""" Execution """
if __name__ == "__main__":

    NODE_NUMBER = 32
    NODE_DEPTH  = 4
    THRESHOLD_OF_CONNECTION  = 0.75  # t_c
    MAX_NUMBER_OF_CONNECTION = 4     # max_n_c >= 1
    MIN_NUMBER_OF_CONNECTION = 1     # min_n_c >= 1
    OUTPUT_PATH = "."

    model = Model(NODE_NUMBER, NODE_DEPTH, THRESHOLD_OF_CONNECTION,
                  MAX_NUMBER_OF_CONNECTION, MIN_NUMBER_OF_CONNECTION, OUTPUT_PATH)
    model.search_path(plot_dag=True)