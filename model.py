""" Libraries """
import os
import time
import torch
import shutil
import numpy as np
import seaborn as sns
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
    def __init__(self, node_num, node_depth, threshold_of_connection,
                 max_number_of_connection, min_number_of_connection, criterion, output_path, device):
        super(Model, self).__init__()
        assert max_number_of_connection >= 1, "max_number_of_connection should >= 1"
        assert min_number_of_connection >= 1, "min_number_of_connection should >= 1"
        self.node_num = node_num
        self.node_depth = node_depth
        self.nodes = torch.nn.ModuleList([
            Node(node_depth, first=True) if node_id == 0 else
            Node(node_depth,  last=True) if node_id == node_num-1 else
            Node(node_depth) for node_id in range(node_num) ])
        # alphas_tensor = (torch.rand(node_num*node_num)-0.5)*0.05 + 1.4  # sigmoid(1.4) ~ 0.8
        alphas_tensor = torch.ones(node_num*node_num) * 1.4
        self.alphas_tensor = torch.nn.parameter.Parameter(alphas_tensor)
        self.alphas_mask = torch.ones(node_num*node_num)
        for i in range(node_num):
            self.alphas_mask[i*node_num + i] = 0  # prevent node from linking to itself
            self.alphas_mask[i*node_num + 0] = 0  # Make first node (node_id=0) source
            self.alphas_mask[node_num*(node_num-1)+i] = 0  # Make last node (node_id=node_num-1) sink
        self.t_c     = threshold_of_connection
        self.max_n_c = max_number_of_connection
        self.min_n_c = min_number_of_connection
        self.epoch = 1
        self.pruned = False
        self.output_path = output_path
        self.device = device

        """ Add from DARTS """
        self.criterion = criterion

    def freeze_alphas(self):
        self.alphas_tensor.requires_grad = False
    
    def unfreeze_alphas(self):
        self.alphas_tensor.requires_grad = True

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

    @property
    def alphas(self) -> torch.Tensor:
        alphas_mask = self.alphas_mask.to(self.alphas_tensor.device)
        return torch.nn.Sigmoid()(
            self.alphas_tensor*alphas_mask)*alphas_mask

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
    
    def plot_DAG(self):
        plt.figure(figsize=(10*max(np.log2(self.node_num)-5, 1), 10*max(np.log2(self.node_num)-5, 1)))
        if not self.pruned:
            os.makedirs(f"{self.output_path}/DAG_before_prune", exist_ok=True)
            plt.title(f"DAG_before_prune (Epoch: {self.epoch})", fontsize=30*max(np.log2(self.node_num)-5, 1))
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
            print("Saving DAG_before_prune... ", end='', flush=True)
            start_time = time.time()
            plt.savefig(f"{self.output_path}/DAG_before_prune/{self.epoch:04}")
            shutil.copy(f"{self.output_path}/DAG_before_prune/{self.epoch:04}.png", f"{self.output_path}/DAG_before_prune/0000_last.png")
            # if self.epoch == 1: shutil.copy(f"{self.output_path}/DAG_before_prune/{self.epoch:04}.png", "DAG_before_prune_first.png")  # Debug
            shutil.copy(f"{self.output_path}/DAG_before_prune/{self.epoch:04}.png", "DAG_before_prune_last.png")  # Debug
            print(f"cost time: {time.time()-start_time:.2f} seconds")
        else:
            os.makedirs(f"{self.output_path}/DAG_after_prune", exist_ok=True)
            plt.title(f"DAG_after_prune (Epoch: {self.epoch})", fontsize=30*max(np.log2(self.node_num)-5, 1))
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
            print("Saving DAG_after_prune... ", end='', flush=True)
            start_time = time.time()
            plt.savefig(f"{self.output_path}/DAG_after_prune/{self.epoch:04}")
            shutil.copy(f"{self.output_path}/DAG_after_prune/{self.epoch:04}.png", f"{self.output_path}/DAG_after_prune/0000_last.png")
            # if self.epoch == 1: shutil.copy(f"{self.output_path}/DAG_after_prune/{self.epoch:04}.png", "DAG_after_prune_first.png")  # Debug
            shutil.copy(f"{self.output_path}/DAG_after_prune/{self.epoch:04}.png", "DAG_after_prune_last.png")  # Debug
            print(f"cost time: {time.time()-start_time:.2f} seconds")
        plt.close()

    def plot_alphas(self):
        os.makedirs(f"{self.output_path}/Alphas", exist_ok=True)
        base = max(np.log2(self.node_num)-3, 1)
        fig, axs = plt.subplots(1, 2, figsize=(12*base, 6*base))
        fig.suptitle(f"Alphas (Epoch: {self.epoch}, t_c: {self.t_c:.3f})", fontsize=12*base)

        axs[0].set_title("alphas", fontsize=10*base)
        alphas = torch.reshape(self.alphas.cpu().detach(), (self.node_num, self.node_num))
        alphas = alphas.numpy()
        if self.node_num <= 64:
            sns.heatmap(alphas, annot=True, fmt=".1f", ax=axs[0], vmax=1, vmin=0)
        else:
            sns.heatmap(alphas, fmt=".1f", ax=axs[1], vmax=1, vmin=0)

        axs[1].set_title("selected alphas", fontsize=10*base)
        alphas_masked = np.zeros((self.node_num, self.node_num))
        for edge in self.edges: alphas_masked[edge[0]][edge[1]] = 1
        alphas_masked = alphas*alphas_masked
        if self.node_num <= 64:
            sns.heatmap(alphas_masked, annot=True, fmt=".1f", ax=axs[1], vmax=1, vmin=0)
        else:
            sns.heatmap(alphas_masked, fmt=".1f", ax=axs[1], vmax=1, vmin=0)

        plt.tight_layout()
        print("Saving Alphas... ", end='', flush=True)
        start_time = time.time()
        plt.savefig(f"{self.output_path}/Alphas/{self.epoch:04}")
        shutil.copy(f"{self.output_path}/Alphas/{self.epoch:04}.png", f"{self.output_path}/Alphas/0000_last.png")
        # if self.epoch == 1: shutil.copy(f"{self.output_path}/Alphas/{self.epoch:04}.png", "Alphas_first.png")  # Debug
        shutil.copy(f"{self.output_path}/Alphas/{self.epoch:04}.png", "Alphas_last.png")  # Debug
        print(f"cost time: {time.time()-start_time:.2f} seconds")
        plt.close()

    def search_path(self, plot_dag=True, plot_alpha=True):
        # Initialize nodes' output every epoch starts
        for node in self.nodes: node.forwarded = False  # type: ignore
        self.link_edges()
        if plot_dag: self.plot_DAG()
        try:
            self.prune_edges(prune_sources=True, prune_sinks=True)
            if plot_dag: self.plot_DAG()
        except:
            self.plot_DAG()
            raise Exception
        if plot_alpha: self.plot_alphas()
        self.epoch += 1
    
    def link_edges(self):
        self.edges:list[tuple[int, int, int]] = []
        input_amounts = self.input_amounts.copy()
        alphas_tmp = list(self.alphas.cpu().detach().numpy())
        pbar = tqdm(desc="Backward searching forwarding path")
        while True:
            max_alpha = max(alphas_tmp)
            if max_alpha < 0.05: break
            max_alpha_id = alphas_tmp.index(max_alpha)
            from_node_id = max_alpha_id // self.node_num
            to_node_id   = max_alpha_id  % self.node_num

            
            # Limitation
            connection_limit = self.max_n_c if max_alpha > self.t_c else self.min_n_c
            # Limitation to inputs of "to_node" (node_id=to_node_id)
            to_node:Node = self.nodes[to_node_id]  # type: ignore
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
                    edges_tmp = list(filter(lambda e: e[1]!=sink_id, edges_tmp))
                    pbar.update(1)
                # print(self.get_sink_ids(edges_tmp), end=' ')  # Debug
            # pbar.close()
            # print('')
        self.pruned = True
        self.edges = edges_tmp

    def get_input(self, to_node_id, from_node_ids:"list[int]", batch_size):
        input = torch.zeros((batch_size, self.node_depth, 32, 32))
        input = input.to(next(self.parameters()).device)
        for from_node_id in from_node_ids:
            from_node:Node = self.nodes[from_node_id]  # type: ignore
            assert from_node.forwarded
            alpha = self.alphas[from_node_id*(self.node_num)+to_node_id]
            input += alpha*from_node.output
        return input

    # def get_input(self, to_node_id, from_node_ids:"list[int]", batch_size):
    #     if len(from_node_ids) == 0:
    #         return torch.zeros(batch_size, self.node_depth, 32, 32)
    #     inputs, alphas = [], []
    #     for from_node_id in from_node_ids:
    #         from_node:Node = self.nodes[from_node_id]  # type: ignore
    #         assert from_node.forwarded
    #         alphas.append(torch.unsqueeze(self.alphas[from_node_id*(self.node_num)+to_node_id], dim=0))
    #         inputs.append(from_node.output)
    #     alphas = torch.concat(alphas)
    #     alphas = torch.nn.Softmax(dim=-1)(alphas)
    #     return sum([ i*a for i, a in zip(inputs, alphas) ])

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
                edges_to_source = sorted(edges_to_source, key=lambda e: e[2], reverse=True)  # sort by alpha in descending order
                from_node_ids = self.get_from_node_ids(edges_to_source)
                self.nodes[source_id](
                    self.get_input(source_id, from_node_ids, input_images.shape[0]))
            edges_from_source = list(filter(lambda e: e[0]==source_id, self.edges))
            for edge in edges_from_source: input_amounts_copy[edge[1]] -= 1
            input_amounts_copy[source_id] = -1
            self.nodes[source_id].forwarded = True  # type: ignore
        assert source_id == self.node_num-1
        return self.nodes[source_id].output  # type: ignore

    """ Add from DARTS """
    def new(self):
        model_new = Model(self.node_num, self.node_depth, self.t_c, self.max_n_c, self.min_n_c,
                          self.criterion, self.output_path, self.device).to(self.device)
        for x, y in zip([ model_new.alphas ], [ self.alphas ]):
            x.data.copy_(y.data)
        return model_new

    def _loss(self, input, target):
        logits = self(input)
        return self.criterion(logits, target)


class Node(torch.nn.Module):
    def __init__(self, node_depth, first=False, last=False):
        super(Node, self).__init__()
        self.last = last
        activation_function = torch.nn.GELU()
        # activation_function = torch.nn.SiLU()
        if first:
            self.layers = torch.nn.ModuleList([
                torch.nn.Conv2d(3, node_depth, 3, padding=1),  # bias=False),
            ])
        elif last:
            self.layers = torch.nn.ModuleList([
                activation_function,
                torch.nn.BatchNorm2d(node_depth),
                # ---
                torch.nn.Conv2d(node_depth, 32, 3, stride=2, padding=1),  # bias=False),
                activation_function,
                torch.nn.BatchNorm2d(32),
                # ---
                torch.nn.Conv2d(32, 8, 3, stride=2, padding=1),  # bias=False),
                activation_function,
                torch.nn.BatchNorm2d(8),
                # ---
                torch.nn.Flatten(),
                torch.nn.Linear(8*8*8, 128),  # bias=False),
                activation_function,
                torch.nn.BatchNorm1d(128),
                # ---
                torch.nn.Linear(128, 10),  # bias=False),
                activation_function,
                torch.nn.Softmax(dim=-1),
            ])
        else:
            self.layers = torch.nn.ModuleList([
                activation_function,
                torch.nn.BatchNorm2d(node_depth),
                # ---
                torch.nn.Conv2d(node_depth, node_depth, 1),  # bias=False),
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


class SigmoidConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SigmoidConv2d, self).__init__()
        self.bias = bias
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        weight = torch.nn.Sigmoid()(self.conv2d.weight)  # / (self.kernel_size**2*self.in_channels)
        bias = torch.nn.Sigmoid()(self.conv2d.bias) if self.bias else self.conv2d.bias
        return self.conv2d._conv_forward(input, weight, bias)


class SigmoidLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(SigmoidLinear, self).__init__()
        self.bias = bias
        self.in_features = in_features
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        weight = torch.nn.Sigmoid()(self.linear.weight)  # / self.in_features
        bias = torch.nn.Sigmoid()(self.linear.bias) if self.bias else self.linear.bias
        return torch.nn.functional.linear(input, weight, bias)


""" Execution """
if __name__ == "__main__":

    NODE_NUMBER = 32
    NODE_DEPTH  = 4
    THRESHOLD_OF_CONNECTION  = 0.25  # t_c
    MAX_NUMBER_OF_CONNECTION = 99    # max_n_c >= 1
    MIN_NUMBER_OF_CONNECTION = 1     # min_n_c >= 1
    OUTPUT_PATH = "."

    model = Model(NODE_NUMBER, NODE_DEPTH, THRESHOLD_OF_CONNECTION,
                  MAX_NUMBER_OF_CONNECTION, MIN_NUMBER_OF_CONNECTION, OUTPUT_PATH)
    model.search_path(plot_dag=True)