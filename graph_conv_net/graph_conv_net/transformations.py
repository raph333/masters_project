from typing import Optional
from itertools import product
import numpy as np

import torch
from torch import tensor
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from torch_geometric.transforms import Distance


class CompleteGraph:
    """
    Adds an edge between any two nodes in the graph.
    If present, previous edges and their edge-features are preserved.
    """

    def __call__(self, graph: Data) -> Data:

        if graph.edge_attr is not None and graph.edge_attr.shape[0] > 0:
            # preserve all previous edges and edge-attributes:
            num_bond_features = graph.edge_attr.shape[1]
            edge_features = torch.zeros(graph.num_nodes**2, num_bond_features).float()
            bond_index = graph.edge_index[0] * graph.num_nodes + graph.edge_index[1]
            edge_features[bond_index] = graph.edge_attr

        else:
            edge_features = None

        all_edges = list(product(range(graph.num_nodes), range(graph.num_nodes)))
        full_edge_index = tensor(all_edges).transpose(0, 1)

        edge_index, edge_attr = remove_self_loops(full_edge_index, edge_features)

        graph.edge_attr = edge_attr
        graph.edge_index = edge_index

        return graph

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class DistanceThreshold:

    def __init__(self,
                 threshold: float = np.inf,
                 index: int = -1):
        """
        :param threshold: remove all edges with distance <= threshold
        :param index: index of edge-attribute column that represents the distance
        """

        self.t = threshold
        self.d_idx = index

    def __call__(self, data: Data) -> Data:
        mask = data.edge_attr[:, self.d_idx] <= self.t
        data.edge_attr = data.edge_attr[mask]
        data.edge_index = data.edge_index[:, mask]
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(threshold={self.t})'


class AddEdges:
    """
    Add an edge between every two atoms with distance <= distance_threshold
    """

    def __init__(self,
                 distance_threshold: Optional[float] = np.inf,
                 norm_dist: bool = True,
                 add_dist_feature: bool = False,
                 allow_removal_of_original_edges: bool = False):

        self.t = distance_threshold
        self.norm_dist = norm_dist
        self.add_dist_feature = add_dist_feature
        self.remove_edges_ok = allow_removal_of_original_edges

        self.calculate_complete_graph = CompleteGraph()
        self.calculate_distance = Distance(norm=False)
        self.apply_distance_threshold = DistanceThreshold(threshold=self.t)

    def __call__(self, data: Data) -> Data:

        # no edges to add:
        if self.t is None or self.t == 0:

            if self.add_dist_feature:
                data = self.calculate_distance(data)

                if self.norm_dist:
                    max_dist = data.edge_attr[:, -1].max()
                    data.edge_attr[:, -1] = data.edge_attr[:, -1] / max_dist
                    # problem: manual normalization is very slow
                    # maybe calculate normalized distance threshold first and use build-in normalization

            return data

        # add edges:
        num_original_edges = data.edge_attr.shape[0]
        data = self.calculate_complete_graph(data)
        data = self.calculate_distance(data)
        data = self.apply_distance_threshold(data)

        if data.edge_attr.shape[0] < num_original_edges and self.remove_edges_ok is False:
            raise AssertionError(f'Original edges are removed with threshold {self.t}')

        if self.add_dist_feature is False:
            data.edge_attr = data.edge_attr[:, :-1]  # remove distance column

        elif self.norm_dist:
            max_dist = data.edge_attr[:, -1].max()
            data.edge_attr[:, -1] = data.edge_attr[:, -1] / max_dist

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(distance_threshold={self.t})'


class NHop:
    """
    Adds the n-hop edges to the edge indices.
    (An edge will be added (if not there already) between any atoms <= n edges apart in the input graph)
    """

    def __init__(self, n):
        self.n = n

    def __call__(self, data):

        # todo: write method - check TwoHop implementation
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}({self.n})'
