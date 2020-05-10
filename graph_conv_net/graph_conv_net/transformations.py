import typing
from typing import Tuple, Union

from itertools import product
import numpy as np

import torch
from torch import tensor
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from torch_geometric.transforms import Distance


class CompleteGraph:

    def __call__(self, graph: Data) -> Data:
        num_bond_features = graph.edge_attr.shape[1]
        edge_features = torch.zeros(graph.num_nodes**2, num_bond_features).float()

        bond_index = graph.edge_index[0] * graph.num_nodes + graph.edge_index[1]
        edge_features[bond_index] = graph.edge_attr

        all_edges = list(product(range(graph.num_nodes), range(graph.num_nodes)))
        full_edge_index = tensor(all_edges).transpose(0, 1)

        edge_index, edge_attr = remove_self_loops(full_edge_index, edge_features)

        graph.edge_attr = edge_attr
        graph.edge_index = edge_index

        return graph

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class AddEdges:
    """
    Add an edge between every two atoms with distance <= distance_threshold
    """

    def __init__(self,
                 distance_threshold: Union[float, None] = np.inf,
                 norm_dist: bool = True,
                 add_dist_feature: bool = False):

        if distance_threshold is None:
            self.add_edges = False
        else:
            self.add_edges = True

        self.t = distance_threshold
        self.add_dist_feature = add_dist_feature

        self.complete_graph = CompleteGraph()
        self.distance = Distance(norm=norm_dist)
        self.distance_threshold = DistanceThreshold(threshold=self.t)

    def __call__(self, data: Data) -> Data:
        if self.add_edges is False:
            return data

        data = self.complete_graph(data)
        data = self.distance(data)
        data = self.distance_threshold(data)

        if self.add_dist_feature:
            return data

        data.edge_attr = data.edge_attr[:, :-1]  # remove distance column
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(distance_threshold={self.t})'


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


class NHop(object):
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
