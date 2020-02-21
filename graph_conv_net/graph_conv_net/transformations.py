import typing
from typing import Tuple, Union

from itertools import product
import numpy as np
from scipy.spatial import distance_matrix

import torch
from torch import tensor
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops


class FullyConnectedGraph(object):

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


class AddEdges(object):
    """
    Add an edge between every two atoms with distance <= distance_threshold
    """

    def __init__(self, distance_threshold=np.inf):

        if distance_threshold is None:
            self.add_edges = False
        else:
            self.add_edges = True

        self.t = distance_threshold

    def __call__(self, data: Data) -> Data:

        if self.add_edges is False:
            return data

        dist_matrix = distance_matrix(data.pos, data.pos)  # todo use torch
        row, col = data.edge_index
        edges = []
        edge_attributes = []

        from_atom, to_atom = data.edge_index

        for i, j in product(range(data.num_nodes), range(data.num_nodes)):

            if i == j:
                continue

            existing_edge = ((from_atom == i) * (to_atom == j)).nonzero()
            assert len(existing_edge) <= 1, 'There can be only one edge from atom a to atom b'

            if len(existing_edge) == 1:
                edges.append((i, j))
                edge_attributes.append(data.edge_attr[existing_edge.item()].unsqueeze(0))

            # create new edge:
            elif dist_matrix[i, j] <= self.t:
                edges.append((i, j))
                edge_attributes.append(torch.zeros(*data.edge_attr.shape[1:]).unsqueeze(0))

        assert len(edges) == len(edge_attributes) > 0

        data.edge_index = torch.tensor(edges).transpose(0, 1)
        data.edge_attr = torch.stack(edge_attributes, dim=1).squeeze()

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(distance_threshold={self.t})'


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
