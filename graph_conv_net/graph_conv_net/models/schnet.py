# import schnetpack.atomistic.output_modules
import torch
import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric.data.batch import Batch

import schnetpack as spk
import schnetpack.representation as rep
from schnetpack.datasets import *

# load qm9 dataset and download if necessary
q9 = QM9("qm9.db")

# split in train and val
train, val, test = q9.create_splits(100000, 10000)
loader = spk.data.AtomsLoader(train, batch_size=100, num_workers=4)
val_loader = spk.data.AtomsLoader(val)

# create model
reps = rep.SchNet()
output = spk.atomistic.Atomwise(n_in=128)
model = spk.atomistic.AtomisticModel(reps, output)

import torch_geometric.nn.models
from math import pi as PI
from torch import nn
from torch_scatter import scatter_cuda
from torch.nn import Embedding, Sequential, Linear, ModuleList
from torch_geometric.nn import radius_graph, MessagePassing
import schnetpack


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class SchNet(nn.Module):

    def __init__(self,
                 hidden_channels=128,
                 num_filters=128,
                 num_interactions=6,
                 num_gaussians=50,
                 cutoff=10.0,
                 output_dim=12):

        super(SchNet, self).__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians

        self.embedding = Embedding(100, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels,
                                     num_gaussians,
                                     num_filters,
                                     cutoff)
            self.interactions.append(block)

        hidden_dim_2 = hidden_channels // 2
        self.lin1 = Linear(hidden_channels, hidden_dim_2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_dim_2, hidden_dim_2)

        self.readout = Linear(hidden_dim_2, output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, data: Batch):

        h = self.embedding(data.x)

        expanded_dist = self.distance_expansion(data.edge_attr)

        for interaction in self.interactions:
            h = h + interaction(h,
                                data.edge_index,
                                data.edge_attr,  # normalized distance
                                expanded_dist  # expanded distance
                                )

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        h_agg = torch.sum(h, dim=0)
        result = self.readout(h_agg)
        return result

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')


# class Structure:
#     """
#     Keys to access structure properties loaded using `schnetpack.data.AtomsData`
#     """
#     Z = '_atomic_numbers'
#     atom_mask = '_atom_mask'
#     R = '_positions'
#     cell = '_cell'
#     neighbors = '_neighbors'
#     neighbor_mask = '_neighbor_mask'
#     cell_offset = '_cell_offset'
#     neighbor_pairs_j = '_neighbor_pairs_j'
#     neighbor_pairs_k = '_neighbor_pairs_k'
#     neighbor_pairs_mask = '_neighbor_pairs_mask'


# class SchNetDGL(nn.Module):
#
#     def __init__(self,
#                  n_atom_basis=64,
#                  n_filters=128,
#                  n_interactions=1,
#                  cutoff=5.0,
#                  n_gaussians=25,
#                  normalize_filter=False,
#                  coupled_interactions=False,
#                  return_intermediate=False,
#                  max_z=8,
#                  distance_expansion=None):
#         super().__init__()
#
#         # atom type embeddings
#         self.embedding = Embedding(max_z, n_atom_basis, padding_idx=0)
#
#         # spatial features
#         # self.distances = schnetpack.nn.neighbors.AtomDistances()
#         if distance_expansion is None:
#             # self.distance_expansion = schnetpack.nn.acsf.GaussianSmearing(0.0,
#             #                                                               cutoff,
#             #                                                               n_gaussians,
#             #                                                               trainable=trainable_gaussians)
#             self.distance_expansion = GaussianSmearing(start=0.0, stop=cutoff, num_gaussians=n_gaussians)
#         else:
#             self.distance_expansion = distance_expansion
#
#         self.return_intermediate = return_intermediate
#
#         # interaction network
#         if coupled_interactions:
#             self.interactions = nn.ModuleList(
#                 [InteractionBlock()]
#                 * n_interactions)
#         else:
#             self.interactions = nn.ModuleList([
#                 InteractionBlock(n_atom_basis=n_atom_basis,
#                                   n_spatial_basis=n_gaussians,
#                                   n_filters=n_filters,
#                                   normalize_filter=normalize_filter)
#                 for _ in range(n_interactions)
#             ])
#
#     def forward(self, data: Batch):
#         """
#         Args:
#             data (dict of torch.Tensor): SchNetPack format dictionary of input tensors.
#
#         Returns:
#             torch.Tensor: Final Atom-wise SchNet representation.
#             torch.Tensor: Atom-wise SchNet representation of intermediate layers.
#         """
#         print()
#
#         atomic_numbers = data[Structure.Z]
#         positions = data[Structure.R]
#         cell = data[Structure.cell]
#         cell_offset = data[Structure.cell_offset]
#         neighbors = data[Structure.neighbors]
#         neighbor_mask = data[Structure.neighbor_mask]
#
#         # atom embedding
#         x = self.embedding(data.x)
#
#         # spatial features
#         # r_ij = self.distances(positions, neighbors, cell, cell_offset)
#         f_ij = self.distance_expansion(data.edge_attr)
#
#         # interactions
#         # if self.return_intermediate:
#         #     xs = [x]
#
#         for interaction in self.interactions:
#             v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
#             x = x + v
#
#             if self.return_intermediate:
#                 xs.append(xs)
#
#         if self.return_intermediate:
#             return x, xs
#         return x


# class SchNetInteraction(nn.Module):
#     """
#     SchNet interaction block for modeling quantum interactions of atomistic systems.
#
#     Args:
#         n_atom_basis (int): number of features used to describe atomic environments
#         n_spatial_basis (int): number of input features of filter-generating networks
#         n_filters (int): number of filters used in continuous-filter convolution
#         normalize_filter (bool): if true, divide filter by number of neighbors over which convolution is applied
#     """
#
#     def __init__(self, n_atom_basis, n_spatial_basis, n_filters,
#                  normalize_filter=False):
#         super(SchNetInteraction, self).__init__()
#
#         # initialize filters
#         self.filter_network = nn.Sequential(
#             schnetpack.nn.base.Dense(n_spatial_basis,
#                                      n_filters,
#                                      activation=schnetpack.nn.activations.shifted_softplus),
#             schnetpack.nn.base.Dense(n_filters, n_filters)
#         )
#
#         # initialize interaction blocks
#         self.cfconv = schnetpack.nn.cfconv.CFConv(n_atom_basis,
#                                                   n_filters,
#                                                   n_atom_basis,
#                                                   self.filter_network,
#                                                   activation=schnetpack.nn.activations.shifted_softplus,
#                                                   normalize_filter=normalize_filter)
#         self.dense = schnetpack.nn.base.Dense(n_atom_basis, n_atom_basis)
#
#     def forward(self, x, r_ij, neighbors, neighbor_mask, f_ij=None):
#         """
#         Args:
#             x (torch.Tensor): Atom-wise input representations.
#             r_ij (torch.Tensor): Interatomic distances.
#             neighbors (torch.Tensor): Indices of neighboring atoms.
#             neighbor_mask (torch.Tensor): Mask to indicate virtual neighbors introduced via zeros padding.
#             f_ij (torch.Tensor): Use at your own risk.
#
#         Returns:
#             torch.Tensor: SchNet representation.
#         """
#         v = self.cfconv.forward(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
#         v = self.dense.forward(v)
#         return v
