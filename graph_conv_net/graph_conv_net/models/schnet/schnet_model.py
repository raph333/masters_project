import torch
from torch import nn
from torch.nn import Embedding, Linear, ModuleList

from torch_geometric.data.batch import Batch

from .layers import GaussianSmearing, ShiftedSoftplus, InteractionBlock


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
        self.output_dim = output_dim

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

        # this part currently only works on 1-molecule
        h_agg = torch.sum(h, dim=0).unsqueeze(0)
        result = self.readout(h_agg)
        return result

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff}), '
                f'output-dim={self.output_dim}')
