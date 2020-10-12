import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set


class MPNN(torch.nn.Module):
    """
    Based on the MPNN from https://github.com/tencent-alchemy/Alchemy/blob/master/pyg/mpnn.py,
    but without weight sharing across graph-conv layers.
    """
    def __init__(self,
                 node_input_dim=15,
                 edge_input_dim=5,
                 output_dim=12,
                 node_hidden_dim=64,
                 edge_hidden_dim=128,
                 num_step_message_passing=6,
                 num_step_set2set=6):
        super().__init__()

        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)

        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim)
        )

        # initialize each conv-layer independently:
        for i in range(self.num_step_message_passing):
            setattr(self,
                    f'conv_{i}',
                    NNConv(node_hidden_dim,
                           node_hidden_dim,
                           edge_network,
                           aggr='mean',
                           root_weight=False)
                    )

        self.gru = nn.GRU(node_hidden_dim, node_hidden_dim)

        self.set2set = Set2Set(node_hidden_dim,
                               processing_steps=num_step_set2set)
        self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        self.lin2 = nn.Linear(node_hidden_dim, output_dim)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.num_step_message_passing):
            conv_layer = getattr(self, f'conv_{i}')
            m = F.relu(conv_layer(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        return self.lin2(out)


# if __name__ == '__main__':
#     model = MPNN()
#     print()
