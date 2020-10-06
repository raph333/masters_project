"""
This code works fine.
Find out why refactored code does not work properly.
"""

import warnings
warnings.filterwarnings("ignore")
from os.path import join
from time import time
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import DataLoader
from torch_geometric.nn import NNConv, Set2Set
import torch_geometric.transforms as T

from old_data import OldAlchemyDataset, Complete

DATA_DIR = '/scratch1/rpeer/alchemy/data'
raw_dir = join(DATA_DIR, 'raw')

trn_df = pd.read_csv(join(raw_dir, 'dev/dev_target.csv'))
val_df = pd.read_csv(join(raw_dir, 'valid/valid_target.csv'))
val_df.head()


class MPNN(torch.nn.Module):
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
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))

        self.conv = NNConv(node_hidden_dim,
                           node_hidden_dim,
                           edge_network,
                           aggr='mean',
                           root_weight=False)
        self.gru = nn.GRU(node_hidden_dim, node_hidden_dim)

        self.set2set = Set2Set(node_hidden_dim,
                               processing_steps=num_step_set2set)

        self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        self.lin2 = nn.Linear(node_hidden_dim, output_dim)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.num_step_message_passing):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)

        return out


batchsize = 64
tf = T.Compose([Complete(), T.Distance(norm=False)])

train_dataset = OldAlchemyDataset(root=DATA_DIR, mode='dev',   transform=tf)
valid_dataset = OldAlchemyDataset(root=DATA_DIR, mode='valid', transform=tf)
print(f'training molecules:   {len(train_dataset)}')
print(f'validation molecules: {len(valid_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batchsize)
print(f'\ntraining batches:   {len(train_loader)}')
print(f'validation batches: {len(valid_loader)}')


DEVICE = torch.device('cuda:0')
net = MPNN(edge_input_dim=5)
net.to(DEVICE)

l_rate = 0.001
#decay  = 0.999
epochs = 300

loss_function = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=l_rate, weight_decay=0)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=10, threshold=1e-4)

print('epoch\ttrain-MAE\tvalid-MAE\tmin\t\tlr')
print('-' * 60)
log_df = pd.DataFrame(columns=['train_mae', 'valid_mae'])

for epoch in range(epochs):

    train_mae = valid_mae = 0
    start = time()

    for i, data in enumerate(train_loader):

        optimizer.zero_grad()
        net.train()

        data = data.to(DEVICE)
        y_hat = net(data)
        assert y_hat.shape == data.y.shape
        loss = loss_function(y_hat, data.y)
        train_mae += loss.item() / len(train_loader)

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        net.eval()
        for data in valid_loader:
            data = data.to(DEVICE)
            y_hat = net(data)
            loss = loss_function(y_hat, data.y)
            valid_mae += loss.item() / len(valid_loader)

    lr = optimizer.param_groups[0]['lr']
    scheduler.step(valid_mae)
    log_df = log_df.append({'train_mae': valid_mae, 'valid_mae': train_mae}, ignore_index=True)
    duration = (time() - start) / 60
    print(f'{epoch}:\t{train_mae:.4f}\t\t{valid_mae:.4f}\t\t{duration:.2f}\t\t{lr:.6f}')

log_df.to_csv('old_mpnn.csv', index=False)