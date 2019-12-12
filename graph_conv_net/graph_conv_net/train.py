from os.path import join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn

from torch_geometric.transforms import Compose, Distance, Cartesian
from torch_geometric.data import DataLoader

from graph_conv_net.alchemy_dataset import AlchemyDataset, FullyConnectedGraph
from graph_conv_net import tencent_mpnn


def count_parameters(net: nn.Module):
    return sum([np.prod(x.shape) for x in net.parameters()])


def print_lr_schedule(lr: float, decay: float, num_epochs=20, steps=5):
    print('\nlearning-rate schedule:')
    for i in range(num_epochs):
        if i % steps == 0:
            print(f'{i}\t{lr:.6f}')
        lr = lr * decay


def plot_error_curves(training_error: list,
                      validation_error: list,
                      error_name='error',
                      plot_name='learning_curve',
                      ylim=None,
                      save_fig=True):
    assert len(training_error) == len(validation_error) > 1

    fig, ax = plt.subplots()
    ax.plot(range(len(training_error)), training_error)
    ax.plot(range(len(validation_error)), validation_error)

    if ylim:
        ax.set_ylim(*ylim)

    ax.set_xlabel('epoch')
    ax.set_ylabel(error_name)
    ax.legend(('training', 'validation'))
    ax.set_title(f'{error_name} over time')

    if save_fig:
        fig.savefig(f'{plot_name}.png', bbox_inches='tight', transparent=True)

    plt.show()


# todo: logging
def train(net: nn.Module,
          train_loader: DataLoader,
          validation_loader: DataLoader,
          device: torch.device,
          loss_function,
          optimizer,
          num_epochs: int):
    print('epoch\ttrain-MAE\tvalid-MAE\tmin\t\tlr')
    print('-' * 60)
    net.to(device)

    for epoch in range(num_epochs):

        train_mae = valid_mae = 0

        for i, data in enumerate(train_loader):

            optimizer.zero_grad()
            net.train()

            data = data.to(device)
            y_hat = net(data)
            assert y_hat.shape == data.y.shape
            loss = loss_function(y_hat, data.y)
            train_mae += loss.item() / len(train_loader)

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            net.eval()
            for data in validation_loader:

                data = data.to(device)
                y_hat = net(data)
                loss = loss_function(y_hat, data.y)
                valid_mae += loss.item() / len(validation_loader)

        lr = optimizer.param_groups[0]['lr']
        print(f'{epoch}:\t{train_mae:.4f}\t\t{valid_mae:.4f}\t\t{lr:.6f}')


def predict(net: nn.Module,
            test_loader: DataLoader,
            device: torch.device) -> pd.DataFrame():

    net.eval()
    col_names = ['gbd_id', *[f'p_{i}' for i in range(12)]]
    answer_df = pd.DataFrame(columns=col_names)

    with torch.no_grad():
        for i, data in enumerate(test_loader):

            data = data.to(device)
            y_hat = net(data)

            ids = data.y.detach().cpu().numpy()
            predictions = y_hat.detach().cpu().numpy()

            pred_matrix = np.concatenate([np.expand_dims(ids, 1), predictions], axis=1)
            answer_df = pd.concat([answer_df, pd.DataFrame(pred_matrix, columns=col_names)])

    answer_df.gbd_id = answer_df.gbd_id.astype(int)
    answer_df = answer_df.sort_values('gbd_id')
    answer_df.to_csv('answer.csv', index=False, header=False)

    return answer_df


if __name__ == '__main__':

    DATA_DIR = '/home/rpeer/masters_project/data'
    trans = Compose([FullyConnectedGraph(), Distance(norm=True)])
    RE_PROCESS = False

    ds_valid = AlchemyDataset(root=join(DATA_DIR, 'valid'),
                              mode='valid',
                              transform=trans,
                              re_process=RE_PROCESS)
    ds_dev = AlchemyDataset(root=join(DATA_DIR, 'dev'),
                            mode='dev',
                            transform=trans,
                            re_process=RE_PROCESS)
    ds_test = AlchemyDataset(root=join(DATA_DIR, 'test'),
                             mode='test',
                             transform=trans,
                             re_process=RE_PROCESS)

    model = tencent_mpnn.MPNN(node_input_dim=11,
                              edge_input_dim=5,
                              output_dim=12)

    l_rate = 0.0001
    decay = 0.999

    opt = torch.optim.Adam(model.parameters(), lr=l_rate)

    train(net=model,
          train_loader=DataLoader(ds_dev, batch_size=64),
          validation_loader=DataLoader(ds_valid, batch_size=64),
          device=torch.device('cuda:1'),
          loss_function=nn.L1Loss(),
          optimizer=opt,
          num_epochs=1)

    predictions = predict(net=model,
                          test_loader=DataLoader(ds_test, batch_size=64),
                          device=torch.device('cuda:1'))


