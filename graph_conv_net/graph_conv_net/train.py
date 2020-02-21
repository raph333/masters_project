from os.path import join
from pprint import pprint

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

import torch
from torch import nn
from torch_geometric.transforms import Compose, Distance, Cartesian
from torch_geometric.data import DataLoader

from graph_conv_net import tencent_mpnn
from graph_conv_net.transformations import FullyConnectedGraph
DATA_DIR = '/home/rpeer/masters_project/data'


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


def train(net: nn.Module,
          train_loader: DataLoader,
          validation_loader: DataLoader,
          device: torch.device,
          loss_function,
          optimizer,
          num_epochs: int) -> pd.DataFrame():

    print('epoch\ttrain-MAE\tvalid-MAE\tmin\t\tlr')
    print('-' * 60)
    log_df = pd.DataFrame(columns=['epoch', 'train_mae', 'valid_mae'])
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

        #lr = optimizer.param_groups[0]['lr']
        print(f'{epoch}:\t{train_mae:.4f}\t\t{valid_mae:.4f}')
        log_df = log_df.append({'epoch': epoch, 'train_mae': valid_mae, 'valid_mae': train_mae}, ignore_index=True)

    return log_df


def run_experiment(config: dict):
    print(f'\nRUNNING EXPERIMENT: {config["name"]}:')
    pprint(config, width=1)

    mlflow.set_experiment(config['name'])
    target_param = config['target_param']['name']
    transform_creator = config['get_transform']
    DatasetClass = config['dataset_class']

    for i, param in enumerate(config['target_param']['values']):
        print(f'\nUSING {target_param} = {param}:')
        process = False  # i == 0  # only re-process the first time

        ds_valid = DatasetClass(root=join(DATA_DIR, 'valid'),
                                mode='valid',
                                transform=transform_creator(param),
                                re_process=process)
        ds_dev = DatasetClass(root=join(DATA_DIR, 'dev'),
                              mode='dev',
                              transform=transform_creator(param),
                              re_process=process)

        for rep in range(config['repeat']):

            print(f'\nrep number {rep}:')
            cuda_number = rep % len(config["cuda"])
            model = tencent_mpnn.MPNN(node_input_dim=ds_dev[0].num_features,
                                      edge_input_dim=ds_dev[0].edge_attr.shape[1],
                                      output_dim=12)

            with mlflow.start_run():

                mlflow.log_params(config)
                mlflow.log_param('rep', rep)
                mlflow.log_param('target_param', target_param)

                opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
                learning_curve_df = train(net=model,
                                          train_loader=DataLoader(ds_dev, batch_size=config['batch_size'], shuffle=True),
                                          validation_loader=DataLoader(ds_valid, batch_size=config['batch_size']),
                                          device=torch.device(f'cuda:{cuda_number}'),
                                          loss_function=nn.L1Loss(),
                                          optimizer=opt,
                                          num_epochs=config['num_epochs'])

                tmp_file_name = 'learning_curve.csv'
                learning_curve_df.to_csv(tmp_file_name, index=False)
                mlflow.log_artifact(tmp_file_name)
                # todo: add plot
    print('EXPERIMENT DONE.')


def run_multiple_experiments(config_file: str):

    with open(config_file) as infile:
        config = json.load(infile)

    for exp_name, exp_config in config.items():
        print(f'\nRUNNING EXPERIMENT: {exp_name}:')
        pprint(config, width=1)
        # run_experiment(exp_name, exp_config)


if __name__ == '__main__':

    TRANS = Compose([FullyConnectedGraph(), Distance(norm=True)])
    RE_PROCESS = False

    run_multiple_experiments(config_file='config.py')
