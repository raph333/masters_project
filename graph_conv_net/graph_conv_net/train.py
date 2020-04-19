from os.path import join
from pprint import pprint
import time
from typing import Union, Callable

import pandas as pd
import mlflow

import torch
from torch import nn
from torch_geometric.data import DataLoader

from graph_conv_net import tencent_mpnn

DATA_DIR = '/home/rpeer/masters_project/data'


def train(net: nn.Module,
          train_loader: DataLoader,
          validation_loader: DataLoader,
          device: torch.device,
          loss_function: Callable,
          optimizer: torch.optim.Optimizer,
          num_epochs: int,
          lr_scheduler: Union[torch.optim.lr_scheduler._LRScheduler, None] = None) -> pd.DataFrame():

    print('epoch\ttrain-MAE\tvalid-MAE\t\tmin\t\tlr')
    print('-' * 60)
    log_df = pd.DataFrame(columns=['epoch', 'train_mae', 'valid_mae', 'minutes', 'lr'])
    net.to(device)
    start = time.time()

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

        minutes = (time.time() - start) / 60
        lr = optimizer.param_groups[0]['lr']
        print(f'{epoch}:\t\t{train_mae:.4f}\t\t{valid_mae:.4f}\t\t\t{minutes:.1f}\t\t{lr:.6f}')
        row = {'epoch': epoch,
               'train_mae': train_mae,
               'valid_mae': valid_mae,
               'minutes:': minutes,
               'lr': lr}
        log_df = log_df.append(row, ignore_index=True)

        if lr_scheduler is not None:
            lr_scheduler.step()

    return log_df


def run_experiment(config: dict):
    print(f'\nRUNNING EXPERIMENT: {config["name"]}\n:')
    pprint(config, width=1)

    mlflow.set_experiment(config['name'])
    target_param = config['target_param']['name']
    transform_creator = config['get_transform']
    DatasetClass = config['dataset_class']

    for i, param in enumerate(config['target_param']['values']):
        print(f'\nUSING {target_param} = {param}:')
        process = i == 0  # only re-process the first time

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
            model = tencent_mpnn.MPNN(node_input_dim=ds_dev[0].num_features,
                                      edge_input_dim=ds_dev[0].edge_attr.shape[1],
                                      output_dim=12)

            with mlflow.start_run():

                mlflow.log_params(config)
                mlflow.log_param('rep', rep)
                mlflow.log_param('target_param_name', target_param)
                mlflow.log_param('target_param_value', param)

                lr_schedule = config['lr_scheduler']
                if lr_schedule is not None:
                    opt = config['optimizer'](model.parameters(), lr=config['lr'])
                    scheduler = lr_schedule['class'](opt, **lr_schedule['kwargs'])

                learning_curve = train(net=model,
                                       train_loader=DataLoader(ds_dev, batch_size=config['batch_size'], shuffle=True),
                                       validation_loader=DataLoader(ds_valid, batch_size=config['batch_size']),
                                       device=torch.device(f'cuda:{config["cuda"]}'),
                                       loss_function=nn.L1Loss(),
                                       optimizer=opt,
                                       num_epochs=config['num_epochs'],
                                       lr_scheduler=scheduler)

                lc_file = 'learning_curve.csv'
                learning_curve.to_csv(lc_file, index=False)
                mlflow.log_artifact(lc_file)

    print('\nEXPERIMENT DONE.')


# def run_multiple_experiments(config_file: str):
#
#     with open(config_file) as infile:
#         config = json.load(infile)
#
#     for exp_name, exp_config in config.items():
#         print(f'\nRUNNING EXPERIMENT: {exp_name}:')
#         pprint(config, width=1)
#         # run_experiment(exp_name, exp_config)
#
#
# if __name__ == '__main__':
#
#     TRANS = Compose([FullyConnectedGraph(), Distance(norm=True)])
#     RE_PROCESS = False
#
#     run_multiple_experiments(config_file='config.py')
