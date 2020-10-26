import os
from os.path import join
from pprint import pprint
import time
from typing import Callable

import pandas as pd
import mlflow

import torch
from torch import nn
from torch_geometric.data import DataLoader, Dataset

from graph_conv_net import tools
# from graph_conv_net.models.schnet.schnet_model import SchNet
from graph_conv_net.models.tencent_mpnn import MPNN


DATA_DIR = '/scratch1/rpeer/tmp'


def evaluate(net: nn.Module,
             data_loader: DataLoader,
             device: torch.device,
             loss_function: Callable = nn.L1Loss()) -> float:
    error = 0
    with torch.no_grad():
        net.eval()

        for data in data_loader:
            data = data.to(device)
            y_hat = net(data)
            loss = loss_function(y_hat, data.y)
            error += loss.item() / len(data_loader)

    return error


def evaluate_state_dict(path: str,
                        model: nn.Module,
                        test_set: Dataset,
                        batch_size: int = 64,
                        cuda: int = 0) -> float:
    model.load_state_dict(torch.load(path))
    return evaluate(net=model,
                    data_loader=DataLoader(test_set, batch_size=batch_size),
                    device=torch.device(f'cuda:{cuda}'))


def train(net: nn.Module,
          train_loader: DataLoader,
          validation_loader: DataLoader,
          device: torch.device,
          optimizer: torch.optim.Optimizer,
          num_epochs: int,
          loss_function: Callable = nn.L1Loss(),
          lr_scheduler=None) -> pd.DataFrame():
    print('epoch\ttrain-MAE\tvalid-MAE\t\tmin\t\tlr')
    print('-' * 60)
    log_df = pd.DataFrame(columns=['epoch', 'train_mae', 'valid_mae', 'minutes', 'lr'])
    net.to(device)
    start = time.time()

    for epoch in range(num_epochs):

        train_error = 0

        for data in train_loader:
            optimizer.zero_grad()
            net.train()

            data = data.to(device)
            y_hat = net(data)
            assert y_hat.shape == data.y.shape
            loss = loss_function(y_hat, data.y)
            train_error += loss.item() / len(train_loader)

            loss.backward()
            optimizer.step()

        valid_error = evaluate(net=net,
                               data_loader=validation_loader,
                               device=device,
                               loss_function=loss_function)

        minutes = (time.time() - start) / 60
        lr = optimizer.param_groups[0]['lr']
        print(f'{epoch}:\t\t{train_error:.4f}\t\t{valid_error:.4f}\t\t\t{minutes:.1f}\t\t{lr:.8f}')
        row = {'epoch': epoch,
               'train_mae': train_error,
               'valid_mae': valid_error,
               'minutes': minutes,
               'lr': lr}
        log_df = log_df.append(row, ignore_index=True)

        if lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(valid_error)
            else:
                lr_scheduler.step()

    return log_df


def run_experiment(config: dict):
    print(f'\nRUNNING EXPERIMENT: {config["name"]}\n:')
    pprint(config, width=1)

    mlflow.set_experiment(config['name'])
    target_param = config['target_param']['name']
    transform_creator = config['get_transform']
    dataset_class = config['dataset_class']

    root_dir = join(DATA_DIR, config['name'])
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    ds_dev = dataset_class(root=root_dir, mode='dev')
    ds_valid = dataset_class(root=root_dir, mode='valid')
    # ds_test = dataset_class(root=root_dir, mode='test')

    for i, param in enumerate(config['target_param']['values']):
        print(f'\nUSING {target_param} = {param}:')

        ds_dev.transform = transform_creator(param)
        ds_valid.transform = transform_creator(param)
        # ds_test.transform = transform_creator(param)

        for rep in range(config['repeat']):

            print(f'\nrep number {rep}:')
            model = MPNN(node_input_dim=ds_dev[0].num_features,  # todo for other architectures: configure
                         edge_input_dim=ds_dev[0].edge_attr.shape[1])
            # model = SchNet()

            with mlflow.start_run():

                mlflow.log_params(config)
                mlflow.log_param('rep', rep)
                mlflow.log_param('target_param_name', target_param)
                mlflow.log_param('target_param_value', param)
                mlflow.log_param('num_params', tools.count_parameters(model))

                lr_schedule = config['lr_scheduler']
                if lr_schedule is not None:
                    opt = config['optimizer'](model.parameters(), lr=config['lr'])
                    scheduler = lr_schedule['class'](opt, **lr_schedule['kwargs'])

                learning_curve = train(net=model,
                                       train_loader=DataLoader(ds_dev, batch_size=config['batch_size'], shuffle=True),
                                       validation_loader=DataLoader(ds_valid, batch_size=config['batch_size']),
                                       device=torch.device(f'cuda:{config["cuda"]}'),
                                       optimizer=opt,
                                       num_epochs=config['num_epochs'],
                                       lr_scheduler=scheduler)

                # test_mae = evaluate(net=model,
                #                     data_loader=DataLoader(ds_test, batch_size=config['batch_size']),
                #                     device=torch.device(f'cuda:{config["cuda"]}'))
                # mlflow.log_metric('test-MAE', test_mae)

                lc_file = 'learning_curve.csv'
                learning_curve.to_csv(lc_file, index=False)
                mlflow.log_artifact(lc_file)

                weights_file = 'state_dict.pt'
                torch.save(model.state_dict(), weights_file)
                mlflow.log_artifact(weights_file)

    print('\nEXPERIMENT DONE.')
