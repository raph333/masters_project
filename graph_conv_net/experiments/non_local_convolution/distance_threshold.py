import numpy as np
import torch
from torch_geometric.transforms import Compose, Distance

from graph_conv_net.alchemy_dataset import TencentAlchemyDataset
from graph_conv_net.transformations import AddEdges
from graph_conv_net.train import run_experiment


def get_transform(threshold: float):
    return Compose([
        AddEdges(distance_threshold=threshold),
        Distance(norm=True)
    ])


CONFIG = {
    'name': 'tencent-mpnn-neighborhood-expansion-lr-decay',  # todo: set this for each experiment!  (default: use 'test-run')
    'target_param': {
        'name': 'distance_threshold',
        'values': [None, 1.5, 2, 3, 4, 5, np.inf]
    },
    'optimizer': torch.optim.Adam,
    'lr_scheduler': {
        'class': torch.optim.lr_scheduler.ExponentialLR,
        'kwargs': {'gamma': 0.98}
    },
    'dataset_class': TencentAlchemyDataset,
    'get_transform': get_transform,
    'repeat': 3,
    'lr':  0.002,
    'model_name': 'tencent_mpnn',
    'batch_size': 64,
    'num_epochs': 150,
    'cuda': 1
}


if __name__ == '__main__':

    run_experiment(CONFIG)
