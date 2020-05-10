import numpy as np
import torch
from torch_geometric.transforms import Compose, Distance

from graph_conv_net.alchemy_dataset import AlchemyCompetitionDataset
from graph_conv_net.transformations import AddEdges
from graph_conv_net.train import run_experiment
from base_configuration import CONFIG


def get_transform(threshold: float):
    return Compose([
        AddEdges(distance_threshold=threshold),
        Distance(norm=True)
    ])


new_config = {
    'name': 'tencent-mpnn-neighborhood-expansion-lr-decay',  # todo: set this for each experiment!  (default 'test-run')
    'target_param': {
        'name': 'distance_threshold',
        'values': [None, 1.5, 2, 3, 4, 5, np.inf]
    },
    'lr_scheduler': {
        'class': torch.optim.lr_scheduler.ExponentialLR,
        'kwargs': {'gamma': 0.99}
    },
    'dataset_class': AlchemyCompetitionDataset,
    'get_transform': get_transform,
    'repeat': 3,
    'lr':  0.001,
    'model_name': 'tencent_mpnn',
    'num_epochs': 150,
    'cuda': 1
}
CONFIG.update(new_config)


if __name__ == '__main__':

    run_experiment(CONFIG)
