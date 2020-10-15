from typing import Callable
import numpy as np
import torch

from graph_conv_net.train import run_experiment
from graph_conv_net.alchemy_dataset import AlchemyCompetitionDataset
from graph_conv_net.data_processing import TencentDataProcessor
from graph_conv_net.transformations import AddEdges
from experiments.base_configuration import CONFIG

AlchemyCompetitionDataset.data_processor = TencentDataProcessor()


def get_transform(threshold: float) -> Callable:
    return AddEdges(distance_threshold=threshold,
                    add_dist_feature=True)


new_config = {
    'name': 'NE-real-root-weight',  # todo: set this for each experiment!  (default: 'test-run')
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
    'num_epochs': 200,
    'cuda': 2
}
CONFIG.update(new_config)


if __name__ == '__main__':

    run_experiment(CONFIG)
