from typing import Callable
import numpy as np
import torch

from graph_conv_net.train import run_experiment
from graph_conv_net.alchemy_dataset import AlchemyCompetitionDataset
from graph_conv_net.data_processing import TencentDataProcessor
# from graph_conv_net.data_processing import SchNetDataProcessor
from graph_conv_net.transformations import AddEdges
from experiments.distance_threshold import CONFIG

AlchemyCompetitionDataset.data_processor = TencentDataProcessor()
# AlchemyCompetitionDataset.data_processor = SchNetDataProcessor()


def get_transform(threshold: float) -> Callable:
    return AddEdges(distance_threshold=threshold,
                    add_dist_feature=True,
                    norm_dist=False)


new_config = {
    'name': 'test-run',  # todo: set this for each experiment!  (default 'test-run')
    'dataset_class': AlchemyCompetitionDataset,
    'data_processor': TencentDataProcessor,  # SchNetDataProcessor,
    'get_transform': get_transform,
    'target_param': {  # usually a parameter for the transformation
        'name': 'distance_threshold',
        'values': [None]
    },
    # 'lr_scheduler': {
    #     'class': torch.optim.lr_scheduler.ReduceLROnPlateau,
    #     'kwargs': {'factor': 0.75,
    #                'threshold': 1e-4,
    #                'patience': 6}
    # },
    'batch_size': 1,
    'lr': 0.01,  # start high
    'lr_scheduler': {
        'class': torch.optim.lr_scheduler.ExponentialLR,
        'kwargs': {'gamma': 0.97  # ... and decrease quickly
                   }
        # after the test run: figure out a better starting learning rate and decrease more slowly
    },
    'cuda': 3,
    'repeat': 1,
    'num_epochs': 200
}
CONFIG.update(new_config)

if __name__ == '__main__':
    run_experiment(CONFIG)
