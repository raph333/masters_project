from typing import Callable
import numpy as np
import torch

from torch_geometric.transforms import Compose, LocalCartesian, Center

from graph_conv_net.train import run_experiment
from graph_conv_net.alchemy_dataset import AlchemyCompetitionDataset
from graph_conv_net.data_processing import RawDataProcessor
# from graph_conv_net.data_processing import SchNetDataProcessor
from graph_conv_net.transformations import AddEdges
from experiments.distance_threshold import CONFIG

AlchemyCompetitionDataset.data_processor = RawDataProcessor(implicit_hydrogens=False)
# AlchemyCompetitionDataset.data_processor = SchNetDataProcessor()


def get_transform(threshold: float) -> Callable:
    return Compose([
        AddEdges(distance_threshold=threshold,
                 add_dist_feature=True,
                 norm_dist=False),
        Center(),
        LocalCartesian()
    ])


new_config = {
    'name': 'test-run',  # todo: set this for each experiment!  (default 'test-run')
    'root_weight': True,
    'dataset_class': AlchemyCompetitionDataset,
    'data_processor': RawDataProcessor,  # SchNetDataProcessor,
    'get_transform': get_transform,
    'target_param': {  # usually a parameter for the transformation
        'name': 'distance_threshold',
        'values': [np.inf]
    },
    'lr': 0.002,  # start a bit higher
    'lr_scheduler': {
        'class': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'kwargs': {'factor': 0.75,
                   'threshold': 1e-4,
                   'patience': 6}
    },
    'batch_size': 32,
    # 'lr': 0.001,
    # 'lr_scheduler': {
    #     'class': torch.optim.lr_scheduler.ExponentialLR,
    #     'kwargs': {'gamma': 0.995}
    # },
    'repeat': 1,
    'cuda': 3,
    'num_epochs':  500
}
CONFIG.update(new_config)


if __name__ == '__main__':
    run_experiment(CONFIG)
