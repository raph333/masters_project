from typing import Callable
import numpy as np
import torch
from torch_geometric.transforms import Compose, Distance

import sys
sys.path.append('tmp')
from old_data import Complete  # , OldAlchemyDataset

from graph_conv_net.train import run_experiment
# from graph_conv_net.transformations import AddEdges
from graph_conv_net.alchemy_dataset import AlchemyDataset
from graph_conv_net.data_processing import TencentDataProcessor
from distance_threshold import CONFIG

AlchemyDataset.data_processor = TencentDataProcessor()


def get_transform(threshold: float) -> Callable:
    return Compose([
        # AddEdges(distance_threshold=threshold),
        Complete(),
        Distance(norm=True)
    ])
# idea: use Complete() and then remove edges with distance > threshold


new_config = {
    # full-ds-tencent-features-complete
    'name': 'test-run',  # todo: set this for each experiment!  (default 'test-run')
    'dataset_class': AlchemyDataset,
    'data_processor': TencentDataProcessor,
    'get_transform': get_transform,
    'target_param': {  # usually a parameter for the transformation
        'name': 'distance_threshold',
        'values': [np.inf]
    },
    'lr_scheduler': {
        'class': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'kwargs': {'factor': 0.75,
                   'threshold': 1e-4,
                   'patience': 5}
    },
    'cuda': 0,
    'repeat': 1,
    'num_epochs': 300
}
CONFIG.update(new_config)


if __name__ == '__main__':

    run_experiment(CONFIG)
