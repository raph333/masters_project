import numpy as np
import torch

from graph_conv_net.train import run_experiment
from graph_conv_net.alchemy_dataset import AlchemyCompetitionDataset, TencentDataProcessor
from distance_threshold import CONFIG

AlchemyCompetitionDataset.data_processor = TencentDataProcessor()

new_config = {
    'name': 'test-run',  # todo: set this for each experiment!  (default 'test-run')
    'dataset_class': AlchemyCompetitionDataset,
    'target_param': {
        'name': 'distance_threshold',
        'values': [np.inf]
    },
    'lr_scheduler': {
        'class': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'kwargs': {'factor': 0.75,
                   'threshold': 1e-4,
                   'patience': 10}
    },
    'cuda': 3,
    'repeat': 1,
    'num_epochs': 1
}
CONFIG.update(new_config)


if __name__ == '__main__':

    run_experiment(CONFIG)
