import numpy as np
import torch

from graph_conv_net.train import run_experiment
from distance_threshold import CONFIG

CONFIG['name'] = 'best-optimization-patience-10'  # todo: set this for each experiment!  (default 'test-run')
CONFIG['target_param'] = {
       'name': 'distance_threshold',
       'values': [np.inf]}
CONFIG['lr_scheduler'] = {
        'class': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'kwargs': {'factor': 0.75,
                   'threshold': 1e-4,
                   'patience': 5}}
CONFIG['cuda'] = 2
CONFIG['repeat'] = 1
CONFIG['num_epochs'] = 300


if __name__ == '__main__':

    run_experiment(CONFIG)
