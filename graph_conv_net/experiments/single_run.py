import numpy as np
import torch

from graph_conv_net.train import run_experiment
from distance_threshold import CONFIG

new_config = {
    'name': 'test-run',  # 'best-optimization-patience-10'  # todo: set this for each experiment!  (default 'test-run')
    'target_param': {
        'name': 'distance_threshold',
        'values': [np.inf]},
    'lr_scheduler': {
        'class': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'kwargs': {'factor': 0.75,
                   'threshold': 1e-4,
                   'patience': 10}},
    'cuda': 0,
    'repeat': 1,
    'num_epochs': 1
}
CONFIG.update(new_config)


if __name__ == '__main__':

    run_experiment(CONFIG)
