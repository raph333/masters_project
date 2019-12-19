"""
Reads parameters and artifacts from ml-flow experiment
"""

from graph_conv_net import tencent_mpnn
import os
from os.path import join
from glob import glob

import yaml
import numpy as np
import pandas as pd

import mlflow


if __name__ == '__main__':

    # mlflow.set_tracking_uri('./mlruns')
    tracker = mlflow.tracking.MlflowClient()
    exp = tracker.get_experiment_by_name('distance_threshold')
    print(exp)
    print('output')
    # df = tracker.search_runs()
