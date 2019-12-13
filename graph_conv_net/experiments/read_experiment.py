"""
Reads parameters and artifacts from ml-flow experiment
"""
import os
from os.path import join
from glob import glob

import yaml
import numpy as np
import pandas as pd

import mlflow


if __name__ == '__main__':
    #id1 = get_experiment_id(mlruns_path='../graph_conv_net/mlruns', experiment_name='mlflow_test')
    # df = combine_learning_curves(id1)

    #mlflow.set_tracking_uri('./mlruns')
    tracker = mlflow.tracking.MlflowClient()
    exp = tracker.get_experiment_by_name('distance_threshold')
    #df = tracker.search_runs()
