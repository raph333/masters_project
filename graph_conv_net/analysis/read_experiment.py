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

# MLRUNS_PATH = '/home/rpeer/masters_project/swag_net/graph_conv_net/mlruns'


# def get_experiment_id(experiment_name: str, mlruns_path=MLRUNS_PATH) -> int:
#     for id_ in os.listdir(mlruns_path):
#
#         if id_.isnumeric():
#
#             with open(join(mlruns_path, id_, 'meta.yaml')) as infile:
#                 meta_dict = yaml.safe_load(infile)
#
#             if meta_dict['name'] == experiment_name:
#                 return int(id_)
#
#     raise AssertionError(f'no experiment named "{experiment_name}".')


# def combine_learning_curves(experiment_id: int,
#                             aggregate=np.mean,
#                             save_dir='data',
#                             mlruns_path=MLRUNS_PATH) -> pd.DataFrame():
#
#     experiment_path = join(mlruns_path, str(experiment_id))
#     combined_df = pd.DataFrame()
#
#     # all run-ids have length 32:
#     run_ids = [x for x in os.listdir(experiment_path) if x not in ('meta.yaml', 'data') and len(x) == 32]
#
#     for id_ in run_ids:
#         lc_path = join(experiment_path, id_, 'artifacts/learning_curve.csv')
#         lc_df = pd.read_csv(lc_path)
#         lc_df['run_id'] = id_
#         combined_df = combined_df.append(lc_df, ignore_index=True)
#
#     if aggregate is not None:
#         combined_df = combined_df.groupby('epoch').apply(aggregate).reset_index(drop=True)
#
#     if save_dir is not None:
#         out_dir = join(experiment_path, save_dir)
#         if not os.path.exists(out_dir):
#             os.makedirs(out_dir)
#         combined_df.to_csv(join(out_dir, 'learning_curve.csv'))
#
#     return combined_df


if __name__ == '__main__':
    #id1 = get_experiment_id(mlruns_path='../graph_conv_net/mlruns', experiment_name='mlflow_test')
    # df = combine_learning_curves(id1)

    mlflow.set_tracking_uri('../graph_conv_net/mlruns')
    tracker = mlflow.tracking.MlflowClient()
    exp = tracker.get_experiment_by_name('mlflow_test')
