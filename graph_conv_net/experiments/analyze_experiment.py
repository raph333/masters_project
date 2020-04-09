"""
Reads parameters and artifacts from ml-flow experiment
"""
import os
from os.path import join
import pandas as pd
import mlflow


def average_learning_curve(run_df: pd.DataFrame) -> pd.DataFrame:

    def mean_losses(df: pd.DataFrame) -> pd.DataFrame:
        learning_curves = [pd.read_csv(join(x, 'learning_curve.csv')) for x in df.artifact_uri]
        return pd.concat(learning_curves).groupby('epoch').mean()

    averaged_df = run_df.groupby('params.target_param_value').apply(mean_losses).reset_index()
    averaged_df.epoch = averaged_df.epoch.astype(int)
    averaged_df = averaged_df.rename({'params.target_param_value': 'target_param'}, axis=1)
    return averaged_df


if __name__ == '__main__':

    experiment_name = 'test-run'

    tracker = mlflow.tracking.MlflowClient()
    exp = tracker.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(exp.experiment_id)

    result_df = average_learning_curve(runs)
    result_df.to_csv(f'results/{experiment_name}.csv', index=False)
