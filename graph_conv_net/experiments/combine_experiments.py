"""
Aggregate learning curves from different experiments.
Assumption: only one set of parameters per experiment => all runs from an experiment (if there are more than one) can
be averaged.
"""
from os.path import join
import argparse
import pandas as pd
import mlflow


def combine_learning_curves(experiment_names: str) -> pd.DataFrame:
    tracker = mlflow.tracking.MlflowClient()
    combined_df = pd.DataFrame()

    for experiment_name in experiment_names.split(','):

        experiment = tracker.get_experiment_by_name(experiment_name)
        assert experiment is not None, f'No experiment with name {experiment_name}'
        runs = mlflow.search_runs(experiment.experiment_id)
        assert len(runs) > 0

        if len(runs) > 1:
            print('Multiple runs for one experiment.')
            print('Remove invalid runs. If multiple valid runs exist: implement averaging.')
            raise AssertionError

        lc_df = pd.read_csv(join(runs.artifact_uri.values.item(), 'learning_curve.csv'))
        lc_df['experiment_name'] = experiment_name
        combined_df = pd.concat([combined_df, lc_df], ignore_index=True)

    return combined_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combine learning curves from multiple mlflow experiments.')
    parser.add_argument('experiment_names', help='csv list of mlflow experiment-names')
    parser.add_argument('--output-name', '-o',
                        default='combined-experiments')
    args = parser.parse_args()

    experiment_names = args.experiment_names.split(',')
    if len(experiment_names) == 1 and args.output_name == 'combined-experiments':
        args.output_name = experiment_names[0]

    result = combine_learning_curves(args.experiment_names)
    result.to_csv(join('results', f'{args.output_name}.csv'), index=False)
