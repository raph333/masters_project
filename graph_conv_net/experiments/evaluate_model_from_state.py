import argparse
import mlflow


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combine learning curves from multiple mlflow experiments.')
    parser.add_argument('experiment_name', help='name of experiment')
    args = parser.parse_args()

    tracker = mlflow.tracking.MlflowClient()
    experiment = tracker.get_experiment_by_name(args.experiment_name)
    runs = mlflow.search_runs(experiment.experiment_id)

