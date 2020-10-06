from os.path import join
import argparse
import mlflow
import pandas as pd

from torch_geometric.data import Dataset

from graph_conv_net import tencent_mpnn
from graph_conv_net.train import evaluate_state_dict
from graph_conv_net.alchemy_dataset import AlchemyDataset
from graph_conv_net.data_processing import TencentDataProcessor
from graph_conv_net.tools import split_dataset_by_id

DATA_DIR = '/scratch1/rpeer/tmp'
ID_DF = pd.read_csv('../../old_ds_split.csv')
TST_IDS = set(ID_DF.query('set_ == "test"').gdb_idx)

AlchemyDataset.data_processor = TencentDataProcessor()


def get_sub_set(ds_path: str, ids: set) -> Dataset:
    ds = AlchemyDataset(root=ds_path)
    return split_dataset_by_id(ds, ds_ids=[ids])[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine learning curves from multiple mlflow experiments.')
    parser.add_argument('experiment_name', help='name of experiment')
    args = parser.parse_args()

    tracker = mlflow.tracking.MlflowClient()
    experiment = tracker.get_experiment_by_name(args.experiment_name)
    runs = mlflow.search_runs(experiment.experiment_id)

    ds = AlchemyDataset(root=join(DATA_DIR, 'full-ds-strat-split'))
    ts = split_dataset_by_id(full_ds=ds, ds_ids=[TST_IDS])[0]

    mae = evaluate_state_dict(path=join(runs.artifact_uri.values.item(), 'state_dict.pt'),
                              model=tencent_mpnn.MPNN(),
                              test_set=ts)
    print(mae)
