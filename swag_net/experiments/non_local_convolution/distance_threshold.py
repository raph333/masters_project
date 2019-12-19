import numpy as np

from torch_geometric.transforms import Compose, Distance, Cartesian

from graph_conv_net.alchemy_dataset import TencentAlchemyDataset
from graph_conv_net.transformations import AddEdges
from graph_conv_net.train import run_experiment


def get_transform(threshold: float):
    transformation = Compose([
        AddEdges(distance_threshold=threshold),
        Distance(norm=True)
    ])
    return transformation


config = {
    "name": "distance_threshold",
    "target_param": "distance_threshold",
    "distance_threshold": [None, *np.arange(1.5, 5.5, 0.5).tolist(), np.inf],
    "dataset_class": TencentAlchemyDataset,
    "get_transform": get_transform,
    "repeat": 3,
    "lr":  0.001,
    "model_name": "tencent_mpnn",
    "batch_size": 64,
    "num_epochs": 1,
    "cuda": 1
}


if __name__ == '__main__':
    run_experiment(config)
