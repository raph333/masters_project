import numpy as np

from torch_geometric.transforms import Compose, Distance, Cartesian

from graph_conv_net.alchemy_dataset import TencentAlchemyDataset
from graph_conv_net.transformations import AddEdges
from graph_conv_net.train import run_experiment


def get_transform(threshold: float):
    return Compose([
        AddEdges(distance_threshold=threshold),
        Distance(norm=True)
    ])


CONFIG = {
    "name": "tencent-mpnn-neighborhood-expansion",  # todo: set this for each experiment!
    "target_param": {
        "name": "distance_threshold",
        "values": [None, 1.5, 2, 3, 4, 5, np.inf]
    },
    "dataset_class": TencentAlchemyDataset,
    "get_transform": get_transform,
    "repeat": 3,
    "lr":  0.001,
    "model_name": "tencent_mpnn",
    "batch_size": 64,
    "num_epochs": 100,
    "cuda": 1
}


if __name__ == '__main__':

    run_experiment(CONFIG)
