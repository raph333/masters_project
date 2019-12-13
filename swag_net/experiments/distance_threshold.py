from torch_geometric.transforms import Compose, Distance, Cartesian
from graph_conv_net.alchemy_dataset import FullyConnectedGraph, AddEdges
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
    "distance_threshold": [1, 2, 3, None],
    "get_transform": get_transform,
    "repeat": 3,
    "lr":  0.001,
    "model_name": "test",
    "batch_size": 64,
    "num_epochs": 2,
    "cuda": 1
}


if __name__ == '__main__':
    run_experiment(config)
