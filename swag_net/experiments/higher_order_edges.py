from torch_geometric.transforms import Compose, Distance, Cartesian
from graph_conv_net.alchemy_dataset import FullyConnectedGraph, NHop


def get_transform(num_edges: int):
    transformation = Compose([
        NHop(num_edges),
        Distance(norm=True)
    ])
    return transformation


config = {
    "name": "higher_order_edges",
    "target_param": "edge_hop",
    "edge_hop": [1, 2, 3],
    "get_transform": get_transform,
    "repeat": 3,
    "lr":  0.001,
    "model_name": "test",
    "batch_size": 64,
    "num_epochs": 2,
    "cuda": 1
}