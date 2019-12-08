from os.path import join
from graph_conv_net.alchemy_dataset import AlchemyDataset, FullyConnected

# from swag_net.subdir.submodule import func
# import matplotlib.pyplot as plt
# import numpy as np

if __name__ == '__main__':

    DATA_DIR = '/home/rpeer/masters_project/data'

    ds = AlchemyDataset(join(DATA_DIR, 'valid'), mode='valid')
    ds_dev = AlchemyDataset(join(DATA_DIR, 'dev'), mode='dev')
    ds_test = AlchemyDataset(join(DATA_DIR, 'test'), mode='test')

    data = ds[0]

    fc = FullyConnected()
    data_ful = fc(data)
