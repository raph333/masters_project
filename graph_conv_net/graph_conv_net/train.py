from os.path import join
from swag_net.alchemy_dataset import AlchemyDataset

# from swag_net.subdir.submodule import func
# import matplotlib.pyplot as plt
# import numpy as np

if __name__ == '__main__':

    DATA_DIR = '/home/rpeer/masters_project/data'
    # download(DATA_DIR)

    ds = AlchemyDataset(join(DATA_DIR, 'valid'), mode='valid')
