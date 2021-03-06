import os
from os.path import join
import shutil
import urllib.request
import zipfile
import time
from typing import Union, Callable

import pandas as pd

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.makedirs import makedirs


class AlchemyDataset(InMemoryDataset):
    """
    Entire 200k molecule Alchemy dataset: fully labeled, no pre-defined split
    See https://alchemy.tencent.com/
    """
    labels_file_name = 'ground_truth.csv'
    atom_numbers = (9, 10, 11, 12)

    def __init__(self,
                 root: str,
                 transform: Union[Callable, None] = None,
                 pre_transform: Union[Callable, None] = None,
                 pre_filter: Union[Callable, None] = None,
                 re_process: bool = False,
                 sample_fraction: float = 1):

        if not hasattr(self, 'data_processor'):
            print('Please specify an instance of the "BasicDataProcessor"-class as class-attribute.')
            raise AssertionError('Missing attribute: "data_processor"')

        self.re_process = re_process
        self.sample_fraction = sample_fraction
        self.url = 'https://alchemy.tencent.com/data/alchemy-v20191129.zip'
        self.target_df = None

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def _process(self):

        if self.re_process is False and all([os.path.exists(f) for f in self.processed_paths]):
            return

        print('Processing...')
        start = time.time()
        makedirs(self.processed_dir)
        self.process()
        print(f'Done! ({(time.time() - start) / 60:.2f} minutes)')

    @property
    def raw_file_names(self) -> list:
        structure_dirs = [f'atom_{n}' for n in self.atom_numbers]
        return [*structure_dirs, self.labels_file_name]

    @property
    def processed_file_names(self) -> list:
        return [f'processed.pt']

    def download(self):
        print(f'Downloading data-set from {self.url} ...')

        zip_file_name = os.path.basename(self.url)
        zip_path = join(self.raw_dir, zip_file_name)
        urllib.request.urlretrieve(self.url, zip_path)

        print('Extracting zip file ...')
        with zipfile.ZipFile(zip_path) as zip_ref:
            zip_ref.extractall(self.raw_dir)
            extracted_dir_name = zip_ref.infolist()[0].filename

        # move files directly into self.raw_dir:
        source_dir = join(self.raw_dir, extracted_dir_name)
        for name in os.listdir(source_dir):
            if name.endswith('.csv'):
                new_name = self.labels_file_name
            else:
                new_name = name
            shutil.move(src=join(source_dir, name),
                        dst=join(self.raw_dir, new_name))

        os.rmdir(source_dir)
        os.remove(zip_path)

    def process(self):
        target_df = pd.read_csv(join(self.raw_dir, self.labels_file_name))
        target_df = target_df.set_index('gdb_idx').drop('atom number', axis=1)
        target_df = target_df.apply(lambda col: (col - col.mean()) / col.std())

        n = int(len(target_df) * self.sample_fraction)
        self.target_df = target_df.sample(n)

        graphs = self.data_processor.get_graphs(structures_dir=self.raw_dir,
                                                target_df=self.target_df,
                                                pre_filter=self.pre_filter,
                                                pre_transform=self.pre_transform)

        combined_data, slices = self.collate(graphs)
        torch.save((combined_data, slices), self.processed_paths[0])


class AlchemyCompetitionDataset(InMemoryDataset):
    """
    Competition dataset with pre-defined dev-, validation- and unlabeled test-set.
    See https://alchemy.tencent.com/
    """
    set_names = ('dev', 'valid', 'test')

    def __init__(self,
                 root: str,
                 mode: str = 'dev',
                 transform: Union[Callable, None] = None,
                 pre_transform: Union[Callable, None] = None,
                 pre_filter: Union[Callable, None] = None,
                 re_process: bool = False):

        if not hasattr(self, 'data_processor'):
            print('Please specify an instance of the "BasicDataProcessor"-class as class-attribute.')
            raise AssertionError('Missing attribute: "data_processor"')

        assert mode in self.set_names, f'Alchemy dataset has only these sets: {self.set_names}'
        self.mode = mode
        self.re_process = re_process
        self.base_url = 'https://alchemy.tencent.com/data/{}_v20190730.zip'
        self.target_df = None

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def _process(self):
        if self.re_process is False and all([os.path.exists(f) for f in self.processed_paths]):
            return

        print('Processing...')
        makedirs(self.processed_dir)
        self.process()
        print('Done!')

    @property
    def raw_file_names(self) -> list:
        sdf_dir = join(self.mode, 'sdf')
        if self.mode != 'test':
            return [sdf_dir, join(self.mode, f'{self.mode}_target.csv')]
        else:
            return [sdf_dir]

    @property
    def processed_file_names(self) -> list:
        return [f'Alchemy_{self.mode}.pt']

    def download(self):
        url = self.base_url.format(self.mode)
        print(f'Downloading {self.mode}-set from {url} ...')

        zip_path = join(self.raw_dir, f'{self.mode}.zip')
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path) as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(zip_path)

    def process(self):
        if self.mode != 'test':
            target_df = pd.read_csv(self.raw_paths[1])
            self.target_df = target_df.set_index('gdb_idx')

        graphs = self.data_processor.get_graphs(structures_dir=self.raw_paths[0],
                                                target_df=self.target_df,
                                                pre_filter=self.pre_filter,
                                                pre_transform=self.pre_transform)

        combined_data, slices = self.collate(graphs)
        torch.save((combined_data, slices), self.processed_paths[0])
