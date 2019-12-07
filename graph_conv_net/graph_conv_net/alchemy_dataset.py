import os
from os.path import join
import glob
import shutil
import urllib.request
import zipfile
import typing
from typing import Tuple

import numpy as np
import pandas as pd

import torch
from torch import tensor
from torch_geometric.data import Data, InMemoryDataset, DataLoader

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

SET_NAMES = ('dev', 'valid', 'test')


# def download(data_dir: str,
#              base_url='https://alchemy.tencent.com/data/{}_v20190730.zip',
#              unzip=True):
#     """
#     :param data_dir: download alchemy data-set into this dir. Create if it does not yet exist.
#     :param base_url: base url-format
#     :param unzip: unzip dev-, valid- and test-set and remove zip-archives if True
#     :return: None
#     """
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)
#     print(f'Alchemy dataset-directory: {data_dir}')
#
#     for set_ in SET_NAMES:
#         url = base_url.format(set_)
#         print(f'Downloading {set_}-set from {url} ...')
#         urllib.request.urlretrieve(url, join(data_dir, f'{set_}.zip'))
#     print('Download finished.')
#
#     if unzip:
#         unzip_datasets(data_dir)
#         print('Unzipped datasets.')
#
#
# def unzip_datasets(data_dir: str, remove_zips=True):
#     """
#     Unzip downloaded alchemy data-set
#     """
#     for set_ in SET_NAMES:
#         zip_path = join(data_dir, f'{set_}.zip')
#         assert os.path.exists(zip_path), ''
#
#         print(f'unzipping {zip_path} ...')
#         with zipfile.ZipFile(zip_path) as zip_ref:
#             zip_ref.extractall(data_dir)
#
#         if remove_zips:
#             os.remove(zip_path)


class AlchemyDataset(InMemoryDataset):

    def __init__(self,
                 root,
                 mode='dev',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        assert mode in SET_NAMES, f'Alchemy dataset has only these sets: {SET_NAMES}'  # generalize?
        self.mode = mode

        self.base_url = 'https://alchemy.tencent.com/data/{}_v20190730.zip'
        self.atom_types = np.array(['H', 'C', 'N', 'O', 'F', 'S', 'Cl'])
        self.bond_types = np.array([Chem.rdchem.BondType.SINGLE,
                                    Chem.rdchem.BondType.DOUBLE,
                                    Chem.rdchem.BondType.TRIPLE,
                                    Chem.rdchem.BondType.AROMATIC])
        self.hybrid_types = np.array([Chem.rdchem.HybridizationType.SP,
                                      Chem.rdchem.HybridizationType.SP2,
                                      Chem.rdchem.HybridizationType.SP3])

        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.mode != 'test':
            return ['sdf', f'{self.mode}_target.csv']
        else:
            return ['sdf']

    @property
    def processed_file_names(self):
        return [f'Alchemy_{self.mode}.pt']

    def download(self):
        url = self.base_url.format(self.mode)
        print(f'Downloading {self.mode}-set from {url} ...')

        zip_path = join(self.raw_dir, f'{self.mode}.zip')
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path) as zip_ref:
            zip_ref.extractall(self.raw_dir)

        # move files directly into self.raw_dir:
        source_dir = join(self.raw_dir, self.mode)
        for name in os.listdir(source_dir):
            shutil.move(src=join(source_dir, name),
                        dst=join(self.raw_dir, name))
        os.rmdir(source_dir)

    def get_atom_features(self, molecule: Chem.rdchem.Mol) -> tensor:

        atom_features = []
        for atom in molecule.GetAtoms():
            atom_type = (atom.GetSymbol() == self.atom_types).astype(int)
            aromatic = np.array([atom.GetIsAromatic()]).astype(int)
            hybridization = (atom.GetHybridization() == self.hybrid_types).astype(int)
            atom_feature_vector = np.concatenate([atom_type, aromatic, hybridization])
            atom_features.append(atom_feature_vector)
            # TODO: expand features

        return tensor(atom_features)

    def get_edges(self, molecule: Chem.rdchem.Mol) -> Tuple[tensor, tensor]:

        bonds = []
        bond_features = []
        for bond in molecule.GetBonds():
            bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            bond_type = (bond.GetBondType() == self.bond_types).astype(int)
            bond_features.append(bond_type)

        return tensor(bonds).transpose(0, 1), tensor(bond_features)  # edge-index, edge-attributes

    def process(self):

        if self.mode != 'test':
            target_df = pd.read_csv(self.raw_paths[1])
            target_df = target_df.set_index('gdb_idx')

        graphs = []
        sdf_dir = self.raw_paths[0]

        for sdf_path in glob.glob(f'{sdf_dir}/atom_*/*'):
            with open(sdf_path, 'r') as f:
                molecule = Chem.MolFromMolBlock(f.read(), removeHs=False)
            if molecule is None:
                print(f'Could not parse sdf-file: {sdf_path}')
                return None

            gdb_idx = int(os.path.basename(sdf_path).replace('.sdf', ''))
            if self.mode != 'test':
                y = tensor(target_df.loc[gdb_idx].values).unsqueeze(0)  # store target for train- / validation-set
            else:
                y = tensor([gdb_idx])  # store molecule-id for test-set

            node_attributes = self.get_atom_features(molecule)
            edge_indices, edge_attributes = self.get_edges(molecule)
            coordinates = tensor(molecule.GetConformer().GetPositions())

            # TODO: check tensor types
            graphs.append(
                Data(x=node_attributes,
                     pos=coordinates,
                     edge_index=edge_indices,
                     edge_attr=edge_attributes,
                     y=y)
            )

        if self.pre_filter:
            graphs = [data for data in graphs if self.pre_filter(data)]

        if self.pre_transform:
            graphs = [self.pre_transform(data) for data in graphs]

        combined_data, slices = self.collate(graphs)
        torch.save((combined_data, slices), self.processed_paths[0])

