import os
from os.path import join
import glob
import shutil
import urllib.request
import zipfile
import typing
from typing import Tuple
from itertools import product

import numpy as np
import pandas as pd

import torch
from torch import tensor
from torch_geometric.data import Data, InMemoryDataset, DataLoader

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

SET_NAMES = ('dev', 'valid', 'test')


class FullyConnected(object):

    def __call__(self, graph: Data) -> Data:
        num_bond_features = graph.edge_attr.shape[1]
        edge_features = torch.zeros(graph.num_nodes**2, num_bond_features).long()

        bond_index = graph.edge_index[0] * graph.num_nodes + graph.edge_index[1]
        edge_features[bond_index] = graph.edge_attr

        all_edges = list(product(range(graph.num_nodes), range(graph.num_nodes)))
        full_edge_index = np.array(all_edges).transpose()

        graph.edge_attr = edge_features
        graph.edge_index = full_edge_index

        return graph


class AlchemyDataset(InMemoryDataset):

    def __init__(self,
                 root,
                 mode='dev',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 fully_connected=False):

        assert mode in SET_NAMES, f'Alchemy dataset has only these sets: {SET_NAMES}'  # generalize?
        self.mode = mode
        self.fully_connected = fully_connected

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
    def raw_file_names(self) -> list:
        if self.mode != 'test':
            return ['sdf', f'{self.mode}_target.csv']
        else:
            return ['sdf']

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

    def get_bonds(self, molecule: Chem.rdchem.Mol) -> Tuple[tensor, tensor]:

        bonds = []
        bond_features = []
        for bond in molecule.GetBonds():
            bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            bond_type = (bond.GetBondType() == self.bond_types).astype(int)
            bond_features.append(bond_type)

        return tensor(bonds).transpose(0, 1), tensor(bond_features)  # bond-indices, bond-features

    def read_sdf(self, sdf_path: str, target_df: pd.DataFrame) -> Data:

        with open(sdf_path, 'r') as f:
            molecule = Chem.MolFromMolBlock(f.read(), removeHs=True)
        if molecule is None:
            print(f'Could not parse sdf-file: {sdf_path}')
            return None

        gdb_idx = int(os.path.basename(sdf_path).replace('.sdf', ''))
        if self.mode != 'test':
            y = tensor(target_df.loc[gdb_idx].values).unsqueeze(0)  # store target for train- / validation-set
        else:
            y = tensor([gdb_idx])  # store molecule-id for test-set

        node_attributes = self.get_atom_features(molecule)
        bond_indices, bond_features = self.get_bonds(molecule)
        coordinates = tensor(molecule.GetConformer().GetPositions())

        # TODO: check tensor types
        graph = Data(x=node_attributes,
                     pos=coordinates,
                     edge_index=bond_indices,
                     edge_attr=bond_features,
                     y=y)

        if self.fully_connected:
            graph = self.make_fully_connected_graph(graph)

        return graph

    def process(self):

        if self.mode != 'test':
            target_df = pd.read_csv(self.raw_paths[1])
            target_df = target_df.set_index('gdb_idx')

        graphs = []
        sdf_dir = self.raw_paths[0]

        for sdf_path in glob.glob(f'{sdf_dir}/atom_*/*'):
            graphs.append(self.read_sdf(sdf_path, target_df))

        if self.pre_filter:
            graphs = [data for data in graphs if self.pre_filter(data)]

        if self.pre_transform:
            graphs = [self.pre_transform(data) for data in graphs]

        combined_data, slices = self.collate(graphs)
        torch.save((combined_data, slices), self.processed_paths[0])

