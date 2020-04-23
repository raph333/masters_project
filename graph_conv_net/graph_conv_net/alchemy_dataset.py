import os
from os.path import join
import glob
import shutil
import urllib.request
import zipfile
from typing import Union, Tuple, Callable

import numpy as np
import pandas as pd

import torch
from torch import tensor

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.makedirs import makedirs


from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

SET_NAMES = ('dev', 'valid', 'test')


class BaseAlchemyDataset(InMemoryDataset):

    def __init__(self,
                 root: str,
                 mode: str = 'dev',
                 transform: Union[Callable, None] = None,
                 pre_transform: Union[Callable, None] = None,
                 pre_filter: Union[Callable, None] = None,
                 re_process: bool = False):

        assert mode in SET_NAMES, f'Alchemy dataset has only these sets: {SET_NAMES}'
        self.mode = mode
        self.re_process = re_process

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

    def _process(self):

        if self.re_process is False and all([os.path.exists(f) for f in self.processed_paths]):
            return

        print('Processing...')
        makedirs(self.processed_dir)
        self.process()
        print('Done!')

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
        """
        Simplest atom-features: only atom-type
        """
        atom_features = []

        for atom in molecule.GetAtoms():
            atom_type = (atom.GetSymbol() == self.atom_types).astype(int)
            atom_features.append(atom_type)

        # requires float for linear layer and long for embedding
        return tensor(atom_features).float()

    def get_bonds(self, molecule: Chem.rdchem.Mol) -> Tuple[tensor, tensor]:
        """
        :return: bond-index, bond-features
        """
        bonds = []
        bond_features = []

        for bond in molecule.GetBonds():
            bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            bond_type = (bond.GetBondType() == self.bond_types).astype(int)
            bond_features.append(bond_type)

        return tensor(bonds).transpose(0, 1).long(), tensor(bond_features).float()

    def read_sdf(self,
                 sdf_path: str,
                 target_df: Union[pd.DataFrame, None] = None) -> Union[Data, None]:

        with open(sdf_path, 'r') as f:
            molecule = Chem.MolFromMolBlock(f.read(), removeHs=True)
        if molecule is None:
            print(f'Could not parse sdf-file: {sdf_path}')
            return None

        gdb_idx = int(os.path.basename(sdf_path).replace('.sdf', ''))
        if target_df is not None:
            y = tensor(target_df.loc[gdb_idx].values).unsqueeze(0)  # store target for train- / validation-set
        else:
            y = tensor([gdb_idx])  # store molecule-id for test-set

        node_attributes = self.get_atom_features(molecule)
        bond_indices, bond_features = self.get_bonds(molecule)
        coordinates = tensor(molecule.GetConformer().GetPositions()).float()

        graph = Data(x=node_attributes,
                     pos=coordinates,
                     edge_index=bond_indices,
                     edge_attr=bond_features,
                     y=y.float())

        return graph

    def process(self):

        if self.mode != 'test':
            target_df = pd.read_csv(self.raw_paths[1])
            target_df = target_df.set_index('gdb_idx')
        else:
            target_df = None

        graphs = []
        sdf_dir = self.raw_paths[0]

        for sdf_path in glob.glob(f'{sdf_dir}/atom_*/*'):

            graph = self.read_sdf(sdf_path, target_df)
            if graph is not None:
                graphs.append(graph)

        if self.pre_filter is not None:
            graphs = [data for data in graphs if self.pre_filter(data)]

        if self.pre_transform is not None:
            graphs = [self.pre_transform(data) for data in graphs]

        combined_data, slices = self.collate(graphs)
        torch.save((combined_data, slices), self.processed_paths[0])


class TencentAlchemyDataset(BaseAlchemyDataset):
    """
    Using the exact same features as in http://arxiv.org/abs/1906.09427
    """

    @staticmethod
    def get_donors_acceptors(molecule: Chem.rdchem.Mol) -> tuple:

        feature_definition = join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(feature_definition)
        molecule_features = chem_feature_factory.GetFeaturesForMol(molecule)

        donor_atoms, acceptor_atoms = tuple(), tuple()

        for feature in molecule_features:
            if feature.GetFamily() == 'Donor':
                donor_atoms = feature.GetAtomIds()
            elif feature.GetFamily() == 'Acceptor':
                acceptor_atoms = feature.GetAtomIds()

        return donor_atoms, acceptor_atoms

    def get_atom_features(self, molecule: Chem.rdchem.Mol) -> tensor:

        donor_atom_ids, acceptor_atom_ids = self.get_donors_acceptors(molecule)
        atom_features = []

        for atom in molecule.GetAtoms():
            atom_type = (atom.GetSymbol() == self.atom_types).astype(int)
            aromatic = np.array([atom.GetIsAromatic()]).astype(int)
            hybridization = (atom.GetHybridization() == self.hybrid_types).astype(int)
            atom_feature_vector = np.concatenate([
                atom_type,
                np.array([atom.GetAtomicNum()]),  # redundant info: same as above
                np.array([int(atom.GetIdx() in acceptor_atom_ids)]),
                np.array([int(atom.GetIdx() in donor_atom_ids)]),
                aromatic,
                hybridization,
                np.array([atom.GetTotalNumHs()])
            ])
            atom_features.append(atom_feature_vector)

        return tensor(atom_features).float()
