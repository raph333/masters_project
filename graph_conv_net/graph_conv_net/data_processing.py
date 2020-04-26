import os
from os.path import join
import glob
from typing import Tuple, Union

import numpy as np
import pandas as pd

from torch import tensor
from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig


class RawDataProcessor:

    atom_types = np.array(['H', 'C', 'N', 'O', 'F', 'S', 'Cl'])
    bond_types = np.array([Chem.rdchem.BondType.SINGLE,
                           Chem.rdchem.BondType.DOUBLE,
                           Chem.rdchem.BondType.TRIPLE,
                           Chem.rdchem.BondType.AROMATIC])

    def __init__(self, implicit_hydrogens: bool = False):
        self.implicit_h = implicit_hydrogens

        if self.implicit_h:
            self.atom_types = self.atom_types[1:]

    def __repr__(self):
        return f'{self.__class__.__name__}(implicit_h={self.implicit_h})'

    def _get_atom_features(self, molecule: Chem.rdchem.Mol) -> tensor:
        """
        Simplest atom-features: only atom-type
        """
        features = []

        for atom in molecule.GetAtoms():

            atom_type = (atom.GetSymbol() == self.atom_types).astype(int)
            features.append(atom_type)

        # requires float for linear layer and long for embedding
        return tensor(features).float()

    def _get_bonds(self, molecule: Chem.rdchem.Mol) -> Tuple[tensor, tensor]:
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

    def _read_sdf(self,
                  sdf_path: str,
                  target_df: Union[pd.DataFrame, None] = None) -> Union[Data, None]:

        with open(sdf_path) as infile:
            molecule = Chem.MolFromMolBlock(infile.read(), removeHs=self.implicit_h)
        if molecule is None:
            print(f'Could not parse sdf-file: {sdf_path}')
            return None

        gdb_idx = int(os.path.basename(sdf_path).replace('.sdf', ''))
        if target_df is not None:
            y = tensor(target_df.loc[gdb_idx].values).unsqueeze(0)  # store target for train- / validation-set
        else:
            y = tensor([gdb_idx])  # store molecule-id for unlabeled test-set

        node_attributes = self._get_atom_features(molecule)
        bond_indices, bond_features = self._get_bonds(molecule)
        coordinates = tensor(molecule.GetConformer().GetPositions()).float()

        graph = Data(x=node_attributes,
                     pos=coordinates,
                     edge_index=bond_indices,
                     edge_attr=bond_features,
                     y=y.float())

        return graph

    def get_graphs(self,
                   structures_dir: str,
                   target_df: pd.DataFrame,
                   pre_filter,
                   pre_transform) -> list:
        graphs = []

        for sdf_path in glob.glob(f'{structures_dir}/atom_*/*'):

            graph = self._read_sdf(sdf_path, target_df)
            if graph is not None:
                graphs.append(graph)

        if pre_filter is not None:
            graphs = [data for data in graphs if pre_filter(data)]

        if pre_transform is not None:
            graphs = [pre_transform(data) for data in graphs]

        return graphs


class TencentDataProcessor(RawDataProcessor):
    """
    Using the exact same features as in http://arxiv.org/abs/1906.09427
    """
    hybridization_types = np.array([Chem.rdchem.HybridizationType.SP,
                                    Chem.rdchem.HybridizationType.SP2,
                                    Chem.rdchem.HybridizationType.SP3])

    @staticmethod
    def _get_donors_acceptors(molecule: Chem.rdchem.Mol) -> tuple:

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

    def _get_atom_features(self, molecule: Chem.rdchem.Mol) -> tensor:

        donor_atom_ids, acceptor_atom_ids = self._get_donors_acceptors(molecule)
        features = []

        for atom in molecule.GetAtoms():
            atom_type = (atom.GetSymbol() == self.atom_types).astype(int)
            aromatic = np.array([atom.GetIsAromatic()]).astype(int)
            hybridization = (atom.GetHybridization() == self.hybridization_types).astype(int)
            atom_feature_vector = np.concatenate([
                atom_type,
                np.array([atom.GetAtomicNum()]),  # redundant info: same as above
                np.array([int(atom.GetIdx() in acceptor_atom_ids)]),
                np.array([int(atom.GetIdx() in donor_atom_ids)]),
                aromatic,
                hybridization,
                np.array([atom.GetTotalNumHs()])
            ])
            features.append(atom_feature_vector)

        return tensor(features).float()


# if __name__ == '__main__':
#
#     LETTERS = []
#     explicit = []
#     implicit = []
#     total = []
#     p = RawDataProcessor(implicit_hydrogens=True)
#     # g = p._read_sdf('../../data_full/raw/atom_10/8273173.sdf')
#
#     graphs = []
#     for file_name in glob.glob('../../data_full/raw/atom_10/*.sdf'):
#         graph = p._read_sdf(file_name)
#         graphs.append(graph)
#
#     h_types = pd.DataFrame({'explicit': explicit,
#                             'implicit': implicit,
#                             'total': total})
#     h_types['sum_ok'] = (h_types.explicit + h_types.implicit) == h_types.total
#
#     atom_counts = pd.Series([g.num_nodes for g in graphs]).value_counts()
#     print(atom_counts)
#
#     print(pd.Series(LETTERS).value_counts())

