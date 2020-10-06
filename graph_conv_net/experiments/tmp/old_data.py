import os.path as osp
import pathlib

import pandas as pd

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

import networkx as nx

import torch
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data, InMemoryDataset


class Complete(object):

    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


class OldAlchemyDataset(InMemoryDataset):
    fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

    def __init__(self,
                 root,
                 mode='dev',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        assert mode in ('dev', 'valid', 'test')
        self.mode = mode
        self.atom_types = ('H', 'C', 'N', 'O', 'F', 'S', 'Cl')
        self.bond_types = (Chem.rdchem.BondType.SINGLE,
                           Chem.rdchem.BondType.DOUBLE,
                           Chem.rdchem.BondType.TRIPLE,
                           Chem.rdchem.BondType.AROMATIC)
        self.hybrid_types = (Chem.rdchem.HybridizationType.SP,
                             Chem.rdchem.HybridizationType.SP2,
                             Chem.rdchem.HybridizationType.SP3)
        self.target = None

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.mode != 'test':
            return [self.mode + '/sdf', self.mode + f'/{self.mode}_target.csv']
        else:
            return [self.mode + '/sdf']

    @property
    def processed_file_names(self):
        return 'TencentAlchemy_%s.pt' % self.mode

    def download(self):
        pass
        # raise NotImplementedError('please download and unzip dataset from %s, and put it at %s' % (_urls[self.mode], self.raw_dir))

    def alchemy_nodes(self, g):
        feat = []
        for n, d in g.nodes(data=True):
            # Atom type (One-hot H, C, N, O, F, ...)
            h_t = [int(d['a_type'] == x) for x in self.atom_types]
            h_t.append(d['a_num'])  # Atomic number  # same thing as type...
            h_t.append(d['acceptor'])
            h_t.append(d['donor'])
            h_t.append(int(d['aromatic']))
            h_t += [int(d['hybridization'] == x) for x in self.hybrid_types]
            h_t.append(d['num_h'])
            feat.append((n, h_t))

        feat.sort(key=lambda item: item[0])
        node_attr = torch.FloatTensor([item[1] for item in feat])
        return node_attr

    def alchemy_edges(self, g):
        """
        One-hot encode edge-types and process for Data object.
        """
        e = {}
        for n1, n2, d in g.edges(data=True):
            e_t = [int(d['b_type'] == x) for x in self.bond_types]
            e[(n1, n2)] = e_t

        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))
        return edge_index, edge_attr

    def sdf_graph_reader(self, sdf_file):
        """
        sdf file reader for Alchemy dataset
        """
        with open(sdf_file, 'r') as f:
            sdf_string = f.read()

        mol = Chem.MolFromMolBlock(sdf_string, removeHs=False)
        if mol is None:
            print("rdkit can not parse file", sdf_file)
            return None

        feats = self.chem_feature_factory.GetFeaturesForMol(mol)
        g = nx.DiGraph()

        if self.mode != 'test':  # store target
            l = torch.FloatTensor(self.target.loc[int(sdf_file.stem)].tolist()).unsqueeze(0)
        else:  # for test: store molecule-id
            l = torch.LongTensor([int(sdf_file.stem)])

        # Create nodes
        assert len(mol.GetConformers()) == 1
        geom = mol.GetConformers()[0].GetPositions()

        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            g.add_node(i,
                       a_type=atom_i.GetSymbol(),
                       a_num=atom_i.GetAtomicNum(),
                       acceptor=0,  # will be overwritten from 'feats'
                       donor=0,  # will be overwritten from 'feats'
                       aromatic=atom_i.GetIsAromatic(),
                       hybridization=atom_i.GetHybridization(),
                       num_h=atom_i.GetTotalNumHs())

        for i in range(len(feats)):

            if feats[i].GetFamily() == 'Donor':
                donor_nodes = feats[i].GetAtomIds()
                for j in donor_nodes:
                    g.node[j]['donor'] = 1

            elif feats[i].GetFamily() == 'Acceptor':  # bug: i has been redefined in inner loop above?
                acceptor_nodes = feats[i].GetAtomIds()
                for j in acceptor_nodes:
                    g.node[j]['acceptor'] = 1

        # Read Edges
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j, b_type=e_ij.GetBondType())

        node_attr = self.alchemy_nodes(g)
        edge_index, edge_attr = self.alchemy_edges(g)
        data = Data(
            x=node_attr,
            pos=torch.FloatTensor(geom),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=l
        )
        setattr(data, 'gbd_idx', int(sdf_file.stem))

        return data

    def process(self):

        if self.mode != 'test':
            num_targets = 12
            target_cols = [f'property_{x}' for x in range(num_targets)]
            self.target = pd.read_csv(self.raw_paths[1], index_col=0)
            self.target = self.target[target_cols]

        sdf_dir = pathlib.Path(self.raw_paths[0])
        data_list = []

        for sdf_file in sdf_dir.glob("**/*.sdf"):
            alchemy_data = self.sdf_graph_reader(sdf_file)
            if alchemy_data is not None:
                data_list.append(alchemy_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])