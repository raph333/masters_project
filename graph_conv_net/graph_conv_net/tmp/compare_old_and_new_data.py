from os.path import join
from tqdm import tqdm
import numpy as np
from typing import Union

from torch_geometric.transforms import Compose, Distance
from torch_geometric.data import Data

from graph_conv_net.alchemy_dataset import AlchemyDataset
from graph_conv_net.data_processing import TencentDataProcessor
from graph_conv_net.transformations import AddEdges, CompleteGraph
from experiments.tmp.old_data import OldAlchemyDataset, Complete

AlchemyDataset.data_processor = TencentDataProcessor(sample_fraction=1)


def get_mol(dataset, gdb_index: int):
    for d in dataset:
        if d.gdb_idx.item() == gdb_index:
            return d


def get_atom_pairs(edge_index):
    return set(zip(edge_index[0].numpy(), edge_index[1].numpy()))


def print_tencent_edges(molecule):
    edges = []
    for i in range(molecule.GetNumAtoms()):
        for j in range(molecule.GetNumAtoms()):
            e_ij = molecule.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                # g.add_edge(i, j, b_type=e_ij.GetBondType())
                bond = e_ij.GetBondType()
                edges.append((i, j, bond))
            else:
                bond = None
            print(f'atoms {i}--{j}: {bond}')


def assert_equality(a: Data,
                    b: Data,
                    attributes: Union[tuple, str] = ('edge_index', 'edge_attr', 'x', 'pos')):
    assert isinstance(attributes, tuple) or attributes == 'all'

    for (a_name, a_value), (b_name, b_value) in zip(a, b):
        assert a_name == b_name
        assert a_value.shape == b_value.shape

        if attributes == 'all' or a_name in attributes:
            if a_value is None and b_value is None:
                continue
            assert (a_value == b_value).all().item(), f'Difference in attribute: {a_name}'

    # for attr_name in attributes:
    #     assert (getattr(a, attr_name) == getattr(b, attr_name)).all().item(), f'Difference in attribute: {attr_name}'


def compare_data(ds_new, ds_old, n_compares=10):
    count = 0

    for n in ds_new:

        o = get_mol(ds_old, n.gbd_idx.item())
        if o is None:
            continue

        for attr_o, attr_n in zip(o, n):
            assert attr_o[0] == attr_n[0]
            assert attr_o[1].shape == attr_n[1].shape

        assert (o.edge_attr.sum(dim=0) == n.edge_attr.sum(dim=0)).numpy().all()
        assert (n.x.sum(dim=0) == o.x.sum(dim=0)).numpy()[:8].all()
        assert (n.x.sum(dim=0) == o.x.sum(dim=0)).numpy()[10:].all()
        assert (n.x.sum(dim=0) == o.x.sum(dim=0)).numpy().all()

        count += 1
        print(f'{count} / {n_compares}')

        if count == n_compares:
            print(f'Successfully compared {n_compares} molecules: no difference')
            return


def check_transformation_equality(n=10):

    for _ in tqdm(range(n)):
        idx = np.random.randint(low=0, high=len(new) - 1)

        new.transform = Compose([Complete(), Distance(norm=True)])
        old_trans = new[idx]

        new.transform = Compose([CompleteGraph(), Distance(norm=True)])
        comp = new[idx]

        new.transform = Compose([AddEdges(distance_threshold=np.inf), Distance(norm=True)])
        ae = new[idx]

        new.transform = AddEdges(distance_threshold=np.inf, add_dist_feature=True)
        ae2 = new[idx]

        assert_equality(old_trans, comp)
        assert_equality(comp, ae)
        assert_equality(ae, ae2)

    print(f'Checked equality for {n} molecules: success.')


if __name__ == '__main__':

    old = OldAlchemyDataset(root='/scratch1/rpeer/tmp/old_data', mode='dev')
    print('initialized old dataset')

    NEW_DATA_DIR = '/scratch1/rpeer/tmp/test-new-data'
    new = AlchemyDataset(root=NEW_DATA_DIR, re_process=False)
    print('initialized new dataset')

    # IDX = 1000170
    # new_sdf_path = join(NEW_DATA_DIR, 'raw', 'atom_10', f'{IDX}.sdf')  # present in old data-set as well
    # new_df = pd.read_csv(join(NEW_DATA_DIR, 'raw', 'ground_truth.csv')).set_index('gdb_idx')
    # new_mol = AlchemyDataset.data_processor._read_sdf(sdf_path=new_sdf_path,
    #                                                   target_df=new_df)
    # old_mol = get_mol(old, gdb_index=IDX)
    # print(old_mol)
    # print(new_mol)

    # compare_data(new, old, n_compares=10)

    # COMPARE TRANSFORMATIONS
    check_transformation_equality(n=100)
