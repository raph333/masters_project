import numpy as np
import tqdm

from graph_conv_net.data_processing import TencentDataProcessor
from graph_conv_net.alchemy_dataset import AlchemyCompetitionDataset, AlchemyDataset
from graph_conv_net.tools import stratified_data_split
from graph_conv_net.tmp.compare_old_and_new_data import assert_equality, get_mol

AlchemyDataset.data_processor = TencentDataProcessor()
AlchemyCompetitionDataset.data_processor = TencentDataProcessor()

if __name__ == '__main__':
    # comp = AlchemyCompetitionDataset(root='/scratch1/rpeer/tmp/competition_data', mode='valid')
    # print('initialized competition dataset.')

    full = AlchemyDataset(root='/scratch1/rpeer/tmp/full_data', re_process=False, sample_fraction=0.01)
    print('initialized full dataset.')

    splits = stratified_data_split(full_ds=full,
                                   strat_col=2)

    # for i in tqdm.tqdm(np.random.randint(low=0, high=len(comp), size=100)):
    #     c = comp[int(i)]
    #     f = get_mol(full, gdb_index=c.gdb_idx)
    #     assert_equality(c, f, attributes=tuple(a for a in comp.__dict__.keys() if a != 'y'))
