import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple

from sklearn.model_selection import train_test_split

import torch
from torch import nn, tensor, LongTensor
from torch.utils.data.dataset import Dataset, Subset


def count_parameters(net: nn.Module):
    return sum([np.prod(x.shape) for x in net.parameters()])


def print_lr_schedule(lr: float, decay: float, num_epochs=100, steps=10):
    print('\nlearning-rate schedule:')
    for i in range(num_epochs):
        if i % steps == 0:
            print(f'{i}\t{lr:.6f}')
        lr = lr * decay


def plot_error_curves(training_error: list,
                      validation_error: list,
                      error_name='error',
                      plot_name='learning_curve',
                      y_limit=None,
                      save_fig=True):
    assert len(training_error) == len(validation_error) > 1

    fig, ax = plt.subplots()
    ax.plot(range(len(training_error)), training_error)
    ax.plot(range(len(validation_error)), validation_error)

    if y_limit:
        ax.set_ylim(*y_limit)

    ax.set_xlabel('epoch')
    ax.set_ylabel(error_name)
    ax.legend(('training', 'validation'))
    ax.set_title(f'{error_name} over time')

    if save_fig:
        fig.savefig(f'{plot_name}.png', bbox_inches='tight', transparent=True)

    plt.show()


def split_dataset_by_id(full_ds: Dataset,
                        ds_ids: List[set],
                        id_attr: str = 'gdb_idx') -> List[Dataset]:
    data_sets = []

    for ids in ds_ids:
        ds_indices = [i for i in range(len(full_ds)) if full_ds[i][id_attr].item() in ids]
        ds = Subset(dataset=full_ds,
                    indices=ds_indices)
        data_sets.append(ds)

    assert [len(ids) for ids in ds_ids] == [len(ds) for ds in data_sets]
    return data_sets


def random_data_split(full_ds: Dataset,
                      random_seed: Union[int, None] = None,
                      fractions: Union[tuple, None] = None,
                      lengths: Union[tuple, None] = None) -> List[Dataset]:
    n = len(full_ds)

    if fractions is not None:
        assert sum(fractions) == 1
        lengths = [int(f * n) for f in fractions]
    elif lengths is not None:
        assert sum(lengths) == n
    else:
        raise AssertionError('Please provide either fractions or lengths.')

    r_state = np.random.RandomState(random_seed)
    permuted_indices = r_state.permutation(n)

    start = 0
    data_sets = []
    for n in lengths:
        end = start + n
        ds = Subset(dataset=full_ds,
                    indices=permuted_indices[start:end].tolist())
        data_sets.append(ds)
        start = end

    return data_sets


def stratified_data_split(ds: Dataset,
                          strat_col: int = 2,
                          bin_size: int = 10,
                          parts: tuple = (8, 1, 1),
                          random_seed: Union[int, None] = None) -> tuple:
    assert len(parts) == 3, 'Please provide train-, validation- and test-set parts (can be zero).'
    assert sum(parts) == bin_size, 'Make sure the ratios sum up to bin-size.'

    r_state = np.random.RandomState(random_seed)
    targets = torch.cat([d.y for d in ds])[:, strat_col]
    _, indices = targets.sort()
    trn_idx = []
    val_idx = []
    tst_idx = []

    for i in range(targets.shape[0] // bin_size):
        start = i * bin_size
        end = start + bin_size
        bin_idx = indices[start: end]

        trn, val, tst = torch.split(tensor(r_state.permutation(bin_size)), parts)
        trn_idx.extend(bin_idx[trn].tolist())
        val_idx.extend(bin_idx[val].tolist())
        tst_idx.extend(bin_idx[tst].tolist())

    return Subset(ds, trn_idx), Subset(ds, val_idx), Subset(ds, tst_idx)
