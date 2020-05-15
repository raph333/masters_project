import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List

from torch import nn
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


def random_split_dataset(full_ds: Dataset,
                         random_seed: int,
                         fractions: Union[tuple, None],
                         lengths: Union[tuple, None] = None) -> List[Dataset]:
    if fractions is not None:
        assert sum(fractions) == 1
        lengths = [int(f * len(full_ds)) for f in fractions]
    elif lengths is not None:
        assert sum(lengths) == len(full_ds)
    else:
        raise AssertionError('Please provide either fractions or lengths.')

    np.random.seed(random_seed)
    permuted_indices = np.random.choice(range(len(full_ds)), size=len(full_ds), replace=False)

    start = 0
    data_sets = []
    for n in lengths:
        end = start + n
        ds = Subset(dataset=full_ds,
                    indices=permuted_indices[start:end].tolist())
        data_sets.append(ds)
        start = end

    return data_sets
