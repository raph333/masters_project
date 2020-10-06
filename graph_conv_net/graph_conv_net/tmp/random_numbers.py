import torch
import numpy as np

state = np.random.RandomState(42)


for _ in range(5):
    # r = torch.randperm(3, generator=g)
    nums = state.permutation(10)
    print(nums)
