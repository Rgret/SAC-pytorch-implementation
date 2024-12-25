import torch.nn.functional as F
import numpy as np
import torch

def get_stats(array, min_max = False):
    x = np.array(array, dtype=np.float32)
    global_sum, global_n = x.sum(), len(x)
    mean = global_sum/global_n 
    global_sum_sq = np.sum((x - mean)**2)
    std = np.sqrt(global_sum_sq / global_n)
    if min_max:
        min, max = x.min(), x.max()
        return mean, std, min, max
    return mean, std

def get_alpha(current_step, initial_alpha, final_alpha, t_max):
    # Parameters
    start = initial_alpha
    end = final_alpha

    # Compute decay rate k
    k = -np.log(end / start) / t_max

    return end + (start - end) * np.exp(-k * current_step)

def pad_observation(state, n, x=0):
    flattened_tensor = state.view(-1)

    # Calculate the number of values to add
    numbers_to_pad = n - flattened_tensor.size(dim=0)

    # Pad the flattened tensor with x value using F.pad
    next_state = F.pad(flattened_tensor, (0, numbers_to_pad), "constant", x)

    return next_state
