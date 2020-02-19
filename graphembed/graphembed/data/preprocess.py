import torch

# NOTE: The whole notion of 'graph pdists preprocessing' is a legacy thing from
# when I was trying to reproduce the results from Albert Gu et al. (ICLR19).


def min_max_scale(dists, min, max):
    dists_std = (dists - dists.min()) / (dists.max() - dists.min())
    return min + dists_std * (max - min)


def exp(dists, temp=None):
    if temp is None:
        temp = dists.max()
    return torch.exp(dists / temp)
