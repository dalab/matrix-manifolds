from itertools import product
import numpy as np
import pytest
from scipy.spatial.distance import cdist, pdist
import torch

from graphembed.manifolds import Euclidean
from utils import assert_allclose


def test_dim():
    man = Euclidean(100)
    assert man.dim == 100
    man = Euclidean(10, 5, 2)
    assert man.dim == 100


def test_distances_batch(seed):
    man = Euclidean(10)
    x = torch.rand(20, 10)
    x_cpu = x.cpu()
    dists_ref = np.diag(cdist(x_cpu, x_cpu))
    dists = man.dist(x, x)
    assert_allclose(dists_ref, dists, atol=1e-4)


@pytest.mark.parametrize('n,d', product([10, 100], range(10, 20)))
def test_pdists(seed, n, d):
    man = Euclidean(d)
    x = torch.rand(n, d)
    dists_ref = pdist(x.cpu())
    dists = man.pdist(x)
    assert_allclose(dists_ref, dists, atol=1e-4)
