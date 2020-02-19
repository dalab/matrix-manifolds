from itertools import product
import logging
from timeit import timeit

import numpy as np
import pytest
from scipy.spatial.distance import squareform
import torch

from graphembed.linalg import fast
from graphembed.manifolds import SymmetricPositiveDefinite as SPD


@pytest.mark.skipif(
        torch.cuda.is_available(), reason='torch.symeig is too slow on CUDA')
@pytest.mark.parametrize('n,d,eig', [
        (1000, 2, fast.symeig2x2),
        (10000, 3, fast.symeig3x3),
])
def test_fast_symeig_against_torch(seed, rand_sym, n, d, eig):
    x = rand_sym(n, d)
    symeig_time = timeit(lambda: torch.symeig(x), number=100)
    eig_time = timeit(lambda: eig(x), number=100)
    logging.info('Fast symeig speedup (n={}, d={}): {:.2f}x'.format(
            n, d, symeig_time / eig_time))
    assert eig_time < symeig_time


@pytest.mark.parametrize('n,d,eig', [
        (1000000, 2, fast.symeig2x2),
        (1000000, 3, fast.symeig3x3),
])
def test_fast_symeig(seed, rand_sym, n, d, eig):
    x = rand_sym(n, d)
    eig_time = timeit(lambda: eig(x), number=10)
    logging.info('Fast symeig time (n={}, d={}): {:.2f}s'.format(
            n, d, eig_time))
    # FIXME(ccruceru): The limits might need to be adjusted on other machines.
    limits = {2: (1, 0.1), 3: (3, 0.1)}  # d to (cpu time, gpu time)
    assert eig_time < limits[d][torch.cuda.is_available()]


@pytest.mark.parametrize('n', [10000, 100000])
def test_fast_cholesky2x2_against_torch(seed, rand_spd, n):
    x = rand_spd(n, 2)
    torch_time = timeit(lambda: torch.cholesky(x), number=10)
    fast_time = timeit(lambda: fast.cholesky2x2(x), number=10)
    logging.info('Fast cholesky2x2 speedup (n={}, d=2): {:.2f}x'.format(
            n, torch_time / fast_time))
    assert fast_time < torch_time


# NOTE: `torch.cholesky` does not work for n too large on GPU so we run a
# separate speed test for our manual implementation.
@pytest.mark.parametrize('n', [1000000])
def test_fast_cholesky2x2(seed, rand_spd, n):
    x = rand_spd(n, 2)
    time = timeit(lambda: fast.cholesky2x2(x), number=10)
    logging.info('Fast cholesky2x2 time (n={}, d=2): {:.2f}s'.format(n, time))
    assert time < (0.1 if torch.cuda.is_available() else 2)


@pytest.mark.parametrize('n', [10000, 100000])
def test_fast_invcholesky2x2_against_torch(seed, rand_spd, n):
    x = rand_spd(n, 2)
    torch_time = timeit(lambda: x.cholesky().inverse(), number=10)
    fast_time = timeit(
            lambda: fast.invcholesky2x2(x, ret_chol=False), number=10)
    logging.info('Fast cholesky2x2 speedup (n={}, d=2): {:.2f}x'.format(
            n, torch_time / fast_time))
    assert fast_time < torch_time


# NOTE: `torch.cholesky` does not work for n too large on GPU so we run a
# separate speed test for our manual implementation.
@pytest.mark.parametrize('n', [1000000])
def test_fast_invcholesky2x2(seed, rand_spd, n):
    x = rand_spd(n, 2)
    time = timeit(lambda: fast.invcholesky2x2(x, ret_chol=False), number=10)
    logging.info('Fast cholesky2x2 time (n={}, d=2): {:.2f}s'.format(n, time))
    assert time < (0.1 if torch.cuda.is_available() else 2)


# FIXME(ccruceru): Parameterizing the following two functions into one yields an
# unexpected slowdown on GPU for the second scenario in the list.
def test_spd_pdist_faster_than_dist2x2(seed, rand_spd):
    spd = SPD(2)
    n = 1000 if not torch.cuda.is_available() else 5000
    x = rand_spd(n, 2)
    m = torch.triu_indices(n, n, 1, device=x.device)
    pdists_time = timeit(lambda: spd.pdist(x), number=10)
    dists_time = timeit(lambda: spd.dist(x[m[0]], x[m[1]]), number=10)
    logging.info('SPD pdists speedup over dists (n={}, d=2): {:.2f}x'.format(
            n, dists_time / pdists_time))
    assert pdists_time < dists_time


def test_spd_pdist_faster_than_dist3x3(seed, rand_spd):
    spd = SPD(3)
    n = 1000
    x = rand_spd(n, 3)
    m = torch.triu_indices(n, n, 1, device=x.device)
    pdists_time = timeit(lambda: spd.pdist(x), number=10)
    dists_time = timeit(lambda: spd.dist(x[m[0]], x[m[1]]), number=10)
    logging.info('SPD pdists speedup over dists (n={}, d=3): {:.2f}x'.format(
            n, dists_time / pdists_time))
    assert pdists_time < dists_time


@pytest.mark.parametrize('n', [500, 2000])
def test_cython_map_faster(seed, rand_graph, n):
    import networkx as nx
    from graphembed.metrics import py_mean_average_precision
    from graphembed.pyx import FastPrecision

    g = rand_graph(nx.erdos_renyi_graph, n, 0.1)
    n = g.number_of_nodes()
    pdists = np.random.rand(n * (n - 1) // 2).astype(np.float32)
    square_pdists = squareform(pdists)

    ref_time = timeit(
            lambda: py_mean_average_precision(square_pdists, g), number=10)
    fp = FastPrecision(g)
    fp_time = timeit(lambda: fp.mean_average_precision(pdists), number=10)
    logging.info('Cython speedup over Python version (n={}): {:.2f}x'.format(
            n, ref_time / fp_time))
    assert fp_time < ref_time
