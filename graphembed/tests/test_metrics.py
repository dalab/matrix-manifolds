from itertools import product

import networkx as nx
import numpy as np
import pytest
from scipy.spatial.distance import squareform

from graphembed.metrics import area_under_curve, py_mean_average_precision
from graphembed.pyx import FastPrecision
from graphembed.data.graph import compute_graph_pdists

from utils import assert_allclose


@pytest.mark.parametrize('n,p', product([50, 100, 1000], [0.01, 0.1, 0.5]))
def test_cython_map(rand_graph, n, p):
    g = rand_graph(nx.erdos_renyi_graph, n, p)
    n = g.number_of_nodes()

    pdists = np.random.rand(n * (n - 1) // 2).astype(np.float32)
    ref_map = py_mean_average_precision(squareform(pdists), g)
    fp = FastPrecision(g)
    assert_allclose(fp.mean_average_precision(pdists), ref_map, atol=1e-6)


@pytest.mark.parametrize('n,p', product([50, 500], [0.01, 0.1]))
def test_f1_trivial(rand_graph, n, p):
    g = rand_graph(nx.erdos_renyi_graph, n, p)
    n = g.number_of_nodes()

    pdists = compute_graph_pdists(g)
    fp = FastPrecision(g)
    means, _ = fp.layer_mean_f1_scores(pdists)
    assert_allclose(means, np.ones(len(means)), atol=1e-6)
    assert_allclose(area_under_curve(means), 1, atol=1e-6)


@pytest.mark.xfail
@pytest.mark.parametrize('n,p', product([50, 500], [0.01, 0.1]))
def test_f1_reversed_order(rand_graph, n, p):
    g = rand_graph(nx.erdos_renyi_graph, n, p)
    n = g.number_of_nodes()

    pdists = compute_graph_pdists(g)
    fp = FastPrecision(g)
    # TODO(ccruceru): Decide if this behaviour makes more sense.
    means, _ = fp.layer_mean_f1_scores(1 / pdists)
    assert_allclose(means, np.zeros(len(means)), atol=1e-6)
    assert_allclose(area_under_curve(means), 0, atol=1e-6)
