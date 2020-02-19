import os
import sys

import networkx as nx
import numpy as np
import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'helpers'))


@pytest.fixture(scope='session', autouse=True)
def set_device():
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)


@pytest.fixture(params=range(5))
def seed(request):
    seed = request.param
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def rand_sym():

    def _sym(n, d):
        x = torch.rand(n, d, d)
        return 0.5 * (x + x.transpose(dim0=1, dim1=2))

    return _sym


@pytest.fixture
def rand_spd():

    def _spd(n, d):
        x = torch.rand(n, d, d)
        return (x @ x.transpose(dim0=1, dim1=2)).add_(torch.eye(d))

    return _spd


@pytest.fixture
def rand_graph():

    def _rand(f, *args, **kwargs):
        g = f(*args, **kwargs)
        g = g.subgraph(max(nx.connected_components(g), key=len))
        g = nx.convert_node_labels_to_integers(g)
        return g

    return _rand
