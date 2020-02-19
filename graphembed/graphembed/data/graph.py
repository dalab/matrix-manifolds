import logging
import os
import random

import networkx as nx
import numpy as np
from scipy.spatial.distance import squareform
import torch

from graphembed.utils import Timer

CACHED_PDISTS_FILE = 'cached_pdists.npy'


def load_graph_pdists(f, cache_dir=None, flip_probability=None):
    if cache_dir is not None and flip_probability is not None:
        logging.warning('Cannot use caching when `flip_probability` is given.')

    g = None
    f = os.path.abspath(os.path.realpath(f))
    if flip_probability is not None:
        cache_dir = None
    elif cache_dir is not None:
        cache_dir = os.path.join(cache_dir, os.path.basename(f))

    extensions = ['.dot', '.edges', '.dir-edges']
    if any([f.endswith(ext) or f.endswith(ext + '.gz') for ext in extensions]):
        # load the graph
        with Timer('graph loading', loglevel=logging.INFO):
            if f.endswith('.dot'):
                g = nx.nx_pydot.read_dot(f)
            else:
                if f.endswith('.dir-edges') or f.endswith('.dir-edges.gz'):
                    container = nx.DiGraph
                else:
                    container = nx.Graph
                g = nx.read_edgelist(f, create_using=container)

        g = nx.convert_node_labels_to_integers(g)
        assert g.is_directed() or nx.number_connected_components(g) == 1

        # check if the graph distances are cached
        if cache_dir and os.path.isdir(cache_dir):
            f = os.path.join(cache_dir, CACHED_PDISTS_FILE)
            assert os.path.isfile(f) and f.endswith('.npy')
            # we fall back to the if statement from below
        else:
            if cache_dir:
                os.makedirs(cache_dir)
            if flip_probability is not None:
                with Timer(
                        f'creating noisy graph (fp={flip_probability})',
                        loglevel=logging.INFO):
                    g = make_noisy_graph_reroute(g, flip_probability)
            with Timer('computing graph distances', loglevel=logging.INFO):
                g_pdists = compute_graph_pdists(g, cache_dir)
            return torch.Tensor(g_pdists), g

    if f.endswith('npy'):
        with Timer('loading distances', loglevel=logging.INFO):
            return torch.Tensor(np.load(f)), g

    raise ValueError('Unrecognized input graph file: {}'.format(f))


def compute_graph_pdists(g, cache_dir=None):
    import networkit as nk

    n = g.number_of_nodes()
    weight_label = None
    if 'weight' in list(g.edges(data=True))[0][2]:
        weight_label = 'weight'

    # NOTE: Use networkit for faster all-pairs shortest paths.
    gk = nk.nxadapter.nx2nk(g, weightAttr=weight_label)
    shortest_paths = nk.distance.APSP(gk).run().getDistances()

    g_pdists = np.zeros(shape=(n, n))
    for i, u in enumerate(g.nodes()):
        for j, v in enumerate(g.nodes()):
            g_pdists[u, v] = shortest_paths[i][j]
    g_pdists = squareform(g_pdists, force='tovector', checks=True)

    if cache_dir and os.path.isdir(cache_dir):
        np.save(os.path.join(cache_dir, CACHED_PDISTS_FILE), g_pdists)

    return g_pdists


def make_noisy_graph(g, flip_probability):
    assert not g.is_directed()

    n = g.number_of_nodes()
    for i in range(n):
        for j in range(0 if g.is_directed() else i + 1, n):
            if i == j:
                continue
            if np.random.rand() < flip_probability:
                if g.has_edge(i, j):
                    g.remove_edge(i, j)
                else:
                    g.add_edge(i, j)

    # reconnect the graph if needed
    return connect_randomly(g)


def make_noisy_graph_reroute(g, flip_probability):
    assert not g.is_directed()

    n = g.number_of_nodes()
    new_g = g.copy()
    edges_modified = 0

    for u, v in g.edges():
        if np.random.rand() < flip_probability:
            # 1. remove this edge
            new_g.remove_edge(u, v)
            # 2. choose another node
            while True:
                new_v = random.choice(range(n))
                if new_v != u and new_v != v:
                    break
            # 3. connect one of u or v to the new node
            new_g.add_edge(random.choice([u, v]), new_v)

            edges_modified += 1
    logging.warning('Modified %d edges in input graph.', edges_modified)

    # no nodes should have been removed or added
    assert list(new_g.nodes()) == list(g.nodes())

    # reconnect the graph if needed
    return connect_randomly(new_g)


def make_noisy_graph_replace(g, flip_probability):
    assert not g.is_directed()

    n = g.number_of_nodes()
    new_g = g.copy()
    edges_modified = 0

    for u, v in g.edges():
        if np.random.rand() < flip_probability:
            # 1. remove this edge
            new_g.remove_edge(u, v)
            # 2. choose other two nodes to connect
            while True:
                new_u, new_v = random.choice(range(n), 2)
                if new_u == new_v:
                    break
            new_g.add_edge(new_u, new_v)

            edges_modified += 1
    logging.warning('Modified %d edges in input graph.', edges_modified)

    # no nodes should have been removed or added
    assert list(new_g.nodes()) == list(g.nodes())

    # reconnect the graph if needed
    return connect_randomly(new_g)


def connect_randomly(g):
    comps = list(nx.connected_components(g))
    if len(comps) == 1:
        return g

    logging.warning('Graph has %d connected components. '
                    'Reconnecting it.', len(comps))
    nodes_to_connect = []
    # Pick a node at random from each component.
    for comp in comps:
        u = random.choice(tuple(comp))
        nodes_to_connect.append(u)

    # Connect them randomly.
    random.shuffle(nodes_to_connect)
    for i in range(len(nodes_to_connect) - 1):
        g.add_edge(nodes_to_connect[i], nodes_to_connect[i + 1])
    assert nx.number_connected_components(g) == 1

    return g


# Allow computing and caching the distances separately from an usual graph
# embedding run.
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description='Parse graph pairwise distances caching.')
    parser.add_argument(
            '--input',
            type=str,
            required=True,
            help='The path of the input graph or input distances.')
    parser.add_argument(
            '--cache_dir',
            type=str,
            required=True,
            help='The path of the caching directory.')
    args = parser.parse_args()
    load_graph_pdists(args.input, args.cache_dir)
