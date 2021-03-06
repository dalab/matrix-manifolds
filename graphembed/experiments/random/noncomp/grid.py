import argparse
import logging
import logging.config
logging.config.fileConfig('logging.conf')

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import sys

import networkx as nx
import numpy as np
from scipy.spatial.distance import squareform
import torch

from graphembed.manifolds import (Euclidean, Lorentz, Sphere,
                                  SymmetricPositiveDefinite)

from experiments.utils import (build_manifold, make_exp_dir,
                               manifold_label_for_paths)


def main():
    torch.set_default_dtype(torch.float64)

    with ProcessPoolExecutor(max_workers=args.num_cpus) as pool:
        futures = []
        for man_name in args.manifold:
            man = build_manifold(man_name)[0]

            # Submit it for processing.
            f = pool.submit(grid_fn, man_name, man)
            futures.append(f)

        # Wait for results.
        for f in futures:
            f.result(None)


def grid_fn(man_name, man):
    path_components = [
            args.save_dir,
            str(man.dim),
            manifold_label_for_paths(man_name)
    ]
    root_dir = make_exp_dir(*path_components)

    # Sample points uniformly.
    xs = sample_from(man)
    torch.save(xs, os.path.join(root_dir, 'xs.pt'))

    # Get pairwise distances.
    pdists = squareform(man.pdist(xs).numpy())
    max_dist = np.max(pdists)

    # The thresholds.
    thresholds = np.linspace(args.radius / 5, 1.5 * args.radius,
                             args.num_thresholds)

    for threshold in thresholds:
        # Create the graph.
        g = nx.Graph()
        g.add_edges_from(np.argwhere(pdists < threshold))
        g.remove_edges_from(nx.selfloop_edges(g))

        # Save it.
        exp_dir = make_exp_dir(*path_components, f'{threshold:.2f}')
        nx.write_edgelist(g, os.path.join(exp_dir, 'graph.edges.gz'))

    logging.warning('Finished processing manifold %s', man_name)


def sample_from(man):
    vs = Sphere(man.dim).rand_ball(args.num_nodes).mul_(args.radius)
    us = eucl_to_tangent_space(man, vs)
    zeros = man.zero(args.num_nodes)
    assert torch.allclose(us, man.proju(zeros, us), atol=1e-4)
    return man.exp(zeros, us)


def eucl_to_tangent_space(man, vs):
    if isinstance(man, Euclidean):
        return vs
    elif isinstance(man, (Lorentz, Sphere)):
        return torch.cat([torch.zeros(len(vs), 1), vs], dim=1)
    elif isinstance(man, SymmetricPositiveDefinite):
        return SymmetricPositiveDefinite.from_vec(vs)

    raise ValueError(f'Manifold {man} is not supported.')


def parse_args():
    parser = argparse.ArgumentParser(
            description='Generate random graphs on compact manifolds.')
    parser.add_argument(
            '--radius',
            type=float,
            required=True,
            help='The radius of the ball to sample from.')
    parser.add_argument(
            '--manifold',
            action='append',
            type=str,
            help='The compact manifolds to sample from.')
    parser.add_argument(
            '--num_nodes',
            type=int,
            default=1000,
            help='The number of nodes in the generated graphs.')
    parser.add_argument(
            '--num_thresholds',
            type=int,
            default=10,
            help='The number of thresholds to use.')
    parser.add_argument(
            '--save_dir',
            type=str,
            required=True,
            help='The directory where to save the generated graphs.')
    parser.add_argument(
            '--num_cpus',
            type=int,
            default=multiprocessing.cpu_count(),
            help='The number of CPUs to use.')
    return parser.parse_args()


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())
