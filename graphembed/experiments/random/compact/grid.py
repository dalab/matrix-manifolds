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
    root_dir = make_exp_dir(args.save_dir, str(man.dim))

    # Sample points uniformly.
    xs = man.rand_uniform(args.num_nodes)
    torch.save(xs, os.path.join(root_dir, 'xs.pt'))

    # Get pairwise distances.
    pdists = squareform(man.pdist(xs).numpy())
    max_dist = np.max(pdists)

    # The thresholds.
    thresholds = np.linspace(max_dist / 10, max_dist, args.num_thresholds)

    for threshold in thresholds:
        # Create the graph.
        g = nx.Graph()
        g.add_edges_from(np.argwhere(pdists < threshold))
        g.remove_edges_from(nx.selfloop_edges(g))

        # Save it.
        exp_dir = make_exp_dir(*path_components, f'{threshold:.2f}')
        nx.write_edgelist(g, os.path.join(exp_dir, 'graph.edges.gz'))

    logging.warning('Finished processing manifold %s', man_name)


def parse_args():
    parser = argparse.ArgumentParser(
            description='Generate random graphs on compact manifolds.')
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
