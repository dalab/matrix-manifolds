import os
import sys

import networkx as nx
import numpy as np
from scipy.spatial.distance import squareform
import torch

from graphembed.manifolds import Euclidean, Sphere

from experiments.utils import (build_manifold, make_exp_dir,
                               manifold_label_for_paths)


def main():
    torch.set_default_dtype(torch.float64)

    man_name = 'euc_{}'.format(args.dim)
    man = build_manifold(man_name)[0]

    # Sample points uniformly within a ball.
    xs = Sphere(args.dim + 1).rand_ball(args.num_nodes).mul_(args.radius)
    root_dir = make_exp_dir(args.save_dir, str(man.dim))
    torch.save(xs, os.path.join(root_dir, 'xs.pt'))

    # Get pairwise distances.
    pdists = squareform(man.pdist(xs).numpy())
    max_dist = np.max(pdists)

    # The thresholds.
    thresholds = np.linspace(args.radius / 10, args.radius, args.num_thresholds)

    for threshold in thresholds:
        # Create the graph.
        g = nx.Graph()
        g.add_edges_from(np.argwhere(pdists < threshold))
        g.remove_edges_from(nx.selfloop_edges(g))

        # Save it.
        exp_dir = make_exp_dir(args.save_dir, str(man.dim),
                               manifold_label_for_paths(man_name),
                               f'{threshold:.2f}')
        nx.write_edgelist(g, os.path.join(exp_dir, 'graph.edges.gz'))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
            description='Generate random graphs on compact manifolds.')
    parser.add_argument('--dim', type=int, required=True, help='The dimension.')
    parser.add_argument(
            '--radius',
            type=float,
            required=True,
            help='The radius of the ball to sample from.')
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
    return parser.parse_args()


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())
