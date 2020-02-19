import argparse
from concurrent.futures import ProcessPoolExecutor
import logging
import multiprocessing
import os
import sys

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from graphembed.utils import check_mkdir, squareform1

from .gen_man_graph import gen_samples, gen_samples_rs
from .gen_man_graph_mc import random_walk_graph
from ..utils import build_manifold

# FIXME(ccruceru): This needs fixes after the latest refactoring. Follow
# `experiments/run_grid.py`.


def main():
    torch.set_default_dtype(torch.float64)

    if args.gen_fn == 'rw':
        fn = lambda *arg: random_walk_graph(*arg, args.burnin, args.take_every)
    elif args.gen_fn == 'rs':
        fn = gen_samples_rs
    elif args.gen_fn == 'exp':
        fn = gen_samples

    with ProcessPoolExecutor(max_workers=args.num_cpus) as pool:
        futures = []
        for dim in args.dims:
            for manifold in args.manifolds:
                man, = build_manifold(manifold, dim)
                for radius in args.radii:
                    save_dir = os.path.join(args.output_dir, str(dim),
                                            f'{radius:.2f}', manifold)
                    check_mkdir(save_dir, increment=False)
                    f = pool.submit(grid_fn, fn, man, radius, save_dir)
                    futures.append(f)
        for f in futures:
            f.result(None)


def grid_fn(fn, man, radius, output_dir):
    # sample
    xs = fn(man, args.num_nodes, radius)
    pdists = man.pdist(xs)

    # save and plot the pdists
    path = os.path.join(output_dir, 'pdists.npy')
    np.save(path, pdists.numpy())
    path = os.path.join(output_dir, 'pdists.pdf')
    plot_distances(pdists, path, axvline=radius, xmax=2 * radius)

    # save and plot the distances from zero too
    zeros = man.zero(len(xs))
    zero_dists = man.dist(zeros, xs)
    path = os.path.join(output_dir, 'zero_dists.npy')
    np.save(path, zero_dists)
    path = os.path.join(output_dir, 'zero_dists.pdf')
    plot_distances(zero_dists, path, xmax=radius)

    pdists = squareform1(pdists).numpy()
    thresholds = np.linspace(radius / 10, radius, args.num_thresholds)
    for threshold in thresholds:
        # create the graph
        g = nx.Graph()
        g.add_edges_from(np.argwhere(pdists < threshold))
        g.remove_edges_from(nx.selfloop_edges(g))

        # save it
        save_dir = os.path.join(output_dir, f'{threshold:.2f}')
        check_mkdir(save_dir, increment=False)
        torch.save(xs, os.path.join(save_dir, 'xs.pt'))
        nx.write_edgelist(g, os.path.join(save_dir, 'graph.edges.gz'))


def plot_distances(dists, filepath, axvline=None, xmax=None):
    plt.hist(dists, bins=20)
    if axvline:
        plt.axvline(axvline, color='red')
    plt.xlabel('Distance')
    plt.ylabel('Number of pairs')
    plt.xlim(xmin=0, xmax=xmax)
    plt.gca().get_yaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
            description='Generate random graphs on manifolds.')
    parser.add_argument(
            '--gen_fn',
            choices=['exp', 'rw', 'rs'],
            required=True,
            help='The generation function to use.')
    parser.add_argument(
            '--burnin',
            type=int,
            default=10,
            help='The number of steps to ignore when using the random walk fn.')
    parser.add_argument(
            '--take_every',
            type=int,
            default=2,
            help='The step size used in skipping samples when using the '
            'random walk fn.')
    parser.add_argument(
            '--dims',
            nargs='+',
            type=int,
            help='The manifold dimensions to use.')
    parser.add_argument(
            '--manifolds',
            nargs='+',
            type=str,
            help='The manifolds to generate the graphs on.')
    parser.add_argument(
            '--num_nodes',
            type=int,
            default=1000,
            help='The number of nodes in the generated graphs.')
    parser.add_argument(
            '--radii',
            nargs='+',
            type=float,
            help='The radii of the ball to limit the sampling to.')
    parser.add_argument(
            '--num_thresholds',
            type=int,
            required=True,
            help='The number of thresholds to use.')
    parser.add_argument(
            '--output_dir',
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
