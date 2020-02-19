import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from graphembed.utils import Timer
from ..utils import fullpath_list, man_dim_label

# FIXME(ccruceru): This needs fixes after the latest refactoring. Follow
# `experiments/plot_run_grid.py`.


def main():
    with ProcessPoolExecutor(max_workers=args.num_cpus) as pool:
        futures = []
        for dim_dir in fullpath_list(args.root_dir, only_dirs=True):
            dim = os.path.basename(dim_dir)
            if args.dims and dim not in args.dims:
                continue
            for radius_dir in fullpath_list(dim_dir, only_dirs=True):
                radius = os.path.basename(radius_dir)
                if args.radii and radius not in args.radii:
                    continue
                f = pool.submit(grid_fn, radius_dir, dim, radius)
                futures.append(f)
        for f in futures:
            f.result(None)


def grid_fn(input_dir, dim, radius):
    results = {}
    for man_dir in fullpath_list(input_dir, only_dirs=True):
        man_name = os.path.basename(man_dir)
        if args.manifolds and man_name not in args.manifolds:
            continue

        results[man_name] = {}
        for thresh_dir in fullpath_list(man_dir, only_dirs=True):
            thresh = os.path.basename(thresh_dir)

            results[man_name][thresh] = {}
            for quantity in args.plots:
                ret = read_quantity_results(thresh_dir, quantity)
                if ret is not None:
                    results[man_name][thresh][quantity] = ret

    quantity_labels = dict(
            degrees='Node Degree',
            hyperbolicities='Hyperbolicity',
            seccurvs='Sectional Curvature')
    for quantity in args.plots:
        fig, ax = plt.subplots()

        for man_name, values in results.items():
            xs = sorted([float(k) for k in values.keys()])
            p25, p50, p75 = zip(*[values[f'{k:.2f}'][quantity] for k in xs])
            label = man_dim_label(man_name, dim)
            line, = plt.plot(xs, p50, label=label, lw=2, marker='o', ms=5)
            ax.fill_between(xs, p25, p75, facecolor=line.get_color(), alpha=0.2)

        ax.set_ylim(top=args.ymax)
        ax.set_xlabel('Distance Threshold')
        ax.set_ylabel(quantity_labels[quantity])
        ax.set_title(f'dim={dim}, radius={radius}')
        ax.grid(color='lightgray', lw=2, alpha=0.5)
        ax.set_axisbelow(True)

        hs, labels = ax.get_legend_handles_labels()
        hs, labels = zip(*sorted(zip(hs, labels), key=lambda t: t[1]))
        ax.legend(hs, labels, loc='best', ncol=len(labels) // 5 + 1)

        plt.tight_layout()
        fig_name = os.path.join(args.output_dir,
                                f'{quantity}-{dim}-{radius}.pdf')
        fig.savefig(fig_name, bbox_inches='tight')
        plt.close()


def read_quantity_results(path, quantity):
    if quantity in ('degrees', 'seccurvs'):
        filename = os.path.join(path, f'{quantity}.npy')
        if not os.path.isfile:
            return None
        values = np.load(os.path.join(path, f'{quantity}.npy'))
        if len(values) == 0:  # empty graph, probably completely disconnected
            return 0, 0, 0
    else:
        values_file = os.path.join(path, 'hyp-values.npy')
        counts_file = os.path.join(path, 'hyp-counts.npy')
        if not os.path.isfile(values_file) or not os.path.isfile(counts_file):
            return None
        values = np.repeat(np.load(values_file), np.load(counts_file))
    return np.percentile(values, [25, 50, 75])


def parse_args():
    parser = argparse.ArgumentParser(
            description='Plots for generated manifold graphs.')
    parser.add_argument(
            '--root_dir',
            type=str,
            required=True,
            help='The root dir of the generated graphs hierarchy.')
    parser.add_argument(
            '--dims',
            nargs='*',
            type=str,
            help='The manifold dimensions to restrict to.')
    parser.add_argument(
            '--manifolds',
            nargs='*',
            type=str,
            help='The manifolds to restrict to.')
    parser.add_argument(
            '--radii', nargs='*', type=str, help='The radii to restrict to.')
    parser.add_argument(
            '--plots',
            nargs='+',
            choices=['degrees', 'hyperbolicities', 'seccurvs'],
            required=True,
            help='What quantities to compute and for each generated graph.')
    parser.add_argument(
            '--output_dir',
            type=str,
            help='The directory where to save the plots.')
    parser.add_argument(
            '--num_cpus',
            type=int,
            default=multiprocessing.cpu_count(),
            help='The number of CPUs to use.')
    parser.add_argument('--ymax', type=float, help='ymax to use in plots.')
    args = parser.parse_args()
    if not args.output_dir:
        args.output_dir = args.root_dir
    return args


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())
