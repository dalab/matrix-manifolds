from concurrent.futures import ProcessPoolExecutor
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from experiments.utils import (fullpath_list, manifold_label_for_display)

matplotlib.rcParams.update({'font.size': 18})


def main():
    with ProcessPoolExecutor(max_workers=args.num_cpus) as pool:
        futures = []
        for dim_dir in fullpath_list(args.root_dir, only_dirs=True):
            dim = os.path.basename(dim_dir)
            if args.dims and dim not in args.dims:
                continue
            f = pool.submit(grid_fn, dim_dir, int(dim))
            futures.append(f)
        for f in futures:
            f.result(None)


def grid_fn(input_dir, dim):
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
            degrees='Node Degree', seccurvs='Graph Sectional Curvature')
    quantity_titles = dict(
            degrees='Degree Distributions', seccurvs='Curvature Estimates')

    for quantity in args.plots:
        ls = ['-', '--', '-.', ':']
        ms = ['o', 'v', '*', 'd', 'x', '1']
        width = 7 if dim == 3 else 6
        fig, ax = plt.subplots(figsize=(width, 5))

        for i, man_name in enumerate(sorted(results.keys())):
            values = results[man_name]
            xs = []
            ys = []
            for k in values.keys():
                thresh_values = values[k]
                if quantity in thresh_values:
                    xs.append(float(k))
                    ys.append(thresh_values[quantity])

            xs, ys = zip(*sorted(zip(xs, ys), key=lambda e: e[0]))
            p25, p50, p75 = zip(*ys)
            label = manifold_label_for_display(man_name)
            line, = plt.plot(
                    xs,
                    p50,
                    label=label,
                    lw=6 - i,
                    ls=ls[i % 4],
                    marker=ms[i % 6],
                    ms=10)
            ax.fill_between(xs, p25, p75, facecolor=line.get_color(), alpha=0.3)

        ax.set_ylim(bottom=args.ymin, top=args.ymax)
        ax.set_xlim(left=0.8, right=args.xmax)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(color='lightgray', lw=2, alpha=0.5)
        ax.set_xlabel('Distance Threshold')
        if dim == 3:
            ax.set_ylabel(quantity_labels[quantity])
        else:
            ax.set_yticklabels([])
        ax.set_title('{} (n={})'.format(quantity_titles[quantity], dim), y=1.18)
        ax.set_axisbelow(True)

        hs, labels = ax.get_legend_handles_labels()
        hs, labels = zip(*sorted(zip(hs, labels), key=lambda t: t[1]))
        ax.legend(
                hs,
                labels,
                bbox_to_anchor=(0, 1.02, 1, 0.2),
                loc='lower left',
                mode='expand',
                borderaxespad=0,
                ncol=4)

        plt.tight_layout()
        fig_name = os.path.join(args.save_dir, f'{quantity}-{dim}.pdf')
        fig.savefig(fig_name, bbox_inches='tight')
        plt.close()


def read_quantity_results(path, quantity):
    if quantity in ('degrees', 'seccurvs'):
        filename = os.path.join(path, f'{quantity}.npy')
        if not os.path.isfile(filename):
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
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser(
            description='Plots for generated compact manifold graphs.')
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
            '--plots',
            nargs='+',
            choices=['degrees', 'hyperbolicities', 'seccurvs'],
            required=True,
            help='What quantities to compute and for each generated graph.')
    parser.add_argument(
            '--save_dir',
            type=str,
            help='The directory where to save the plots.')
    parser.add_argument(
            '--num_cpus',
            type=int,
            default=multiprocessing.cpu_count(),
            help='The number of CPUs to use.')
    parser.add_argument('--ymin', type=float, help='ymin to use in plots.')
    parser.add_argument('--ymax', type=float, help='ymax to use in plots.')
    parser.add_argument('--xmax', type=float, help='xmax to use in plots.')
    args = parser.parse_args()
    if not args.save_dir:
        args.save_dir = args.root_dir
    return args


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())
