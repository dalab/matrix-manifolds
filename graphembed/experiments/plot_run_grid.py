import logging
import logging.config
logging.config.fileConfig('logging.conf')

import glob
import os
import sys

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from graphembed.data import load_graph_pdists
from graphembed.modules import ManifoldEmbedding
from graphembed.pyx import FastPrecision
from graphembed.utils import basename_numeric_order, Timer

from .agg_grid_results import load_best_embedding
from .utils import (MANIFOLD_IDENTIFIERS, build_manifold, fullpath_list,
                    get_color_for_manifold, make_run_id,
                    manifold_label_for_display,
                    manifold_factors_from_path_label)

matplotlib.rcParams.update({'font.size': 18})


def main():
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # use double precision by default
    torch.set_default_dtype(torch.float64)

    for ds_dir in fullpath_list(args.root_dir):
        ds_name = os.path.basename(ds_dir)
        if args.datasets and ds_name not in args.datasets:
            continue
        # Load the dataset graph in order to compute F1 scores.
        _, g = load_graph_pdists(
                os.path.join('../data', ds_name + '.edges.gz'),
                cache_dir='.cached_pdists')
        n_nodes = g.number_of_nodes()
        with Timer('constructing FastPrecision'):
            fp = FastPrecision(g)
            nodes_per_layer = fp.nodes_per_layer()[1:]
            nodes_per_layer = nodes_per_layer / np.sum(nodes_per_layer)

        for flipp_dir in fullpath_list(ds_dir):
            flipp = os.path.basename(flipp_dir).split('_')[1]
            if args.flip_probabilities and flipp not in args.flip_probabilities:
                continue

            for loss_fn_dir in fullpath_list(flipp_dir):
                loss_fn_str = os.path.basename(loss_fn_dir)
                if args.loss_fns and loss_fn_str not in args.loss_fns:
                    continue
                # create one plot per (dataset, flipp, loss_fn) combination
                has_ylabel = not args.no_left_ylabel or not args.no_right_ylabel
                width = 6.5 if not has_ylabel else 7
                fig, ax = plt.subplots(figsize=(width, 5))
                plot_id = 0

                for man_dir in fullpath_list(loss_fn_dir):
                    man_name = os.path.basename(man_dir)
                    if args.manifolds and man_name not in args.manifolds:
                        continue
                    factor_names = manifold_factors_from_path_label(man_name)
                    man_factors = build_manifold(*factor_names)
                    man_label = manifold_label_for_display(*factor_names)
                    dim = sum([m.dim for m in man_factors])
                    if args.dims and dim not in args.dims:
                        continue

                    # compute the metric
                    means, stds = comp_metric(ds_name, n_nodes, fp, flipp,
                                              loss_fn_str, man_dir, man_factors,
                                              man_label)
                    if means is None:
                        continue

                    # add them to the plot
                    color = None
                    if args.use_std_colors:
                        color = get_color_for_manifold(man_name)
                    plot_f1_scores(
                            ax,
                            means,
                            stds,
                            plot_id,
                            color=color,
                            label=man_label)
                    plot_id += 1

                # save the figure
                configure_and_save_plots(ax, fig, ds_name, flipp, loss_fn_str,
                                         nodes_per_layer)


def comp_metric(ds_name, n_nodes, fp, flipp, loss_fn_str, man_dir, man_factors,
                man_label):
    run_dirs = list(fullpath_list(man_dir))
    num_pdists = n_nodes * (n_nodes - 1) // 2
    n_runs = len(run_dirs)
    all_pdists = np.ndarray(shape=(num_pdists * n_runs))

    for i, run_dir in enumerate(run_dirs):
        # load the embedding
        embedding = ManifoldEmbedding(n_nodes, man_factors)
        emb_state_dict, _, _ = load_best_embedding(run_dir)
        embedding.load_state_dict(emb_state_dict)

        # compute the pairwise distances
        with Timer('computing pdists'), torch.no_grad():
            man_pdists = embedding.compute_dists(None)
        man_pdists.sqrt_()
        indices = np.arange(i * num_pdists, (i + 1) * num_pdists)
        all_pdists[indices] = man_pdists.numpy()

    # compute the f1 scores
    run_id = make_run_id(
            dataset=ds_name, fp=flipp, loss_fn=loss_fn_str, manifold=man_label)
    logging.info('Computing F1 scores for (%s)', run_id)
    with Timer('computing F1 scores'):
        means, stds = fp.layer_mean_f1_scores(all_pdists, n_runs)

    return means[:args.max_layers], stds[:args.max_layers]


def plot_f1_scores(ax, means, stds, plot_id, color, label):
    ls = ['-', '--', '-.', ':']
    ms = ['o', 'v', '*', 'd', 'x', '1']
    lw = 4 - plot_id // 2

    x = np.arange(1, len(means) + 1)
    line, = ax.plot(
            x,
            means,
            color=color,
            label=label,
            marker=ms[plot_id % 6],
            ms=10,
            lw=lw,
            ls=ls[plot_id % 4])
    ax.fill_between(
            x,
            means - stds,
            means + stds,
            facecolor=line.get_color(),
            alpha=0.3)


def configure_and_save_plots(ax, fig, ds_name, flipp, loss_fn_str,
                             nodes_per_layer):
    if float(flipp) == 0:
        title = f'Original F1@k ({ds_name})'
        fig_name = f'{ds_name}-{loss_fn_str}.pdf'
    else:
        title = 'Noisy F1@k ({}, rp={:.2f})'.format(ds_name, float(flipp))
        fig_name = f'{ds_name}-{loss_fn_str}-flipp{flipp}.pdf'

    ax2 = ax.twinx()
    color = 'dimgray'
    x = np.arange(1, len(nodes_per_layer) + 1)
    ax2.bar(x, nodes_per_layer, color=color, alpha=0.4)
    ax2.set_ylim([0, args.right_ymax])
    if not args.no_right_ylabel:
        ax2.set_ylabel('Nodes per Layer (PDF)', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        yvals = ax2.get_yticks()
        ax2.set_yticklabels(['{:,.1%}'.format(y) for y in yvals])
    else:
        ax2.yaxis.set_ticks([])
        ax2.set_yticklabels([])

    ax.set_zorder(1)  # default zorder is 0 for ax1 and ax2
    ax.patch.set_visible(False)  # prevents ax1 from hiding ax2

    ax.set_xlabel('Layer')
    if not args.no_left_ylabel:
        ax.set_ylabel('F1@k')
    else:
        ax.set_yticklabels([])
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(color='lightgray', lw=1, alpha=0.5)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))

    if args.leftmost:
        hs, labels = ax.get_legend_handles_labels()
        hs, labels = zip(*sorted(zip(hs, labels), key=lambda t: t[1]))
        ax.legend(hs, labels, loc='center', ncol=1)

    plt.tight_layout()
    fig.savefig(fig_name, bbox_inches='tight')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Graph embedding driver.')
    parser.add_argument(
            '--root_dir', type=str, required=True, help='The root directory.')
    parser.add_argument('--datasets', nargs='*', help='The datasets.')
    parser.add_argument(
            '--manifolds', nargs='*', type=str, help='The manifolds.')
    parser.add_argument(
            '--flip_probabilities', nargs='*', help='The flip probabilities.')
    parser.add_argument(
            '--loss_fns', nargs='+', type=str, help='The loss functions.')
    parser.add_argument(
            '--dims', nargs='*', type=int, help='The manifold dimensions.')

    # Plotting settings
    parser.add_argument(
            '--use_std_colors',
            action='store_true',
            help='Use the standard manifolds colors.')
    parser.add_argument('--right_ymax', type=float, help='The right ymax.')
    parser.add_argument(
            '--leftmost', action='store_true', help='It is the left-most plot.')
    parser.add_argument(
            '--max_layers',
            type=int,
            help='The maximum number of layers to look at.')
    parser.add_argument(
            '--no_left_ylabel',
            action='store_true',
            help='Do not add the left ylabel.')
    parser.add_argument(
            '--no_right_ylabel',
            action='store_true',
            help='Do not add the right ylabel.')

    parser.add_argument(
            '--verbose', action='store_true', help='Show timing information.')
    return parser.parse_args()


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())
