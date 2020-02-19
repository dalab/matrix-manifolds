import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .agg_grid_results import get_best_optim
from .utils import (build_manifold, fullpath_list, get_color_for_manifold,
                    manifold_label_for_display,
                    manifold_factors_from_path_label)

matplotlib.rcParams.update({'font.size': 18})


def main():
    for ds_dir in fullpath_list(args.root_dir):
        ds_name = os.path.basename(ds_dir)
        if args.datasets and ds_name not in args.datasets:
            continue
        for flipp_dir in fullpath_list(ds_dir):
            flipp = os.path.basename(flipp_dir).split('_')[1]
            if args.flip_probabilities and flipp not in args.flip_probabilities:
                continue
            for loss_fn_dir in fullpath_list(flipp_dir):
                loss_fn_str = os.path.basename(loss_fn_dir)
                if args.loss_fns and loss_fn_str not in args.loss_fns:
                    continue
                # create one plot per (dataset, flipp, loss_fn) combination
                width = 6.5 if args.no_ylabel else 7
                fig, ax = plt.subplots(figsize=(width, 5))

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

                    # load the angle ratio samples
                    samples = load_angle_ratio_samples(man_dir)

                    # plot them
                    plot_angle_ratios(
                            ax,
                            samples,
                            label=man_label,
                            color=get_color_for_manifold(man_name))

                # save the figure
                configure_and_save_plots(ax, fig, ds_name, flipp, loss_fn_str)


def load_angle_ratio_samples(man_dir):
    all_samples = []
    for run_dir in fullpath_list(man_dir):
        samples = load_samples_for_best_embedding(run_dir)
        all_samples.append(samples)

    return np.concatenate(all_samples)


def load_samples_for_best_embedding(exp_dir):
    best_optim, _ = get_best_optim(exp_dir)
    path = os.path.join(exp_dir, best_optim, 'angle_ratios.npy')
    return np.load(path)


def plot_angle_ratios(ax, samples, label, color):
    bins = 100
    if args.xmin and args.xmax:
        bins = np.linspace(args.xmin, args.xmax, 100)
    plt.hist(samples, bins, density=True, label=label, color=color, alpha=0.5)


def configure_and_save_plots(ax, fig, ds_name, flipp, loss_fn_str):
    # The saved file name.
    if float(flipp) == 0:
        fig_name = f'{ds_name}-{loss_fn_str}.pdf'
    else:
        fig_name = f'{ds_name}-{loss_fn_str}-flipp{flipp}.pdf'
    # The title.
    title = '{}; {}'.format(
            ds_name_for_display(ds_name), loss_fn_for_display(loss_fn_str))
    if float(flipp) != 0:
        title += '; {flipp}'

    ax.grid(color='lightgray', lw=2, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel('Normalized Sum of Angles')
    if not args.no_ylabel:
        ax.set_ylabel('PDF')
    ax.set_title(title, y=1.18)
    ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylim(top=args.ymax)
    ax.yaxis.set_major_locator(plt.MaxNLocator(6, integer=True))

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
    fig.savefig(fig_name, bbox_inches='tight')


def ds_name_for_display(ds_name):
    if ds_name == 'drill_shaft_zip':
        return 'drill-shaft'
    return ds_name


def loss_fn_for_display(loss_fn_str):
    if loss_fn_str == 'stress':
        return 'Stress'
    elif loss_fn_str == 'sste-incl_50':
        return 'SST (Low Temp, Incl)'
    elif loss_fn_str == 'sste-incl_10':
        return 'SST (High Temp, Incl)'
    elif loss_fn_str == 'sne-incl_50':
        return 'SNE (Low Temp, Incl)'
    elif loss_fn_str == 'sne-incl_10':
        return 'SNE (High Temp, Incl)'
    elif loss_fn_str == 'sne-excl_50':
        return 'SNE (Low Temp, Excl)'
    elif loss_fn_str == 'sne-excl_10':
        return 'SNE (High Temp, Excl)'
    elif loss_fn_str == 'dist_1':
        return 'Distortion 1'
    elif loss_fn_str == 'dist_2':
        return 'Distortion 2'

    assert False, f'Unknown loss function {loss_fn_str}'


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
            description='Angle ratio plot across factorizations.')
    parser.add_argument('--root_dir', type=str, help='The root directory.')
    parser.add_argument('--datasets', nargs='*', help='The datasets.')
    parser.add_argument(
            '--flip_probabilities', nargs='*', help='The flip probabilities.')
    parser.add_argument(
            '--loss_fns', nargs='+', type=str, help='The loss functions.')
    parser.add_argument(
            '--dims', nargs='*', type=int, help='The manifold dimensions.')
    parser.add_argument(
            '--manifolds', nargs='*', type=str, help='The manifolds.')
    parser.add_argument(
            '--no_ylabel', action='store_true', help='Do not add the ylabel.')
    parser.add_argument(
            '--xmin',
            type=float,
            help='The minimuum value to restrict the x-axis to.')
    parser.add_argument(
            '--xmax',
            type=float,
            help='The maximum value to restrict the x-axis to.')
    parser.add_argument('--ymax', type=float, help='ymax to use in plots.')
    return parser.parse_args()


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())
