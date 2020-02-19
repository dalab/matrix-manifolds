import argparse
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ..utils import fullpath_list

matplotlib.rcParams.update({'font.size': 12})


def main():
    for ds_dir in fullpath_list(args.root_dir):
        ds_name = os.path.basename(ds_dir)
        for loss_fn_dir in fullpath_list(ds_dir):
            loss_fn = os.path.basename(loss_fn_dir)
            loss_fn_short = make_loss_fn_for_display(loss_fn)

            # One figure per (graph, loss_fn)
            fig, ax = plt.subplots()

            for n_factors_dir in fullpath_list(loss_fn_dir):
                n_factors = os.path.basename(n_factors_dir)
                if n_factors == 'baseline':
                    continue

                run_ratios = []
                for run_dir in fullpath_list(n_factors_dir):
                    path = os.path.join(run_dir, 'angle_ratios.npy')
                    ratios = np.load(path)
                    run_ratios.append(ratios)

                ratios = np.concatenate(run_ratios)
                xs, cdf = samples_to_cdf(ratios)
                label = '{} factor'.format(n_factors)
                if n_factors != '1':
                    label += 's'
                ax.plot(xs, cdf, lw=1.5, label=label)

            ax.set_xlabel('Sectional Curvature Estimate')
            ax.set_ylabel('Empirical CDF')
            ax.set_title(f'dataset={ds_name}, loss_fn={loss_fn_short}')
            ax.set_xlim(-args.xmax - .05, args.xmax + .05)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc='best')

            plt.tight_layout()
            fig.savefig(f'{ds_name}_{loss_fn_short}.pdf')


def samples_to_cdf(samples, n_bins=100):
    bin_counts, bin_edges = np.histogram(samples, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    y_values = np.cumsum(bin_counts) / np.sum(bin_counts)
    return bin_centers, y_values


def make_loss_fn_for_display(loss_fn_str):
    lhs, rhs = loss_fn_str.split('_')
    return '{}_{}'.format(lhs, int(float(rhs)))


def parse_args():
    parser = argparse.ArgumentParser(
            description='Angle ratio plot across factorizations.')
    parser.add_argument('--root_dir', type=str, help='The root directory.')
    parser.add_argument(
            '--xmax',
            type=float,
            default=0.5,
            help='The maximum value to restrict the x-axis to.')
    return parser.parse_args()


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())
