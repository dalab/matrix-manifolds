import argparse
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .plot_angle_ratios_facts import (make_loss_fn_for_display, parse_args,
                                      samples_to_cdf)
from ..utils import fullpath_list

matplotlib.rcParams.update({'font.size': 12})


def main():
    for ds_dir in fullpath_list(args.root_dir):
        ds_name = os.path.basename(ds_dir)
        for n_factors in [1, 2, 3, 4, 5, 6]:
            # One figure per (graph, n_factors)
            fig, ax = plt.subplots()

            found = False
            for loss_fn_dir in fullpath_list(ds_dir):
                loss_fn = os.path.basename(loss_fn_dir)
                loss_fn_short = make_loss_fn_for_display(loss_fn)

                n_factors_dir = os.path.join(loss_fn_dir, str(n_factors))
                if not os.path.isdir(n_factors_dir):
                    continue
                found = True

                run_ratios = []
                for run_dir in fullpath_list(n_factors_dir):
                    path = os.path.join(run_dir, 'angle_ratios.npy')
                    ratios = np.load(path)
                    run_ratios.append(ratios)

                ratios = np.concatenate(run_ratios)
                xs, cdf = samples_to_cdf(ratios)
                ax.plot(xs, cdf, lw=1.5, label=loss_fn_short)

            if not found:
                continue
            ax.set_xlabel('Sectional Curvature Estimate')
            ax.set_ylabel('Empirical CDF')
            ax.set_title(f'dataset={ds_name}, n_factors={n_factors}')
            ax.set_xlim(-args.xmax - .05, args.xmax + .05)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc='best')

            plt.tight_layout()
            fig.savefig(f'{ds_name}_fact{n_factors}.pdf')


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())
