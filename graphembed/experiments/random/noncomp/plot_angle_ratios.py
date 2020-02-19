import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from experiments.utils import fullpath_list, manifold_label_for_display

matplotlib.rcParams.update({'font.size': 18})

xmin = -0.55
xmax = 0.05


def main():
    for dim_dir in fullpath_list(args.root_dir, only_dirs=True):
        dim = os.path.basename(dim_dir)
        if args.dims and dim not in args.dims:
            continue
        dim = int(dim)

        width = 7 if dim == 3 else 6
        fig, ax = plt.subplots(figsize=(width, 5))
        bins = np.linspace(xmin, xmax, 100)
        colors = ['tab:orange', 'tab:green']

        i = 0
        ymax = 0
        for man_dir in sorted(fullpath_list(dim_dir, only_dirs=True)):
            man_name = os.path.basename(man_dir)
            if args.manifolds and man_name not in args.manifolds:
                continue

            filename = os.path.join(man_dir, 'angle_ratios.npy')
            values = np.load(filename)
            ret = plt.hist(
                    values,
                    bins,
                    density=True,
                    label=manifold_label_for_display(man_name),
                    color=colors[i],
                    alpha=0.5)
            ymax = max(ymax, ret[0].max())
            i += 1

        ax.grid(color='lightgray', lw=2, alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel('Normalized Sum of Angles')
        if dim == 3:
            ax.set_ylabel('PDF')
        else:
            ax.annotate(
                    'cut off (max {:.2f})'.format(ymax),
                    xy=(-0.47, 19.5),
                    xytext=(-0.38, 17),
                    arrowprops=dict(facecolor='k', arrowstyle='-|>'),
                    bbox=dict(boxstyle='round', facecolor='none'))
            ax.set_yticklabels([])
        ax.set_title('Triangles Thickness (n={})'.format(dim), y=1.18)
        ax.set_xlim(xmin, xmax)
        if args.ymax:
            ax.set_ylim(top=args.ymax + 0.1)

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
        figpath = os.path.join(args.save_dir, f'angles-{dim}.pdf')
        fig.savefig(figpath, bbox_inches='tight')
        plt.close()


def parse_args():
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser(description='Plots for angle ratios.')
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
            '--save_dir',
            type=str,
            help='The directory where to save the plots.')
    parser.add_argument('--ymax', type=float, help='ymax to use in plots.')
    args = parser.parse_args()
    if not args.save_dir:
        args.save_dir = args.root_dir
    return args


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())
