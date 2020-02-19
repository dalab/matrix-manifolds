r"""NOTE: This script is very data specific!"""
import os
import sys

import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from graphembed.utils import nnp1d2_to_n

from .agg_grid_results import get_best_optim
from .plot_angle_ratios import loss_fn_for_display
from .utils import (build_manifold, fullpath_list,
                    manifold_factors_from_path_label)

matplotlib.rcParams.update({'font.size': 22})


def main():
    # create one plot per dataset
    height = 9 if args.dim == 3 else 8
    fig, ax = plt.subplots(figsize=(20, height))

    ds_dirs = sorted(list(fullpath_list(args.root_dir)))
    n_dirs = len(args.datasets) if args.datasets else len(ds_dirs)
    ds_id = None
    offsets = 1.5 * np.arange(n_dirs) / n_dirs
    offsets = offsets - np.median(offsets)
    bplot_width = 0.5 if n_dirs == 1 else offsets[-1] - offsets[-2]
    x_step = 2

    colors = []
    labels = []

    for ds_dir in ds_dirs:
        ds_name = os.path.basename(ds_dir)
        if args.datasets and ds_name not in args.datasets:
            continue
        ds_id = 0 if ds_id is None else ds_id + 1
        labels.append(ds_name)
        color = plt.cm.tab10.colors[len(colors)]
        colors.append(color)

        fp_dir = os.path.join(ds_dir, 'flipp_0.0000')
        loss_fns = []
        loss_fn_dirs = sorted(
                list(fullpath_list(fp_dir)), key=loss_fns_key_sorted)
        for loss_fn_id, loss_fn_dir in enumerate(loss_fn_dirs):
            loss_fn_str = os.path.basename(loss_fn_dir)
            if args.loss_fns and loss_fn_str not in args.loss_fns:
                continue
            loss_fns.append(loss_fn_str)

            for man_dir in fullpath_list(loss_fn_dir):
                man_name = os.path.basename(man_dir)
                if not man_name.startswith('spd_'):
                    continue
                factor_names = manifold_factors_from_path_label(man_name)
                man_factors = build_manifold(*factor_names)
                dim = sum([m.dim for m in man_factors])
                if dim != args.dim:
                    continue

                # load the samples
                samples = load_seccurv_samples(man_dir)

                pos = x_step * loss_fn_id + offsets[ds_id]
                bplot = ax.boxplot([samples],
                                   sym='',
                                   whis=[10, 90],
                                   positions=[pos],
                                   labels=[ds_name],
                                   widths=bplot_width,
                                   notch=True,
                                   patch_artist=True,
                                   medianprops=dict(linewidth=0, color='gray'))
                for patch in bplot['boxes']:
                    patch.set_facecolor(color)
                if args.dim == 3:
                    add_annotation_dim3(ax, samples, pos, ds_name, loss_fn_str)
                elif args.dim == 6:
                    add_annotation_dim6(ax, samples, pos, ds_name, loss_fn_str)

        ax.set_xticks(np.arange(0, x_step * len(loss_fns), x_step))
        ax.set_xticklabels([loss_fn_for_display(lfn) for lfn in loss_fns])
        ax.set_xlim(-1.0, x_step * len(loss_fns) - 0.75)
        for i in range(len(loss_fns)):
            ax.axvline(
                    x_step * i + x_step // 2, color='lightblue', ls='--', lw=2)

    # save the figure
    plot_seccurvs(ax, fig, labels, colors)


def load_seccurv_samples(man_dir):
    return load_samples_for_best_embedding(os.path.join(man_dir, 'orig'))


def load_samples_for_best_embedding(exp_dir):
    best_optim, _ = get_best_optim(exp_dir)
    path = os.path.join(exp_dir, best_optim, 'spd-seccurvs_0.npy')
    return np.load(path)


def add_annotation_dim3(ax, samples, xpos, ds_name, loss_fn_str):
    ypos = -9
    arrowprops = dict(facecolor='k', arrowstyle='-|>')
    bbox = dict(boxstyle='round', facecolor='none')
    if loss_fn_str == 'sne-incl_10' and ds_name == 'facebook':
        p10 = np.percentile(samples, 10)
        ax.annotate(
                'p10={}'.format(int(p10)),
                xy=(xpos - 0.05, ypos + 0.05),
                xytext=(xpos - 1.5, ypos + 0.5),
                fontsize=16,
                arrowprops=arrowprops,
                bbox=bbox)
    elif loss_fn_str == 'sste-incl_10' and ds_name == 'facebook':
        p10 = np.percentile(samples, 10)
        ax.annotate(
                'p10={}'.format(int(p10)),
                xy=(xpos - 0.05, ypos + 0.05),
                xytext=(xpos - 1.25, ypos + 0.5),
                fontsize=16,
                arrowprops=arrowprops,
                bbox=bbox)
    elif loss_fn_str == 'sne-incl_10' and ds_name == 'power':
        p10, p25 = np.percentile(samples, [10, 25])
        ax.annotate(
                'p25={}\np10={}'.format(int(p25), int(p10)),
                xy=(xpos + 0.1, ypos + 0.1),
                xytext=(xpos + 0.5, ypos + 0.5),
                fontsize=16,
                arrowprops=arrowprops,
                bbox=bbox)
    elif loss_fn_str == 'sste-incl_50' and ds_name == 'power':
        p10, p25 = np.percentile(samples, [10, 25])
        ax.annotate(
                'p25={}\np10={}'.format(int(p25), int(p10)),
                xy=(xpos + 0.1, ypos + 0.1),
                xytext=(xpos + 0.25, ypos + 2.0),
                fontsize=16,
                arrowprops=arrowprops,
                bbox=bbox)
    elif loss_fn_str == 'sne-excl_50' and ds_name == 'bio-diseasome':
        p10 = np.percentile(samples, 10)
        ax.annotate(
                'p10={}'.format(int(p10)),
                xy=(xpos - 0.1, ypos + 0.05),
                xytext=(xpos - 1.25, ypos + 0.5),
                fontsize=16,
                arrowprops=arrowprops,
                bbox=bbox)
    elif loss_fn_str == 'sne-excl_10' and ds_name == 'california':
        p10, p25 = np.percentile(samples, [10, 25])
        ax.annotate(
                'p25={}\np10={}'.format(int(p25), int(p10)),
                xy=(xpos - 0.1, ypos + 0.05),
                xytext=(xpos - 1.40, ypos + 1.0),
                fontsize=16,
                arrowprops=arrowprops,
                bbox=bbox)
    elif loss_fn_str == 'sne-excl_10' and ds_name == 'facebook':
        p10 = np.percentile(samples, 10)
        ax.annotate(
                'p10={}'.format(int(p10)),
                xy=(xpos + 0.05, ypos + 0.1),
                xytext=(xpos + 0.8, ypos + 1.5),
                fontsize=16,
                arrowprops=arrowprops,
                bbox=bbox)
    elif loss_fn_str == 'sne-excl_10' and ds_name == 'web-edu':
        p10 = np.percentile(samples, 10)
        ax.annotate(
                'p10={}'.format(int(p10)),
                xy=(xpos + 0.05, ypos + 0.1),
                xytext=(xpos + 0.1, ypos + 0.5),
                fontsize=16,
                arrowprops=arrowprops,
                bbox=bbox)
    elif loss_fn_str == 'stress' and ds_name == 'facebook':
        p10 = np.percentile(samples, 10)
        ax.annotate(
                'p10={}'.format(int(p10)),
                xy=(xpos + 0.05, ypos + 0.1),
                xytext=(xpos + 0.5, ypos + 0.5),
                fontsize=16,
                arrowprops=arrowprops,
                bbox=bbox)
    elif loss_fn_str == 'dist_1' and ds_name == 'web-edu':
        p10 = np.percentile(samples, 10)
        ax.annotate(
                'p10={}'.format(int(p10)),
                xy=(xpos + 0.05, ypos + 0.1),
                xytext=(xpos + 0.5, ypos + 0.5),
                fontsize=16,
                arrowprops=arrowprops,
                bbox=bbox)


def add_annotation_dim6(ax, samples, xpos, ds_name, loss_fn_str):
    ypos = -9
    arrowprops = dict(facecolor='k', arrowstyle='-|>')
    bbox = dict(boxstyle='round', facecolor='none')
    if (loss_fn_str in ('sne-incl_50', 'dist_1') and ds_name == 'web-edu') or \
            (loss_fn_str == 'sste-incl_50' and ds_name == 'power'):
        p10 = np.percentile(samples, 10)
        ax.annotate(
                'p10={}'.format(int(p10)),
                xy=(xpos - 0.05, ypos + 0.05),
                xytext=(xpos - 1.25, ypos + 0.5),
                fontsize=16,
                zorder=2,
                arrowprops=arrowprops,
                bbox=bbox)


def plot_seccurvs(ax, fig, labels, colors):
    ax.set_ylim(top=0.5, bottom=-9)
    ax.set_ylabel('SPD({})\n\nSectional Curvatures'.format(
            nnp1d2_to_n(args.dim)))
    if args.dim == 3:
        ax.xaxis.set_ticks([])
        ax.set_xticklabels([])
        ax.set_title(
                'Distributions of SPD Sectional Curvatures as Sampled '
                'Around the Learned Embeddings',
                y=1.26)
        # The legend.
        patches = [
                mpatches.Patch(color=color, label=label)
                for color, label in zip(colors, labels)
        ]
        ax.legend(
                handles=patches,
                bbox_to_anchor=(0, 1.02, 1, 0.2),
                loc='lower left',
                mode='expand',
                borderaxespad=0,
                ncol=3)
    elif args.dim == 6:
        ax.tick_params(axis='x', labelrotation=-15)

    plt.tight_layout()
    fig.savefig('spd-seccurvs{}.pdf'.format(args.dim), bbox_inches='tight')


def loss_fns_key_sorted(loss_fn_str):
    order = {
            'sne-incl_50': 0,
            'sne-incl_10': 1,
            'sste-incl_50': 2,
            'sste-incl_10': 3,
            'sne-excl_50': 4,
            'sne-excl_10': 5,
            'stress': 6,
            'dist_1': 7,
            'dist_2': 8
    }
    return order[os.path.basename(loss_fn_str)]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
            description='SPD sectional curvatures plot across datasets.')
    parser.add_argument('--root_dir', type=str, help='The root directory.')
    parser.add_argument(
            '--dim', type=int, required=True, help='The manifold dimension.')
    parser.add_argument('--datasets', nargs='*', help='The datasets.')
    parser.add_argument(
            '--loss_fns', nargs='+', type=str, help='The loss functions.')
    return parser.parse_args()


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())
