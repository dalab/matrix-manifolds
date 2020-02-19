r"""Numbers are based on the aggregated results that can be found at:

https://docs.google.com/spreadsheets/d/1_HnytwoYd4SaBIBpya3aCyLm-5ApejorJHGvX1Kv8I8/edit?usp=sharing
"""
import os
import sys

import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from .utils import ds_short_name
from ..plot_angle_ratios import load_samples_for_best_embedding
from ..utils import fullpath_list

matplotlib.rcParams.update({'font.size': 22})

LOSS_FNS = {
        1: 'stress',
        2: 'sste-incl_50',
        3: 'sste-incl_10',
        4: 'sne-incl_50',
        5: 'sne-incl_10',
        6: 'sne-excl_50',
        7: 'sne-excl_10',
        8: 'dist_2',
        9: 'dist_1',
}
MANIFOLDS = ['hyp_4', 'spd_2', 'spdstein_2']
DS_TO_BEST_LOSS_FN = {
        'power': (9, 9, 9),
        'csphd': (9, 9, 9),
        'facebook': (9, 9, 9),
        'bio-diseasome': (9, 9, 9),
        'bio-wormnet': (9, 9, 9),
        'road-minnesota': (9, 8, 9),
        'california': (1, 1, 1),
        'grqc': (1, 9, 8),
        'web-edu': (9, 9, 8),
}

HYP_DIST_COLOR = 'moccasin'
SPD_DIST_COLOR = 'palegreen'
SPDSTEIN_DIST_COLOR = 'lightpink'


def main():
    fig, ax = plt.subplots(figsize=(20, 6))
    x_step = 2
    bplot_width = 0.375

    dss = sorted(DS_TO_BEST_LOSS_FN.keys())
    for ds_id, ds_name in enumerate(dss):
        ds_dir = os.path.join(args.root_dir, ds_name)
        hyp_dist_best_id, spd_dist_best_id, spdstein_dist_best_id = \
                DS_TO_BEST_LOSS_FN[ds_name]

        hyp_best_dist = load_samples_for_best_embedding(
                os.path.join(ds_dir, 'flipp_0.0000', LOSS_FNS[hyp_dist_best_id],
                             'hyp_4', 'orig'))
        bplot = ax.boxplot([hyp_best_dist],
                           sym='',
                           whis=[10, 90],
                           positions=[x_step * ds_id - 0.375],
                           widths=bplot_width,
                           notch=True,
                           patch_artist=True,
                           medianprops=dict(linewidth=0, color='gray'))
        bplot['boxes'][0].set_facecolor(HYP_DIST_COLOR)

        spd_best_dist = load_samples_for_best_embedding(
                os.path.join(ds_dir, 'flipp_0.0000', LOSS_FNS[spd_dist_best_id],
                             'spd_2', 'orig'))
        bplot = ax.boxplot([spd_best_dist],
                           sym='',
                           whis=[10, 90],
                           positions=[x_step * ds_id],
                           widths=bplot_width,
                           notch=True,
                           patch_artist=True,
                           medianprops=dict(linewidth=0, color='gray'))
        bplot['boxes'][0].set_facecolor(SPD_DIST_COLOR)

        spdstein_best_dist = load_samples_for_best_embedding(
                os.path.join(ds_dir, 'flipp_0.0000',
                             LOSS_FNS[spdstein_dist_best_id], 'spdstein_2',
                             'orig'))
        bplot = ax.boxplot([spdstein_best_dist],
                           sym='',
                           whis=[10, 90],
                           positions=[x_step * ds_id + 0.375],
                           widths=bplot_width,
                           notch=True,
                           patch_artist=True,
                           medianprops=dict(linewidth=0, color='gray'))
        bplot['boxes'][0].set_facecolor(SPDSTEIN_DIST_COLOR)

        ax.axvline(x_step * ds_id - 1.0, color='lightblue', ls='--', lw=2)

    ax.set_xlim(-1, x_step * len(dss) - 1)
    ax.set_ylim(-0.52, 0.02)
    ax.set_xticks(np.arange(0, x_step * len(dss), x_step))
    ax.set_xticklabels([ds_short_name(ds) for ds in dss])
    ax.yaxis.grid(color='lightgray', lw=1, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel('(Best Distortion)\n\nNormalized Sum of Angles')
    patches = [
            mpatches.Patch(
                    color=HYP_DIST_COLOR, label='HYP(3) (best distortion)'),
            mpatches.Patch(
                    color=SPD_DIST_COLOR, label='SPD(2) (best distortion)'),
            mpatches.Patch(
                    color=SPDSTEIN_DIST_COLOR,
                    label='SPDSTEIN(2) (best distortion)'),
    ]
    ax.legend(
            handles=patches,
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc='lower left',
            mode='expand',
            borderaxespad=0,
            ncol=3)
    plt.tight_layout()
    fig.savefig('best-angle-ratios-down.pdf', bbox_inches='tight')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
            description='Plot angle ratios for best setups for all datasets.')
    parser.add_argument('--root_dir', type=str, help='The root directory.')
    return parser.parse_args()


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())
