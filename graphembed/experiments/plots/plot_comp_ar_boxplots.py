r"""Numbers are based on the aggregated results that can be found at:

    [1]: https://docs.google.com/spreadsheets/d/1APoC4r1F7LUmwZSpTki75PRkpt6abXXVej1xAglWqFY/edit?usp=sharing
    [2]: https://docs.google.com/spreadsheets/d/1zOLBjPybr6pvaf2RPcbU0irwpXFOlXM0tfydQbX7Ro4/edit?usp=sharing
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
MANIFOLDS = ['sph_5', 'grass_5_1', 'grass_4_2']
DS_TO_BEST_LOSS_FN = {
        'bun_zipper_res3': (4, 6, 8),
        'bun_zipper_res4': (6, 4, 2),
        'drill_shaft_zip': (9, 9, 9),
        'road-minnesota': (9, 9, 9),
        'regular_sphere1000': (6, 2, 2),
        'catcortex': (9, 8, 9),
}

SPH_COLOR = 'tab:orange'
GR51_COLOR = 'tab:purple'
GR42_COLOR = 'tab:green'


def main():
    fig, ax = plt.subplots(figsize=(15, 7))
    x_step = 2
    bplot_width = 0.375

    dss = DS_TO_BEST_LOSS_FN.keys()
    for ds_id, ds_name in enumerate(dss):
        if ds_name == 'catcortex':
            root_dir = 'runs/spherical-diss-oct21'
        else:
            root_dir = 'runs/spherical-sep'
        ds_dir = os.path.join(root_dir, ds_name)
        sph_best_id, gr51_best_id, gr42_best_id = DS_TO_BEST_LOSS_FN[ds_name]

        sph_best = load_samples_for_best_embedding(
                os.path.join(ds_dir, 'flipp_0.0000', LOSS_FNS[sph_best_id],
                             'sph_5', 'orig'))
        bplot = ax.boxplot([sph_best],
                           sym='',
                           whis=[10, 90],
                           positions=[x_step * ds_id - 0.375],
                           widths=bplot_width,
                           notch=True,
                           patch_artist=True,
                           medianprops=dict(linewidth=0, color='gray'))
        bplot['boxes'][0].set_facecolor(SPH_COLOR)

        gr51_best = load_samples_for_best_embedding(
                os.path.join(ds_dir, 'flipp_0.0000', LOSS_FNS[gr51_best_id],
                             'grass_5_1', 'orig'))
        bplot = ax.boxplot([gr51_best],
                           sym='',
                           whis=[10, 90],
                           positions=[x_step * ds_id],
                           widths=bplot_width,
                           notch=True,
                           patch_artist=True,
                           medianprops=dict(linewidth=0, color='gray'))
        bplot['boxes'][0].set_facecolor(GR51_COLOR)

        gr42_best = load_samples_for_best_embedding(
                os.path.join(ds_dir, 'flipp_0.0000', LOSS_FNS[gr42_best_id],
                             'grass_4_2', 'orig'))
        bplot = ax.boxplot([gr42_best],
                           sym='',
                           whis=[10, 90],
                           positions=[x_step * ds_id + 0.375],
                           widths=bplot_width,
                           notch=True,
                           patch_artist=True,
                           medianprops=dict(linewidth=0, color='gray'))
        bplot['boxes'][0].set_facecolor(GR42_COLOR)

        ax.axvline(x_step * ds_id - 1.0, color='lightblue', ls='--', lw=2)

    ax.set_xlim(-1, x_step * len(dss) - 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(np.arange(0, x_step * len(dss), x_step))
    ax.set_xticklabels([ds_short_name(ds) for ds in dss])
    ax.yaxis.grid(color='lightgray', lw=1, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel('Normalized Sum of Angles')
    ax.set_title('Normalized Sum-of-Angles Distributions', y=1.14)
    patches = [
            mpatches.Patch(color=SPH_COLOR, label='SPH(4)'),
            mpatches.Patch(color=GR51_COLOR, label='GR(1, 5)'),
            mpatches.Patch(color=GR42_COLOR, label='GR(2, 4)'),
    ]
    ax.legend(
            handles=patches,
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc='lower left',
            mode='expand',
            borderaxespad=0,
            ncol=3)
    plt.tight_layout()
    fig.savefig('best-angle-ratios-comp.pdf', bbox_inches='tight')


if __name__ == '__main__':
    sys.exit(main())
