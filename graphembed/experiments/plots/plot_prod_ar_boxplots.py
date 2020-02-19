r"""Numbers are based on the aggregated results that can be found at:

    [1]: https://docs.google.com/spreadsheets/d/14SzV8r05FDcWoEzirgDKlVAKayeksTG7Udm8fBa6ELY/edit?usp=sharing
"""
import os
import sys

import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from .utils import ds_short_name
from ..plot_angle_ratios import load_samples_for_best_embedding
from ..utils import (fullpath_list, manifold_label_for_display,
                     manifold_factors_from_path_label)

matplotlib.rcParams.update({'font.size': 22})

ROOT_DIR = 'runs/products-all'
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
MANIFOLDS = [
        'hyp_4+hyp_4+hyp_4+hyp_4',
        'hyp_7+hyp_7',
        'hyp_7+sph_7',
        'spdstein_2+spdstein_2+spdstein_2+spdstein_2',
        'spdstein_3+spdstein_3',
        'spdstein_3+grass_5_3',
]
DS_TO_BEST_F1_LOSS_FN = {
        'california': (4, 4, 4, 4, 4, 4),
        'grqc': (4, 2, 4, 4, 4, 2),
        'web-edu': (4, 6, 4, 4, 4, 4),
}
DS_TO_BEST_DIST_LOSS_FN = {
        'california': (9, 9, 9, 9, 8, 9),
        'grqc': (8, 9, 9, 9, 9, 9),
        'web-edu': (9, 9, 9, 1, 1, 9),
}

COLORS = ['#d95f0e', '#fec44f', '#fff7bc', '#31a354', '#a1d99b', '#e5f5e0']


def main():
    fig, ax = plt.subplots(figsize=(20, 7))

    n_manifolds = len(MANIFOLDS)
    offsets = 1.5 * np.arange(n_manifolds) / n_manifolds
    offsets = offsets - np.median(offsets)
    bplot_width = 0.5 if n_manifolds == 1 else offsets[-1] - offsets[-2]
    x_step = 2

    dss_f1 = DS_TO_BEST_F1_LOSS_FN.keys()
    for ds_id, ds_name in enumerate(dss_f1):
        ds_dir = os.path.join(ROOT_DIR, ds_name)
        for i, man_name in enumerate(MANIFOLDS):
            loss_fn_id = DS_TO_BEST_F1_LOSS_FN[ds_name][i]
            loss_fn = LOSS_FNS[loss_fn_id]
            exp_dir = os.path.join(ds_dir, 'flipp_0.0000', loss_fn, man_name,
                                   'orig')
            samples = load_samples_for_best_embedding(exp_dir)

            pos = x_step * ds_id + offsets[i]
            bplot = ax.boxplot([samples],
                               sym='',
                               whis=[10, 90],
                               positions=[pos],
                               widths=bplot_width,
                               notch=True,
                               patch_artist=True,
                               medianprops=dict(linewidth=0, color='gray'))
            bplot['boxes'][0].set_facecolor(COLORS[i])

        vline_pos = x_step * ds_id + x_step // 2
        if ds_id < len(dss_f1) - 1:
            ax.axvline(vline_pos, color='lightblue', ls='--', lw=2)
        else:
            ax.axvline(vline_pos, color='k', lw=2)

    off = x_step * len(dss_f1)
    offsets += off

    dss_dist = DS_TO_BEST_DIST_LOSS_FN.keys()
    for ds_id, ds_name in enumerate(dss_f1):
        ds_dir = os.path.join(ROOT_DIR, ds_name)
        for i, man_name in enumerate(MANIFOLDS):
            loss_fn_id = DS_TO_BEST_DIST_LOSS_FN[ds_name][i]
            loss_fn = LOSS_FNS[loss_fn_id]
            exp_dir = os.path.join(ds_dir, 'flipp_0.0000', loss_fn, man_name,
                                   'orig')
            samples = load_samples_for_best_embedding(exp_dir)

            pos = x_step * ds_id + offsets[i]
            bplot = ax.boxplot([samples],
                               sym='',
                               whis=[10, 90],
                               positions=[pos],
                               widths=bplot_width,
                               notch=True,
                               patch_artist=True,
                               medianprops=dict(linewidth=0, color='gray'))
            bplot['boxes'][0].set_facecolor(COLORS[i])
        ax.axvline(
                off + x_step * ds_id + x_step // 2,
                color='lightblue',
                ls='--',
                lw=2)

    n_dss = len(dss_f1) + len(dss_dist)
    ax.set_xlim(-1, x_step * n_dss - 1)
    ax.set_ylim(-0.52, 0.05)
    ax.set_xticks(np.arange(0, x_step * n_dss, x_step))
    ax.set_xticklabels(
            ['{} (F1@1)'.format(ds_short_name(ds)) for ds in dss_f1] +
            ['{} (dist)'.format(ds_short_name(ds)) for ds in dss_dist])
    ax.yaxis.grid(color='lightgray', lw=1, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel('Normalized Sum of Angles')
    plt.suptitle(
            'Normalized Sum-of-Angles Distributions for Cartesian Products '
            '(Local & Global Settings)',
            y=1.02)
    patches = []
    for man_name, color in zip(MANIFOLDS, COLORS):
        man_name = man_name.replace('spdstein', 'spd')
        man_factors = manifold_factors_from_path_label(man_name)
        label = manifold_label_for_display(*man_factors)
        patch = mpatches.Patch(color=color, label=label)
        patches.append(patch)
    ax.legend(
            handles=patches,
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc='lower left',
            mode='expand',
            borderaxespad=0,
            ncol=6)
    plt.tight_layout()
    fig.savefig('best-angle-ratios-prod.pdf', bbox_inches='tight')


if __name__ == '__main__':
    sys.exit(main())
