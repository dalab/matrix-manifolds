import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from .utils import ds_short_name

matplotlib.rcParams.update({'font.size': 18})

parser = argparse.ArgumentParser(description='Display graph property .')
parser.add_argument('--datasets', nargs='+', type=str, help='The input graphs.')
parser.add_argument(
        '--big_datasets',
        nargs='+',
        type=str,
        help='Input graphs with larger degrees to plot separately.')
parser.add_argument(
        '--properties', nargs='+', type=str, help='The properties to plot.')
parser.add_argument(
        '--only_print', action='store_true', help='Do not plot, just print.')
args = parser.parse_args()

PROPERTIES = {
        'degrees': 'Node Degrees',
        'seccurvs': 'Sectional Curvatures',
        'hyperbolicities': 'Hyperbolicities'
}


def basename_no_ext(ds):
    basename = os.path.basename(ds)
    basename = basename[:basename.index('.')]
    return basename


def load_data(ds, which):
    basename = basename_no_ext(ds)
    dirname = os.path.dirname(ds)
    analysis_dir = os.path.join(dirname, 'analysis')

    if which in ('degrees', 'seccurvs'):
        return np.load(os.path.join(analysis_dir, f'{basename}+{which}.npy'))

    assert which == 'hyperbolicities'
    counts = np.load(os.path.join(analysis_dir, f'{basename}+hyp-counts.npy'))
    values = np.load(os.path.join(analysis_dir, f'{basename}+hyp-values.npy'))
    return np.repeat(values, counts)


fig, ax = plt.subplots(figsize=(9, 6))
for prop_name in PROPERTIES.keys():
    if args.properties and prop_name not in args.properties:
        continue

    datasets = sorted(args.datasets)
    data = []
    for ds in datasets:
        ds_data = load_data(ds, prop_name)
        if args.only_print:
            mean_val = np.mean(ds_data)
            max_val = np.max(ds_data)
            print('ds: {}, hyp-mean: {}, hyp-max: {}'.format(
                    ds, mean_val, max_val))
        else:
            data.append(ds_data)
    if args.only_print:
        continue

    bplot = ax.boxplot(
            data,
            sym='',
            whis=[10, 90],
            showmeans=True,
            notch=True,
            patch_artist=True,
            medianprops=dict(linewidth=2),
            meanprops=dict(
                    marker='D',
                    markeredgecolor='black',
                    markerfacecolor='firebrick'))
    for patch in bplot['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel(PROPERTIES[prop_name])
    ax.set_xticklabels([ds_short_name(basename_no_ext(ds)) for ds in datasets])
    ax.tick_params(axis='x', labelrotation=30)
    ax.grid(axis='y', color='lightgray', lw=1, alpha=0.5)
    if prop_name == 'seccurvs':
        ax.set_ylim([-1.05, 1.05])
    elif prop_name == 'degrees':
        ax.set_ylim(bottom=0)

    if args.big_datasets:
        datasets = sorted(args.big_datasets)
        data = []
        for ds in sorted(datasets):
            data.append(load_data(ds, prop_name))

        divider = make_axes_locatable(ax)
        ax2 = divider.append_axes('right', size='30%', pad=0.7)

        bplot = ax2.boxplot(
                data,
                sym='',
                whis=[10, 90],
                widths=0.5,
                showmeans=True,
                notch=True,
                patch_artist=True,
                medianprops=dict(linewidth=2),
                meanprops=dict(
                        marker='D',
                        markeredgecolor='black',
                        markerfacecolor='firebrick'))
        for patch in bplot['boxes']:
            patch.set_facecolor('lightblue')
        ax2.set_xticklabels(
                [ds_short_name(basename_no_ext(ds)) for ds in datasets])
        ax2.tick_params(axis='x', labelrotation=30)
        ax2.grid(axis='y', color='lightgray', lw=1, alpha=0.5)

    fig.suptitle('{} of Input Graphs'.format(PROPERTIES[prop_name]), y=1.02)
    plt.tight_layout()
    fig.savefig(prop_name + '.pdf', bbox_inches='tight')
