import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from graphembed.utils import (basename_numeric_order,
                              latest_path_by_basename_numeric_order)

matplotlib.rcParams.update({'font.size': 10})


def main():
    args = parse_args()
    ds_name = os.path.basename(args.save_dir)

    # load the metrics
    path = glob.glob(os.path.join(args.save_dir, 'hyp2*'))[0]
    hyp_epochs, hyp_f1s, hyp_stds = load_metrics(path)
    dim = 2 * len(os.path.basename(path).split('_'))
    path = glob.glob(os.path.join(args.save_dir, 'spd*'))[0]
    spd_epochs, spd_f1s, spd_stds = load_metrics(path)

    # only keep a few of the saved metrics
    n_values, n_layers = hyp_f1s.shape
    hyp_indices = np.concatenate([
            np.arange(0, 20),
            np.arange(20, n_values, max(1, (n_values - 20) // 20)),
    ])
    hyp_f1s = hyp_f1s[hyp_indices]
    hyp_frames = len(hyp_indices)

    spd_indices = hyp_indices[np.where(hyp_indices < len(spd_epochs))]
    spd_f1s = spd_f1s[spd_indices]
    spd_frames = len(spd_indices)

    # the values are the layer ids
    x = np.arange(1, n_layers + 1)

    # plot them
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    hyp_label = lambda epoch: f'hyp2^{dim//2} (epoch={epoch})'
    spd_label = lambda epoch: f'spd{dim} (epoch={epoch})'

    hyp_line, = ax.plot(x, hyp_f1s[0], label=hyp_label(0))
    spd_line, = ax.plot(x, spd_f1s[0], label=f'spd{dim} (epoch=0)')

    def update(i):
        if i < hyp_frames:
            hyp_i = i
            spd_i = 0
        else:
            hyp_i = hyp_frames - 1
            spd_i = i - hyp_frames
        ax.collections.clear()

        hyp_line.set_ydata(hyp_f1s[hyp_i])
        ax.fill_between(
                x,
                hyp_f1s[hyp_i] - hyp_stds[hyp_i],
                hyp_f1s[hyp_i] + hyp_stds[hyp_i],
                color=hyp_line.get_color(),
                alpha=0.2)
        hyp_line.set_label(hyp_label(hyp_epochs[hyp_indices[hyp_i]]))

        spd_line.set_ydata(spd_f1s[spd_i])
        ax.fill_between(
                x,
                spd_f1s[spd_i] - spd_stds[spd_i],
                spd_f1s[spd_i] + spd_stds[spd_i],
                color=spd_line.get_color(),
                alpha=0.2)
        spd_line.set_label(spd_label(spd_epochs[spd_indices[spd_i]]))

        ax.legend(loc='lower right')
        return hyp_line, spd_line, ax

    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean F1 Score')
    ax.set_title(f'H(2)^{dim//2} to SPD({dim}), dataset={ds_name}')
    anim = FuncAnimation(
            fig,
            update,
            frames=np.arange(0, hyp_frames + spd_frames),
            interval=100)
    anim.save(f'{ds_name}.gif', dpi=80, writer='imagemagick')


def load_metrics(path):
    pattern = os.path.join(path, 'm_Layer_Mean_F1_*.npy')
    epochs, sorted_mean_files = zip(*sorted([(basename_numeric_order(f), f)
                                             for f in glob.glob(pattern)]))
    means = np.array([np.load(f) for f in sorted_mean_files])
    sorted_std_files = [
            re.sub(r'(.*)m(_Layer_Mean_F1)(.*)', r'\1s\2\3', f)
            for f in sorted_mean_files
    ]
    stds = np.array([np.load(f) for f in sorted_std_files])
    return epochs, means, stds


def parse_args():
    parser = argparse.ArgumentParser(description='Animating the F1 metric.')
    parser.add_argument(
            '--save_dir', type=str, help='The path to the injection save dir.')
    return parser.parse_args()


if __name__ == '__main__':
    sys.exit(main())
