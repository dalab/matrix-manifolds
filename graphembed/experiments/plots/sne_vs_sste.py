import argparse
import os
import sys
import tqdm

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from sklearn.utils import shuffle
from scipy.spatial.distance import pdist
import torch

from graphembed.manifolds import Euclidean
from graphembed.objectives import KLDiveregenceLoss

matplotlib.rcParams.update({'font.size': 14})


def main():
    if args.gen_sne:
        loss_fn = KLDiveregenceLoss('sne', inclusive=not args.excl)
        gen_and_save_loss_matrix(loss_fn, 'sne')
    if args.gen_sste:
        loss_fn = KLDiveregenceLoss('sste', inclusive=not args.excl)
        gen_and_save_loss_matrix(loss_fn, 'sste')
    if args.plot_sne:
        lm = np.load(os.path.join(args.save_dir, 'sne.npy'))
        plot_loss_matrix(lm, os.path.join(args.save_dir, 'sne.pdf'))
    if args.plot_sste:
        lm = np.load(os.path.join(args.save_dir, 'sste.npy'))
        plot_loss_matrix(lm, os.path.join(args.save_dir, 'sste.pdf'))
    if args.plot_data:
        xs, ys = gen_star_data(200)
        plot_data(xs, ys, os.path.join(args.save_dir, 'input.pdf'))


def gen_and_save_loss_matrix(loss_fn, filename):
    xs, labels = gen_star_data(200)
    xs, labels = shuffle(xs, labels, random_state=args.random_state)
    lm = evaluate_loss(xs, labels, loss_fn)
    np.save(os.path.join(args.save_dir, filename), lm)


def evaluate_loss(xs, ys, loss_fn):
    max_off = get_max_offset()
    n_steps = 150
    dxs = np.linspace(-max_off, +max_off, n_steps)
    dys = np.linspace(-max_off, +max_off, n_steps)

    xs = torch.as_tensor(xs)
    ys = torch.as_tensor(ys)
    if torch.cuda.is_available():
        xs = xs.to('cuda')
        ys = ys.to('cuda')

    euc = Euclidean(2)
    xpdists = euc.pdist(xs, squared=True).mul_(args.alpha)
    y_center_indices = ys == 0

    loss_matrix = np.zeros(shape=(len(dys), len(dxs)))
    for i, dx in tqdm.tqdm(enumerate(dxs), desc='Generating loss matrix'):
        for j, dy in enumerate(dys):
            dxdy = torch.tensor([dx, dy], device=xs.device)
            zs = xs.clone()
            zs[y_center_indices] += dxdy
            zpdists = euc.pdist(zs, squared=True).mul_(args.alpha)
            loss = loss_fn(xpdists, zpdists, alpha=1.0)
            loss_matrix[i, j] = loss.item()

    return loss_matrix


def get_max_offset():
    return 10 * args.dist_multiplier


def plot_loss_matrix(loss_matrix, filepath):
    fig, ax = plt.subplots(figsize=(7.468, 7))  # (7.468, 7) when plotting colorbar
    max_off = get_max_offset()
    im = ax.imshow(
            loss_matrix,
            origin='lower',
            extent=(-max_off, +max_off, -max_off, +max_off),
            interpolation='nearest',
            aspect='equal',
            cmap='inferno',
            alpha=0.4)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    # fig.colorbar(im, cax=cax)
    ax.get_yaxis().set_visible(False)
    plt.savefig(filepath)


def plot_data(xs, ys, filename):
    max_off = get_max_offset()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(xs[:, 0], xs[:, 1], c=ys, s=5, cmap='tab20')
    ax.set_xlim([-max_off, +max_off])
    ax.set_ylim([-max_off, +max_off])
    ax.grid(color='lightgray', lw=2, alpha=0.5)
    ax.set_axisbelow(True)
    plt.savefig(filename)


def gen_star_data(n):
    means = [(0, 0), (10, 10), (10, -10), (-10, -10), (-10, 10)]

    xs = np.ndarray((len(means) * n, 2))
    labels = np.ndarray(xs.shape[0], dtype=np.int8)
    for i, mean in enumerate(means):
        indices = range((i * n), ((i + 1) * n))
        xs[indices] = np.random.multivariate_normal(mean, np.eye(2), n)
        labels[indices] = i

    return xs, labels


def parse_args():
    parser = argparse.ArgumentParser(description='Plot SNE vs. SSTE')
    parser.add_argument(
            '--alpha', type=float, default=1.0, help='The scale parameter.')
    parser.add_argument(
            '--excl', action='store_true', help='Use exclusive KL instead.')
    parser.add_argument(
            '--dist_multiplier',
            type=int,
            default=3,
            help='How far away from the generated clusters to go.')
    parser.add_argument(
            '--gen_sne', action='store_true', help='Generate SNE loss matrix.')
    parser.add_argument(
            '--gen_sste',
            action='store_true',
            help='Generate SSTE loss matrix.')
    parser.add_argument(
            '--plot_sne', action='store_true', help='Plot SNE loss matrix.')
    parser.add_argument(
            '--plot_sste', action='store_true', help='Plot SSTE loss matrix.')
    parser.add_argument(
            '--plot_data', action='store_true', help='Plot input data.')
    parser.add_argument(
            '--save_dir', type=str, default='.', help='The save directory')
    parser.add_argument(
            '--random_state', type=int, default=42, help='The random state.')
    return parser.parse_args()


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())
