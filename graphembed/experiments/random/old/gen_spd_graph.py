import argparse
import collections
import logging
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from graphembed.linalg import torch_batch as tb
from graphembed.manifolds import SymmetricPositiveDefinite
from graphembed.manifolds.lorentz import ldot
from graphembed.utils import check_mkdir, squareform1

from experiments.utils import build_manifold


def main():
    torch.set_default_dtype(torch.float64)

    save_dir = os.path.join(args.output_dir, args.manifold)
    check_mkdir(save_dir, increment=False)

    # generate the samples on the manifold, uniformly around "origin"
    man = build_manifold(args.manifold)[0]
    if args.use_rs:
        samples = gen_samples_rs(man, args.num_nodes, args.radius)
    elif args.use_gen2:
        samples = gen_samples2(man, args.num_nodes, args.radius)
    else:
        samples = gen_samples(man, args.num_nodes, args.radius)
    torch.save(samples, os.path.join(save_dir, 'xs.pt'))
    dists = squareform1(man.pdist(samples))
    plot_distances(dists, save_dir)

    # create the graph
    g = nx.Graph()
    g.add_edges_from(np.argwhere(dists.numpy() < args.radius))
    g.remove_edges_from(nx.selfloop_edges(g))

    # plots
    plot_degree_distribution(g, save_dir)
    plot_points(man, samples, args.radius, save_dir)


def gen_samples(man, num_nodes, radius):
    vs = gen_from_ball(num_nodes, man.dim, radius)
    us = SymmetricPositiveDefinite.from_vec(vs)
    zeros = man.zero(num_nodes)
    assert torch.allclose(us, man.proju(zeros, us), atol=1e-4)
    return man.exp(zeros, us)


def gen_samples2(man, num_nodes, radius):
    # 1. Generate eigenvalues in the ball of radius `radius`.
    eigs = gen_from_ball(num_nodes, man.n, radius)
    # 2. Sample the orthogonal matrices.
    us = gen_ortho(num_nodes, man.n)
    # 3. Get the SPD matrices
    xs = tb.hgie(eigs.exp(), us)
    assert torch.allclose(man.projx(xs), xs, atol=1e-4)

    return xs


def gen_samples_rs(man, num_nodes, radius):
    # 1. Sample the eigenvalues
    n = man.n
    zero = man.zero()
    eigs = torch.empty(num_nodes, n)
    max_rm = riemannian_measure(man, torch.full((1, n), -radius / np.sqrt(n)))
    # max_rm = riemannian_measure(man, gen_from_sphere(1, n, radius)).squeeze()

    n_samples = 0
    while n_samples < num_nodes:
        xs = gen_from_ball(num_nodes, n, radius)
        rms = riemannian_measure(man, xs) / max_rm
        probs = torch.rand(num_nodes)
        idx, = torch.where(probs < rms)

        n_acc = min(len(idx), num_nodes - n_samples)
        if n_acc > 0:
            eigs[n_samples:(n_samples + n_acc)] = xs[idx[:n_acc]]
            n_samples += n_acc

    # 2. Sample the orthogonal matrices.
    us = gen_ortho(num_nodes, n)

    # 3. Get the SPD matrices
    xs = tb.hgie(eigs.exp(), us)
    assert torch.allclose(man.projx(xs), xs, atol=1e-4)

    return xs


def riemannian_measure(man, eigs):
    return eigs.sum(dim=1).exp_().reciprocal_()
    # return eigs.pow(2).sum(dim=1).sqrt_().exp_()

    # return eigs.sum(dim=1).exp_().pow_(0.5 * (man.n + 1))
    # return eigs.pow(2).sum(dim=1).sqrt_().exp_().pow_(0.5 * (man.n + 1))


# Reference: https://arxiv.org/pdf/math-ph/0609050.pdf
def gen_ortho(m, n):
    z = torch.randn(m, n, n)
    q, r = z.qr()
    d = r.diagonal(dim1=-2, dim2=-1).sign_()
    return q * d.reshape(m, 1, n)


def gen_from_ball(num_samples, d, r):
    u = gen_from_sphere(num_samples, d, r)
    dist = torch.rand(num_samples, 1).pow_(1 / d)
    u.mul_(dist)
    return u


def gen_from_sphere(num_samples, d, r):
    u = torch.randn(num_samples, d)
    u_norm = u.norm(dim=1, keepdim=True)
    u.div_(u_norm).mul_(r)
    return u


# The multiplication by :math:`sqrt{2}` corresponds to positioning the points on
# the hyperboloid with :math:`x0^2 - x1^2 - x2^2 = 2` which has constant
# sectional curvature :math:`-1/2`.
sspd2_hyp_radius_ = np.sqrt(2)


def plot_points(man, xs, radius, output_dir):
    if isinstance(man, SymmetricPositiveDefinite):
        if man.dim != 3:
            return
        ds = xs.det()
        xspb = sspd2_to_h2(xs)

        fig, ax = plt.subplots()
        p = ax.scatter(xspb[:, 0], xspb[:, 1], c=ds, cmap='winter', alpha=.8)
        cbar = fig.colorbar(p)
        cbar.ax.set_ylabel('Determinant')
        circle = plt.Circle((0, 0),
                            sspd2_hyp_radius_,
                            color='r',
                            fill=False,
                            clip_on=False)
        ax.add_artist(circle)
        ax.set_xlim(-sspd2_hyp_radius_ - .05, sspd2_hyp_radius_ + .05)
        ax.set_ylim(-sspd2_hyp_radius_ - .05, sspd2_hyp_radius_ + .05)
        ax.set_title('SPD(2) samples (radius = {})'.format(radius))
        fig.savefig(os.path.join(output_dir, 'sample.pdf'))


def sspd2_to_h2(xs):
    ys = sspd2_to_hyp(xs)
    y0 = ys[..., 0]
    y1 = ys[..., 1]
    y2 = ys[..., 2]
    denom = y0.reshape(-1, 1) / sspd2_hyp_radius_ + 1
    return torch.stack([y1, y2], axis=-1).div_(denom)


def sspd2_to_hyp(xs):
    a = xs[..., 0, 0]
    b = xs[..., 1, 1]
    c = xs[..., 0, 1]

    x0 = (a + b).mul_(0.5)
    x1 = (a - b).mul_(0.5)
    x2 = c

    y = torch.stack([x0, x1, x2], axis=-1)
    y.div_(torch.sqrt(-ldot(y, y, keepdim=True))).mul_(sspd2_hyp_radius_)
    return y


def plot_degree_distribution(g, output_dir):
    degrees = list(dict(g.degree()).values())
    mean = np.mean(degrees)

    plt.hist(degrees, bins=20)
    annotate_vline(mean, text=f'Mean: {mean:.2f}', color='red', lw=2)
    plt.xlabel('Node Degree')
    plt.ylabel('#nodes')
    plt.xlim(xmin=0, xmax=args.deg_limit)
    plt.gca().get_yaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.title(
            'manifold = {}, radius = {}'.format(args.manifold, args.radius),
            y=1.08)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'degrees.pdf'))
    plt.close()


def plot_distances(dists, output_dir):
    plt.hist(squareform1(dists), bins=20)
    plt.axvline(args.radius, color='red')
    plt.xlabel('Distance')
    plt.ylabel('Number of pairs')
    xmax = 2 * args.radius if not args.dist_limit else args.dist_limit
    plt.xlim(xmin=0, xmax=xmax)
    plt.gca().get_yaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.title('manifold = {}, radius = {}'.format(args.manifold, args.radius))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dists.pdf'))
    plt.close()


def annotate_vline(value, text, left=True, **kwargs):
    plt.axvline(value, **kwargs)
    xoff, align = (-15, 'left') if left else (15, 'right')
    plt.annotate(
            text,
            xy=(value, 1),
            xytext=(xoff, 15),
            xycoords=('data', 'axes fraction'),
            textcoords='offset points',
            horizontalalignment=align,
            verticalalignment='center')


def parse_args():
    parser = argparse.ArgumentParser(
            description='Generate random graphs on manifolds.')
    parser.add_argument(
            '--manifold',
            type=str,
            required=True,
            help='The manifold to generate the graph on.')
    parser.add_argument(
            '--use_rs',
            action='store_true',
            help='Use Rejection Sampling scheme from the Riemannian measure.')
    parser.add_argument(
            '--use_gen2',
            action='store_true',
            help='Use second non-uniform sampling algorithm.')
    parser.add_argument(
            '--num_nodes',
            type=int,
            default=500,
            help='The number of nodes in the generated graph.')
    parser.add_argument(
            '--radius',
            type=float,
            required=True,
            help='The radius of the ball to uniformly sample.')
    parser.add_argument(
            '--deg_limit', type=int, help='The xlim for the degree plot.')
    parser.add_argument(
            '--dist_limit', type=int, help='The xlim for the dists plot.')
    parser.add_argument(
            '--output_dir',
            type=str,
            default='output/tmp',
            help='The directory where to save plots, etc.')
    return parser.parse_args()


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())
