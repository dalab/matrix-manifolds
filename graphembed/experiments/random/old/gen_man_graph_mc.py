import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from graphembed.data.graph import compute_graph_pdists
from graphembed.manifolds import *
from graphembed.manifolds.lorentz import ldot
from graphembed.utils import check_mkdir, squareform1

from ..utils import build_manifold


def main():
    torch.set_default_dtype(torch.float64)
    logging.getLogger().setLevel(logging.DEBUG)

    save_dir = os.path.join(args.output_dir, args.manifold)
    check_mkdir(save_dir, increment=False)
    man = build_manifold(args.manifold)[0]

    # generate the samples
    xs = random_walk_graph(man, args.num_nodes, args.radius)
    torch.save(xs, os.path.join(save_dir, 'xs.pt'))
    plot_points(man, xs, args.radius, save_dir)

    # the pairwise distances
    pdists = man.pdist(xs)
    xmax = 2 * args.radius if not args.dist_limit else args.dist_limit
    plot_distances(pdists, os.path.join(save_dir, 'pdists.pdf'), xmax)
    pdists = squareform1(pdists)

    # the distances from 0
    zeros = man.zero(len(xs))
    zero_dists = man.dist(zeros, xs)
    xmax = args.radius
    plot_distances(zero_dists, os.path.join(save_dir, 'zero_dists.pdf'), xmax)

    # create the graph
    g = nx.Graph()
    threshold = args.link_distance if args.link_distance else args.radius
    g.add_edges_from(np.argwhere(pdists.numpy() < threshold))
    g.remove_edges_from(nx.selfloop_edges(g))

    # save it
    filename = '{}_n{}_r{:.2f}_ld{:.2f}'.format(args.manifold, args.num_nodes,
                                                args.radius, threshold)
    filename = filename.replace('.', 'p') + '.edges.gz'
    nx.write_edgelist(g, os.path.join(save_dir, filename))

    # plots
    plot_degree_distribution(g, save_dir)
    plot_graph_distances(g, save_dir)


def random_walk_graph(man, num_nodes, radius, burnin=100, take_every=3):
    num_samples = burnin + num_nodes * take_every
    zero = man.zero()
    xs = torch.empty(num_samples, *zero.size())
    xs[0] = zero

    for i in range(1, num_samples):
        while True:
            r = np.random.rand() * radius
            u = man.randvec(xs[i - 1], norm=r)
            x = man.projx(man.exp(xs[i - 1], u), inplace=True)
            if man.dist(zero, x) < radius:
                xs[i] = x
                break
        if i % 100 == 0:
            logging.debug('Sampled %d', i)

    return xs[burnin::take_every]


# The multiplication by :math:`sqrt{2}` corresponds to positioning the points on
# the hyperboloid with :math:`x0^2 - x1^2 - x2^2 = 2` which has constant
# sectional curvature :math:`-1/2`.
sspd2_hyp_radius_ = np.sqrt(2)


def plot_points(man, xs, radius, output_dir):
    if isinstance(man, Euclidean):
        fig, ax = plt.subplots()
        if man.dim == 2:
            ax.scatter(xs[:, 0], xs[:, 1], s=2**4)
        else:
            p = ax.scatter(xs[:, 0], xs[:, 1], c=xs[:, 2], cmap='winter')
            cbar = fig.colorbar(p)
            cbar.ax.set_ylabel('Depth on 3rd dimension')
        r = radius if not args.dist_limit else args.dist_limit / 2
        circle = plt.Circle((0, 0), r, color='r', fill=False, clip_on=False)
        ax.add_artist(circle)
        ax.set_xlim(-r - .1, r + .1)
        ax.set_ylim(-r - .1, r + .1)
        ax.set_title('Euclidean samples (radius = {})'.format(radius))
        fig.savefig(os.path.join(output_dir, 'sample.pdf'))

    elif isinstance(man, Sphere):
        if man.dim not in (1, 2):
            return

        r = 1
        fig, ax = plt.subplots()
        if man.dim == 1:
            ax.scatter(xs[:, 0], xs[:, 1], s=2**4)
        else:
            p = ax.scatter(xs[:, 1], xs[:, 0], c=xs[:, 2], cmap='winter')
            cbar = fig.colorbar(p)
            cbar.ax.set_ylabel('Depth on 3rd dimension')
        circle = plt.Circle((0, 0), r, color='r', fill=False, clip_on=False)
        ax.add_artist(circle)
        ax.set_xlim(-r - .1, r + .1)
        ax.set_ylim(-r - .1, r + .1)
        ax.set_title('Sphere samples (radius = {})'.format(radius))
        fig.savefig(os.path.join(output_dir, 'sample.pdf'))

    elif isinstance(man, SymmetricPositiveDefinite):
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

    elif isinstance(man, Lorentz):
        r = 1
        xs = man.to_poincare_ball(xs).mul_(r)
        fig, ax = plt.subplots()
        if man.dim == 2:
            ax.scatter(xs[:, 0], xs[:, 1], s=2**4)
        else:
            p = ax.scatter(xs[:, 0], xs[:, 1], c=xs[:, 2], cmap='winter')
            cbar = fig.colorbar(p)
            cbar.ax.set_ylabel('Depth on 3rd dimension')
        circle = plt.Circle((0, 0), r, color='r', fill=False, clip_on=False)
        ax.add_artist(circle)
        ax.set_xlim(-r - .05, r + .05)
        ax.set_ylim(-r - .05, r + .05)
        ax.set_title('Poincaré ball samples (radius = {})'.format(radius))
        fig.savefig(os.path.join(output_dir, 'sample.pdf'))

    plt.close()


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


def plot_graph_distances(g, output_dir):
    g = g.subgraph(max(nx.connected_components(g), key=len))
    g = nx.convert_node_labels_to_integers(g)
    logging.info('The largest connected component has %d nodes',
                 g.number_of_nodes())

    pdists = compute_graph_pdists(g)
    plt.hist(pdists, bins=20)
    plt.xlabel('Distance')
    plt.ylabel('Number of node pairs')
    plt.title('manifold = {}, radius = {}'.format(args.manifold, args.radius))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gdists.pdf'))
    plt.close()


def plot_distances(dists, filename, xmax):
    plt.hist(dists, bins=20)
    plt.axvline(args.radius, color='red')
    plt.xlabel('Distance')
    plt.ylabel('Number of pairs')
    plt.xlim(xmin=0, xmax=xmax)
    plt.gca().get_yaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.title('manifold = {}, radius = {}'.format(args.manifold, args.radius))
    plt.tight_layout()
    plt.savefig(filename)
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
            '--num_nodes',
            type=int,
            required=500,
            help='The number of nodes in the generated graph.')
    parser.add_argument(
            '--radius',
            type=float,
            required=True,
            help='The radius of the ball to uniformly sample.')
    parser.add_argument(
            '--link_distance',
            type=float,
            required=False,
            help='The distance used as a threshold to link nodes in the graph.')
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
