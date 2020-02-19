import argparse
import collections
import logging
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
import numpy as np
import torch

from graphembed.data.graph import compute_graph_pdists
from graphembed.manifolds import *
from graphembed.utils import check_mkdir, squareform1
from graphembed.manifolds.lorentz import ldot

from experiments.utils import build_manifold

matplotlib.rcParams.update({'font.size': 18})


def main():
    torch.set_default_dtype(torch.float64)

    save_dir = os.path.join(args.output_dir, args.manifold)
    check_mkdir(save_dir, increment=False)
    man = build_manifold(args.manifold)[0]

    # generate the samples on the manifold, uniformly around "origin"
    if args.use_rs:
        samples = gen_samples_rs(man, args.num_nodes, args.radius,
                                 np.sqrt(args.curvature_r_squared))
    else:
        samples = gen_samples_exp(man, args.num_nodes, args.radius)
    plot_points(man, samples, args.radius, save_dir)

    # the pairwise distances
    torch.save(samples, os.path.join(save_dir, 'xs.pt'))
    dists = squareform1(pdists(man, samples))
    plot_distances(dists, save_dir)

    # create the graph
    g = nx.Graph()
    g.add_edges_from(np.argwhere(dists.numpy() < args.radius))
    g.remove_edges_from(nx.selfloop_edges(g))
    # save it
    filename = '{}_n{}_r{:.2f}'.format(args.manifold, args.num_nodes,
                                       args.radius)
    filename = filename.replace('.', 'p') + '.edges.gz'
    nx.write_edgelist(g, os.path.join(save_dir, filename))

    # plots
    plot_degree_distribution(g, save_dir)
    plot_graph_distances(g, save_dir)


def pdists(man, xs):
    if isinstance(man, (Lorentz, Sphere)):
        r = np.sqrt(args.curvature_r_squared)
        return r * man.pdist(xs)
    return man.pdist(xs)


def gen_samples_exp(man, num_nodes, radius):
    vs = gen_from_ball(num_nodes, man.dim, radius)
    us = from_vec(man, vs)
    zeros = man.zero(num_nodes)
    assert torch.allclose(us, man.proju(zeros, us), atol=1e-4)
    return man.exp(zeros, us)


def gen_samples_rs(man, num_nodes, radius, curv_r=1.0):
    xs = man.zero(num_nodes)
    max_rm = max_rm_value(man, radius, curv_r)

    n_samples = 0
    while n_samples < num_nodes:
        directions = from_vec(man, gen_from_sphere(num_nodes, man.dim, 1))
        rs = torch.rand(num_nodes).mul_(radius)
        rms = riemannian_measure(man, rs, curv_r) / max_rm
        probs = torch.rand(num_nodes)
        idx, = torch.where(probs < rms)

        n_acc = min(len(idx), num_nodes - n_samples)
        if n_acc > 0:
            zeros = man.zero(n_acc)
            us = directions * rs.reshape(-1, 1)
            xs[n_samples:(n_samples + n_acc)] = man.exp(zeros, us[idx[:n_acc]])
            n_samples += n_acc
    return xs


def max_rm_value(man, radius, curv_r):
    return riemannian_measure(man, torch.linspace(0, radius, 100), curv_r).max()


def riemannian_measure(man, rs, curv_r):
    rs = torch.as_tensor(rs)
    if isinstance(man, Lorentz):
        return torch.sinh(rs / curv_r).pow_(man.dim - 1)
    elif isinstance(man, Euclidean):
        return rs.pow(man.dim - 1)
    elif isinstance(man, Sphere):
        return torch.sin(rs / curv_r).pow_(man.dim - 1).abs()

    raise ValueError(f'Manifold {man} is not supported.')


def from_vec(man, vs):
    if isinstance(man, (Lorentz, Sphere)):
        return torch.cat([torch.zeros(len(vs), 1), vs], dim=1)
    elif isinstance(man, Euclidean):
        return vs
    elif isinstance(man, SymmetricPositiveDefinite):
        return SymmetricPositiveDefinite.from_vec(vs)

    raise ValueError(f'Manifold {man} is not supported.')


# This is the proper way of directly sampling uniformly from H(2).
def gen_samples_hyp2(num_nodes, radius):
    directions = gen_from_sphere(num_nodes, d=2, r=1)
    rs_cosh = torch.rand(num_nodes, 1).mul_(np.cosh(radius) + 1).add_(1)
    norms = torch.sqrt((rs_cosh - 1) / (rs_cosh + 1))
    xs_pb = norms * directions  # uniform on the Poincare Ball

    # map them to the hyperboloid
    rest = xs_pb * (rs_cosh + 1)
    xs = torch.cat([rs_cosh, rest], axis=1)
    assert torch.allclose(xs_pb, Lorentz.to_poincare_ball(xs), atol=1e-4)

    return xs


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


def plot_points(man, xs, radius, output_dir):
    if man.dim not in (1, 2, 3):
        return

    if isinstance(man, Lorentz):
        r = np.sqrt(args.curvature_r_squared)
        xs = man.to_poincare_ball(xs).mul_(r)
        width = 5.5 if args.use_rs else 6
        fig, ax = plt.subplots(figsize=(width, 5))
        if man.dim == 2:
            ax.scatter(xs[:, 0], xs[:, 1], s=2**4)
        else:
            p = ax.scatter(
                    xs[:, 0], xs[:, 1], c=xs[:, 2], s=2**5, cmap='gist_heat')
            if not args.use_rs:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = fig.colorbar(p, cax=cax)
                cbar.set_ticks([-0.95, 0.95])
                cbar.set_ticklabels(['-1', '1'])
                cbar.ax.set_ylabel('Depth on 3rd dimension', labelpad=-15)
        circle = plt.Circle((0, 0), r, color='r', fill=False, clip_on=False)
        ax.add_artist(circle)
        ax.set_xlim(-r - .05, r + .05)
        ax.set_ylim(-r - .05, r + .05)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        if args.use_rs:
            ax.set_title('Uniform Poincaré ball samples')
        else:
            ax.set_title('Exp-map Poincaré ball samples')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'sample.pdf'))

    elif isinstance(man, Euclidean):
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
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_title('Euclidean samples (radius = {})'.format(radius))
        fig.savefig(os.path.join(output_dir, 'sample.pdf'))

    elif isinstance(man, Sphere):
        if man.dim not in (1, 2):
            return

        r = np.sqrt(args.curvature_r_squared)
        width = 5.5 if args.use_rs else 6
        fig, ax = plt.subplots(figsize=(width, 5))
        if man.dim == 1:
            ax.scatter(xs[:, 0], xs[:, 1], s=2**4)
        else:
            p = ax.scatter(xs[:, 1], xs[:, 0], c=xs[:, 2], cmap='winter')
            if not args.use_rs:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = fig.colorbar(p, cax=cax)
                cbar.set_ticks([-0.95, 0.95])
                cbar.set_ticklabels(['-1', '1'])
                cbar.ax.set_ylabel('Depth on 3rd dimension', labelpad=-15)
        circle = plt.Circle((0, 0), r, color='r', fill=False, clip_on=False)
        ax.add_artist(circle)
        ax.set_xlim(-r - .05, r + .05)
        ax.set_ylim(-r - .05, r + .05)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        if args.use_rs:
            ax.set_title('Uniform sphere samples')
        else:
            ax.set_title('Exp-map sphere samples')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'sample.pdf'))

    plt.close()


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
            '--curvature_r_squared',
            type=float,
            default=1,
            help='The R^2 constant that gives the curvature R^2 = -1 / K.')
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
