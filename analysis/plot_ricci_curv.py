import argparse
import os
import sys

import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import remove_extensions

matplotlib.rcParams.update({'font.size': 20})


def main():
    set_seeds(args.random_seed)

    # Load the graph and sub-sample the nodes.
    g = nx.convert_node_labels_to_integers(nx.nx_pydot.read_dot(args.input))
    if args.nodes_sampling_percentage < 0.99:
        g = subsample(g)

    # Properties of the drawing.
    def read_curv(attrs):
        curv = attrs[args.curv_property]
        if curv.startswith('"'):
            curv = attrs[args.curv_property][1:-1]
        return float(curv)

    nodes = [n[0] for n in g.degree]
    edges = list(g.edges())
    degrees = np.array([n[1] for n in g.degree])
    edge_curvs = np.ndarray(shape=(len(edges)))
    node_curvs = np.ndarray(shape=(len(nodes)))
    for i, (_, _, attrs) in enumerate(g.edges(data=True)):
        edge_curvs[i] = read_curv(attrs)
    for i, (_, attrs) in enumerate(g.nodes(data=True)):
        node_curvs[i] = read_curv(attrs)

    # Show negative and positive curvatures with diverging color scheme:
    # make sure we normalize them so that transparent edges correspond to 0.
    def normalize_between(curvs, indices, a, b):
        x = curvs[indices]
        if len(np.unique(x)) > 1:
            x = (b - a) * (x - x.min()) / (x.max() - x.min()) + a
            curvs[indices] = x

    orig_edge_curvs = edge_curvs.copy()
    normalize_between(edge_curvs, np.where(edge_curvs > 0), 0.5, 1.0)
    normalize_between(edge_curvs, np.where(edge_curvs <= 0), 0, 0.5)
    orig_node_curvs = node_curvs.copy()
    normalize_between(node_curvs, np.where(node_curvs > 0), 0.5, 1.0)
    normalize_between(node_curvs, np.where(node_curvs <= 0), 0, 0.5)

    fig = plt.figure(figsize=(20, 12), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])

    curv_cmap = matplotlib.cm.get_cmap('RdBu')
    layout = nx.drawing.layout.kamada_kawai_layout(g, scale=2.0)
    kwargs = {
            'pos': layout,
            'with_labels': False,
            'nodelist': nodes,
            'edgelist': edges,
            'ax': ax,

            # node settings
            # -> no scaling for facebook and wormnet
            # -> 2 * d**1.5 for web-edu, grqc
            # -> 2 * d**3 for road-minnesota
            # -> 2 * d**2 for the others
            'node_size': 2 * degrees**2,
            'cmap': curv_cmap,
            'vmin': 0.0,
            'vmax': 1.0,
            'node_color': node_curvs,
            'linewidths': 0.01,
            'edgecolors': 'k',

            # edge settings
            'edge_cmap': curv_cmap,
            'edge_vmin': 0.0,
            'edge_vmax': 1.0,
            'edge_color': edge_curvs,
            'width': 0.5
    }
    drawing = nx.draw_networkx(g, **kwargs)
    ax.axis('off')

    # Plot the curvature colorbar separately.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cb = matplotlib.colorbar.ColorbarBase(
            cax, cmap=curv_cmap, orientation='vertical')
    cb.set_ticks([0, 0.5, 1.0])
    # We have to do this because of the curvature hack from above
    curv_ticks = [
            min(0, orig_edge_curvs.min(), orig_node_curvs.min()),
            0,
            max(0, orig_edge_curvs.max(), orig_node_curvs.min())
    ] # yapf: disable
    cb.set_ticklabels(floats_to_str(curv_ticks))
    # cb.outline.set_linewidth(0)
    cb.ax.xaxis.set_tick_params(width=0)
    cb.set_label('Ollivier-Ricci Curvature')
    filename = os.path.join(
            os.path.dirname(args.input),
            os.path.basename(args.input).split('.')[0] + '+cbar.pdf')

    # Save it
    fig.savefig(args.input.replace('dot', 'pdf'), bbox_inches='tight')


def subsample(g):
    n = g.number_of_nodes()
    degrees = np.asarray(list(dict(g.degree()).values()))
    ps = degrees / degrees.sum()
    nodes_to_keep = np.random.choice(
            n, int(n * args.nodes_sampling_percentage), replace=False, p=ps)
    g = g.subgraph(nodes_to_keep).copy()
    print('#nodes: {}, #edges: {}, #conn-comps: {}'.format(
            g.number_of_nodes(), g.number_of_edges(),
            nx.number_connected_components(g)))
    if args.keep_largest_component:
        g = g.subgraph(max(nx.connected_components(g), key=len))
        print('Keeping the largest component: #nodes: {}, #edges: {}'.format(
                g.number_of_nodes(), g.number_of_edges()))
    return g


def floats_to_str(floats):
    return ['{:.2f}'.format(f) for f in floats]


def set_seeds(seed):
    import numpy
    import random
    import torch
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
            description='Plot graphs annotated with curvature information')
    parser.add_argument(
            '--input',
            type=str,
            required=True,
            help='The input .dot graph file.')
    parser.add_argument(
            '--curv_property',
            type=str,
            default='ricciCurvature',
            help='The name of the curvature property in the graph.')
    parser.add_argument(
            '--nodes_sampling_percentage',
            type=float,
            default=1.0,
            help='The percentage of nodes to keep.')
    parser.add_argument(
            '--keep_largest_component',
            action='store_true',
            help='When sub-sampling the nodes, keep the largest connected '
            'component only.')
    parser.add_argument(
            '--random_seed',
            type=int,
            default=42,
            help='The manual random seed.')
    args = parser.parse_args()
    if args.keep_largest_component and args.nodes_sampling_percentage > 0.99:
        raise ValueError('The option `keep_largest_component` is meaningless '
                         'when not doing node subsampling')
    return args


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())
