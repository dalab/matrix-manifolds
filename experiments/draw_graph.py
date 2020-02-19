import sys

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
plt.rcParams.update({'font.size': 22})

graph_file = 'output/graph.dot'
nodes_to_keep_percentage = 1.0


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    # Stolen from https://stackoverflow.com/a/18926541/3183129
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(
                    n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def floats_to_str(floats):
    return ['{:.2f}'.format(f) for f in floats]


def plot_graph_and_curvatures(g, curv_property, curv_name, filename):
    # properties of the drawing below
    nodes = [n[0] for n in g.degree]
    edges = list(g.edges())
    degrees = np.array([n[1] for n in g.degree])
    curvs = []
    for _, _, attrs in g.edges(data=True):
        curv = attrs[curv_property]
        if curv.startswith('"'):
            curv = attrs[curv_property][1:-1]
        curvs.append(float(curv))
    curvs = np.array(curvs)

    # show negative and positive curvatures with diverging color scheme: make
    # sure we normalize them so that transparent edges correspond to 0 curvature
    def normalize_between(indices, a, b):
        x = curvs[indices]
        if len(np.unique(x)) > 1:
            x = (b - a) * (x - x.min()) / (x.max() - x.min()) + a
            curvs[indices] = x

    orig_curvs = curvs.copy()
    normalize_between(np.where(curvs > 0), 0.5, 1.0)
    normalize_between(np.where(curvs <= 0), 0, 0.5)

    fig = plt.figure(figsize=(52, 28), dpi=150)
    nx_ax = fig.add_axes([0, 0, 1, 1])
    cb1_ax = fig.add_axes([0.8, 0.9, 0.15, 0.03])
    cb2_ax = fig.add_axes([0.8, 0.82, 0.15, 0.03])

    nodes_cmap = truncate_colormap(matplotlib.cm.get_cmap('Greys'), 0.2, 0.8)
    edges_cmap = matplotlib.cm.get_cmap('RdBu')
    kwargs = {
            'with_labels': False,
            'nodelist': nodes,
            'edgelist': edges,
            'ax': nx_ax,

            # node settings
            'node_size': 10,
            'cmap': nodes_cmap,
            'vmin': degrees.min(),
            'vmax': degrees.max(),
            'node_color': degrees,

            # edge settings
            'edge_cmap': matplotlib.cm.get_cmap('RdBu'),
            'edge_vmin': 0.0,
            'edge_vmax': 1.0,
            'edge_color': curvs,
            'width': 0.5
    }
    nx.draw_kamada_kawai(g, **kwargs)

    # Curvature colorbar
    cb1 = matplotlib.colorbar.ColorbarBase(
            cb1_ax, cmap=edges_cmap, orientation='horizontal')
    cb1.set_ticks([0, 0.5, 1.0])
    # we have to do this because of the curvature hack from above
    curv_ticks = [min(0, orig_curvs.min()), 0, max(0, orig_curvs.max())]
    cb1.set_ticklabels(floats_to_str(curv_ticks))
    cb1.outline.set_linewidth(0)
    cb1.ax.xaxis.set_tick_params(width=0)
    cb1.set_label('{} Curvature'.format(curv_name))

    # Node degreee colorbar
    cb2 = matplotlib.colorbar.ColorbarBase(
            cb2_ax,
            cmap=nodes_cmap,
            norm=matplotlib.colors.Normalize(
                    vmin=degrees.min(), vmax=degrees.max()),
            orientation='horizontal')
    cb2.outline.set_linewidth(0)
    cb2.ax.xaxis.set_tick_params(width=0)
    cb2.set_label('Node Degree')

    # save it
    fig.savefig(filename, bbox_inches='tight')


def main():
    g = nx.convert_node_labels_to_integers(nx.nx_pydot.read_dot(graph_file))
    n = g.number_of_nodes()
    nodes_to_keep = np.random.choice(
            n, int(n * nodes_to_keep_percentage), replace=False)

    # we do not need the full one from here on
    g = g.subgraph(nodes_to_keep).copy()
    print('#nodes: {}, #edges: {}, #conn-comps: {}'.format(
            g.number_of_nodes(), g.number_of_edges(),
            nx.number_connected_components(g)))

    # save both of them
    plot_graph_and_curvatures(g, 'ricciCurvature', 'Ollivier-Ricci',
                              'output/ollivier_graph.pdf')
    plot_graph_and_curvatures(g, 'formanCurvature', 'Forman-Ricci',
                              'output/forman_graph.pdf')


if __name__ == '__main__':
    sys.exit(main())
