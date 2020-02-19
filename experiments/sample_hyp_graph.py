"""
References
----------
[1]: Nagano, Yoshihiro, et al. "A Wrapped Normal Distribution on Hyperbolic
Space for Gradient-Based Learning." International Conference on Machine
Learning. 2019.
"""
import collections
import sys
import time

from GraphRicciCurvature.FormanRicci import formanCurvature as forman_curvature
from GraphRicciCurvature.OllivierRicci import ricciCurvature as ricci_curvature
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.linalg as nl

import sage.all
from sage.graphs.graph import Graph
import sage.graphs.hyperbolicity as sage_hyp

hyperboloid_dim = 3
n_samples = 500
variance = None

# the origin of the Hyperboloid manifold
origin = np.insert(np.zeros(hyperboloid_dim), 0, 1)


def minkowski_form(x, y):
    return -x[0] * y[0] + np.dot(x[1:], y[1:])


def pdist(X):
    inner_prods = -np.outer(X[:, 0], X[:, 0]) + X[:, 1:] @ X[:, 1:].T
    dists = np.arccosh(np.clip(-inner_prods, a_min=1.0, a_max=None))

    return dists


def next_sample(x):
    r"""Same algorithm as in :py:`./sample_spd_graph.py` but for the hyperboloid
    model of the hyperbolic space. The implementation follows [1].
    """
    n = hyperboloid_dim

    # steps 1 & 2
    var = variance if variance is not None else 1.0
    v = np.random.multivariate_normal(np.zeros(n), var * np.eye(n))
    if variance is None:
        v = v / nl.norm(v)
    v = np.insert(v, 0, 0)

    # step 3: transport
    alpha = -minkowski_form(origin, x)
    frac = minkowski_form(x - alpha * origin, v) / (alpha + 1)
    v_transp = v + frac * (origin + x)

    # step 4: project
    norm_v = np.sqrt(minkowski_form(v_transp, v_transp))  # the norm
    z = np.cosh(norm_v) * x + (np.sinh(norm_v) / norm_v) * v_transp

    return z


def random_walk_graph():
    # We use a (reversed) geometric distribution to (at least try to) unbias the
    # selection of the previous nodes used as reference for new ones.
    #
    # NOTE: If p is too large, longer "chains" tend to be created which takes us
    # to numerical unstable lands (i.e., very small or very large eigenvalues).
    p = 0.01
    samples = np.ndarray(shape=(n_samples, hyperboloid_dim + 1))

    samples[0] = origin
    for i in range(1, n_samples):
        j = i - np.random.geometric(p)
        while j < 0:
            j = j + i
        samples[i] = next_sample(samples[j])

    return samples


def plot_distances(dists):
    plt.hist(dists[np.triu_indices(n_samples, 1)], bins=20)
    plt.xlim(xmin=0, xmax=15)
    plt.xlabel('Distance')
    plt.ylabel('Number of pairs')
    plt.savefig('output/dists.png')
    plt.close()


def plot_curvatures(curvatures, name):
    plt.hist(curvatures, bins=20)
    plt.xlabel('Curvature')
    plt.ylabel('Edges')
    plt.savefig('output/{}_curvatures.png'.format(name))
    plt.close()


def plot_degree_distribution(graph, name):
    degrees = sorted([d for n, d in graph.degree()], reverse=True)
    counts = collections.Counter(degrees)
    deg, cnt = zip(*counts.items())
    plt.bar(deg, cnt)
    plt.xlabel('Node Degree')
    plt.ylabel('#nodes')
    plt.savefig('output/{}_degrees.png'.format(name))
    plt.close()


def main():
    start = time.time()
    samples = random_walk_graph()
    end = time.time()
    print('Time taken to generate the graph: {:.2f}s'.format(end - start))

    start = time.time()
    dists = pdist(samples)
    end = time.time()
    print('Time taken to compute distances: {:.2f}s'.format(end - start))
    plot_distances(dists)

    # NOTE: Using (1 + eps) here as a "neighborhood threshold" makes the most
    # sense. That's because our sampling yields nodes at distance 1 from the
    # reference node.
    graph = nx.Graph()
    graph.add_edges_from(np.argwhere(dists < 1.0001))
    graph.remove_edges_from(nx.selfloop_edges(graph))

    # plots, summaries
    plot_degree_distribution(graph, 'g')
    print('The number of edges is: ', graph.number_of_edges())
    print('The number of connected components is: ',
          nx.number_connected_components(graph))

    # curvatures
    graph = ricci_curvature(graph, alpha=0.99, method='OTD', compute_nc=True)
    graph = forman_curvature(graph)
    ollivier_curvatures = []
    forman_curvatures = []
    for _, _, attrs in graph.edges(data=True):
        ollivier_curvatures.append(attrs['ricciCurvature'])
        forman_curvatures.append(attrs['formanCurvature'])
    plot_curvatures(np.array(ollivier_curvatures), 'ollivier')
    plot_curvatures(np.array(forman_curvatures), 'forman')

    # hyperbolicity
    sage_graph = Graph(graph)
    ss = int(1e7)
    h_dict = sage_hyp.hyperbolicity_distribution(sage_graph, sampling_size=ss)
    h, _, _ = sage_hyp.hyperbolicity(sage_graph, algorithm='BCCM')
    print('The graph hyperbolicity is: ', h)
    # plot it
    values = [h.n() for h, _ in h_dict.items()]
    counts = [c.n() for _, c in h_dict.items()]
    plt.bar(values, counts, align='center', width=0.25)
    plt.xticks(np.unique(values))
    plt.xlabel('Hyperbolicity')
    plt.ylabel('Relative count')
    plt.savefig('output/hyperbolicity.png')

    # save it for later processing
    nx.nx_pydot.write_dot(graph, 'output/graph.dot')


if __name__ == '__main__':
    sys.exit(main())
