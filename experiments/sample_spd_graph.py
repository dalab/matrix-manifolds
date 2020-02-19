r"""Early experiments with randomly sampled graphs on the SPD manifold.

References
----------
[1]: Pennec, Xavier, Pierre Fillard, and Nicholas Ayache. "A Riemannian
framework for tensor computing." International Journal of computer vision 66.1
(2006): 41-66.

[2]: Dolcetti, Alberto, and Donato Pertici. "Differential properties of spaces
of symmetric real matrices." arXiv preprint arXiv:1807.01113 (2018).

[3]: Sra, Suvrit, and Reshad Hosseini. "Conic geometric optimization on the
manifold of positive definite matrices." SIAM Journal on Optimization 25.1
(2015): 713-739.
"""
import collections
import math
import sys
import time

from GraphRicciCurvature.FormanRicci import formanCurvature as forman_curvature
from GraphRicciCurvature.OllivierRicci import ricciCurvature as ricci_curvature
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import networkx as nx
import numpy as np
import numpy.linalg as nl
from pymanopt.manifolds import PositiveDefinite
import scipy.linalg as sl

import sage.all
from sage.graphs.graph import Graph
import sage.graphs.hyperbolicity as sage_hyp

matrix_dim = 3
n_samples = 500
variance = None
manifold = PositiveDefinite(matrix_dim)


def dim_to_n(dim):
    return int((-1 + math.sqrt(1 + 8 * dim)) / 2)


def sym_to_vec(x):
    r"""The :math:`Vec(\cdot)` mapping from [1, Sec3.5]."""
    n = x.shape[0]
    y = np.copy(x)
    y[np.triu_indices(n, 1)] *= math.sqrt(2)

    return y[np.triu_indices(n)]


def vec_to_sym(x_vec):
    r"""The inverse of the :math:`Vec(\cdot)` mapping from [1, Sec3.5]."""
    n = dim_to_n(x_vec.shape[0])
    x = np.ndarray(shape=(n, n))

    x[np.triu_indices(n)] = x_vec
    idx_lower = np.tril_indices(n, -1)
    x[idx_lower] = x.T[idx_lower]
    # NOTE: This is needed because the off-diagonal elements are added twice
    # when using the usual trace inner product on the vector space of symmetric
    # matrices. See [1, Sec 3.5].
    x[~np.eye(n, dtype=bool)] /= math.sqrt(2)

    return x


def multilog(X):
    W, V = nl.eigh(X)
    W = np.expand_dims(np.log(W), axis=-1)
    X_log = V @ (W * np.einsum('...ji', V))

    return X_log


def pdist(X):
    C = nl.cholesky(X)
    C_inv = nl.inv(C)
    A = np.einsum('mij,njk,mlk->mnil', C_inv, X, C_inv)
    A_log = multilog(A)
    dists = nl.norm(A_log, axis=(2, 3))

    return dists


def continuous_ricci_curvature(x, u):
    r"""Computes :math:`Ric_x(u, u)`, the value of the Ricci tensor at the point
    x on the SPD manifold for the unit vector u.

    It is negative because of the inequality:

    .. math::
        \big[ \mathop{Tr} Y \big]^2 - n \mathop{Tr} Y^2 \le 0

    which holds for all symmetric matrices :math:`Y`. Additionally, when
    normalizing the tangent vectors u, the curvature is lower-bounded by
    :math:`\frac{-n}{4}`.

    See [2, Prop2.4].
    """
    n = x.shape[0]
    u = u / manifold.norm(x, u)
    x_inv_u = nl.solve(x, u)

    return 0.25 * (np.trace(x_inv_u)**2 - n * np.trace(x_inv_u @ x_inv_u))


def next_sample_and_curvature(x):
    r"""Samples a point on a manifold around another given point. It also
    returns the Ricci curvature that corresponds to the sampled direction.

    NOTE: When not normalizing the step fed into the exponential map, the degree
    power-law distribution is explained by the curvature of the manifold: for a
    few points, the unit vectors sampled uniformly at random from the unit
    sphere correspond to large steps.

    See [1, Sec3.8].
    """
    dim = int(manifold.dim)
    var = variance if variance is not None else 1.0
    u_vec = np.random.multivariate_normal(np.zeros(dim), var * np.eye(dim))
    if variance is None:
        u_vec = u_vec / nl.norm(u_vec)
    u = vec_to_sym(u_vec)
    # vector transport from identity (see [3]) + exponential map at x
    # (some matrix multiplications cancel out)
    x_sqrt = sl.sqrtm(x)
    x_next = x_sqrt @ sl.expm(u) @ x_sqrt

    # compute the actual curvature in the direction of the transported vector
    u_transp = x_sqrt @ u @ x_sqrt
    ricci_curv = continuous_ricci_curvature(x, u_transp)

    return x_next, ricci_curv


def random_walk_graph():
    # We use a (reversed) geometric distribution to (at least try to) unbias the
    # selection of the previous nodes used as reference for new ones.
    #
    # NOTE: If p is too large, longer "chains" tend to be created which takes us
    # to numerical unstable lands (i.e., very small or very large eigenvalues).
    p = 0.01
    samples = np.ndarray(shape=(n_samples, matrix_dim, matrix_dim))
    curvatures = np.ndarray(shape=(n_samples - 1, ))

    samples[0] = np.eye(matrix_dim)
    for i in range(1, n_samples):
        j = i - np.random.geometric(p)
        while j < 0:
            j = j + i
        samples[i], curvatures[i - 1] = next_sample_and_curvature(samples[j])

    # plot the actual Ricci curvatures for later inspection
    plot_curvatures(curvatures, 'ricci')

    # do a scatter plot of the representation of the points on the tangent space
    # of the identity matrix
    if matrix_dim == 2:
        X_log = multilog(samples)
        X_vec = np.array([sym_to_vec(x) for x in X_log])

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_vec[:, 0], X_vec[:, 1], X_vec[:, 2])
        fig.savefig('output/scatter.pdf', bbox_inches='tight')
        plt.close()

    return samples


def plot_distances(dists):
    plt.hist(dists[np.triu_indices(n_samples, 1)], bins=20)
    plt.xlim(xmin=0, xmax=15)
    plt.xlabel('Distance')
    plt.ylabel('Number of pairs')
    plt.savefig('output/dists.png')
    plt.close()


def plot_determinants(dets):
    plt.hist(np.log(dets), bins=20)
    plt.xlabel('Log-Determinant')
    plt.ylabel('Number of samples')
    plt.savefig('output/dets.png')
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

    dets = np.ndarray(shape=(n_samples, ))
    for i in range(n_samples):
        dets[i] = nl.det(samples[i])
    plot_determinants(dets)

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
    graph = ricci_curvature(graph, alpha=0.99, method='OTD', compute_nc=False)
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
