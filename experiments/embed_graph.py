r""""
References
----------
[0]: Yuan, Xinru, et al. "A Riemannian quasi-Newton method for computing the
Karcher mean of symmetric positive definite matrices."
[1]: De Sa, Christopher, et al. "Representation tradeoffs for hyperbolic
embeddings."
"""
import argparse
import sys

import autograd.numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pymanopt import Problem
from pymanopt.manifolds import PositiveDefinite
from pymanopt.solvers import ConjugateGradient, SteepestDescent, TrustRegions
from pymanopt.solvers.linesearch import LineSearchAdaptive, LineSearchBackTracking

from metrics import average_distortion, mean_average_precision
from util import Timer


def config_parser(parser=None):
    if not parser:
        parser = argparse.ArgumentParser(description='Parse graph embedding.')
    parser.add_argument(
            '--input_graph',
            type=str,
            help='The path to the input graph in graphviz .dot format. '
            'Weighted graphs should use the edge property "weight".')
    parser.add_argument(
            '--input_dists',
            type=str,
            help='Input pairwise distances in numpy format.')
    parser.add_argument(
            '--manifold_dim',
            type=int,
            default=2,
            help='The dimension of the SPD manifold to embed to.')

    return parser


def multitrans(X):
    r"""Returns the tranpose of matrices stacked in an (...,n,n)-shaped array.
    """
    return np.einsum('...ji', X)


def multisym(X):
    r"""Returns the symmetrized version of X."""
    return 0.5 * (X + multitrans(X))


def multihgie(W, V):
    r"""The inverse of :py:`numpy.linalg.eigh` for stacked matrices. The name
    "hgie" is simply the string "eigh" reversed.
    """
    return np.einsum('...ij,...j,...kj->...ik', V, W, V)


def multipsd(X, f):
    r"""Template function acting on stacked matrices that applies a given
    analytic function on them via eigenvalue decomposition.

    Parameters
    ----------
    X : numpy.ndarray
        The (...,n,n)-shaped array containing the stacked matrices.
    f : callable
        A function that takes as input an (...,n)-shaped array with the
        corresponding eigenvalues and returns the same kind of array.
    """
    W, V = np.linalg.eigh(X)
    W = f(W)
    X_new = multihgie(W, V)

    return X_new


def multilog(X):
    r"""Computes the matrix-logarithm of several matrices at once. It uses a
    faster approach than :py:`scipy.linalg.logm`.

    Parameters
    ----------
    X : numpy.ndarray
        Several stacked matrices in an array of shape (...,n,n).
    """
    return multipsd(X, np.log)


class ReduceLROnPlateau(object):
    r"""A dummy line-search class that does not perform any search but instead
    uses a step size that is adapted once a plateau is reached.
    """

    import numpy as np  # no need for gradients here

    def __init__(self,
                 start_lr=1e-4,
                 factor=0.1,
                 patience=10,
                 threshold=1e-4,
                 verbose=0):
        self.lr = start_lr
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.verbose = verbose

        self.best = np.inf
        self.num_bad_epochs = 0

    def _retr(self, x, u):
        r"""The more efficient retraction used in [0]."""
        # multisym needed for numerical reasons
        return multisym(x + u + 0.5 * u @ np.linalg.solve(x, u))

    def search(self, objective, man, x, d, f0, df0):
        if f0 > self.best - self.threshold:
            self.num_bad_epochs += 1
            if self.num_bad_epochs == self.patience:
                self.lr *= self.factor
                self.num_bad_epochs = 0
                if self.verbose > 0:
                    print('Reducing learning rate to: ', self.lr)
        else:
            self.best = min(f0, self.best)
            self.num_bad_epochs = 0

        return self.lr, self._retr(x, self.lr * d)


def graph_pdists(g):
    r"""Computes the pairwise distances as taken on a possibly weighted graph.

    Parameters
    ----------
    g : networkx.Graph
        The graph to measure distances on. Graphs should use the 'weight' edge
        attribute to assign weights.
    """
    # All-paths Dijkstra has complexity O(|V| |E| log|V|) which is better than
    # Floyd-Warshall for sparse graphs.
    # NOTE: Dijkstra requires all weights to be positive.
    assert all([w > 0 for w in nx.get_edge_attributes(g, 'weight').values()])

    n = g.number_of_nodes()
    shortest_paths = nx.all_pairs_dijkstra_path_length(g)
    g_pdists = np.ndarray(shape=(n, n))
    for u, dists in shortest_paths:
        for v, d in dists.items():
            g_pdists[u, v] = d

    return g_pdists


def load_pdists(args):
    if args.input_graph:
        with Timer('graph loading') as t:
            if args.input_graph.endswith('.dot'):
                g = nx.nx_pydot.read_dot(args.input_graph)
            else:
                # assume it's list of edges
                g = nx.read_edgelist(args.input_graph)
            g = nx.convert_node_labels_to_integers(g)
            print('#nodes: ', g.number_of_nodes())
            print('#edges: ', g.number_of_edges())
        with Timer('computing graph distances') as t:
            return graph_pdists(g), g
    elif args.input_dists:
        with Timer('loading pairwise distances from file'):
            return np.load(args.input_dists), None

    raise ValueError('Either a graph in .dot format or a pairwise '
                     'distances should be given')


def manifold_pdists(X, squared=False):
    r"""Computes the pairwise distances between N points on the SPD manifold.
    It uses a faster approach than calling
    :py:`pymanopt.manifolds.PositiveDefinite.dist`
    :math:`\frac{n (n + 1)}{2}`-times. It uses the fact that the inverse of the
    square root of a SPD matrix can be computed via Cholesky decomposition and
    inversion, and this needs to be done only once for each matrix.

    The distance function is

    .. math::
        d(A, B) = \lVert \log(A^{-1/2} B A^{-1/2}) \rVert_F

    Parameters
    ----------
    X : numpy.ndarray
        The SPD matrices to compute distances between, given as an
        (n,d,d)-shaped array.
    squared : bool
        Whether the squared distances should be returned. This is given as a
        parameter as opposed to leaving it up to the caller to square because
        returning the squared distances directly is better for automatic
        differentiation than returning the distances and manually squaring.
    """
    mask = np.triu_indices(X.shape[0], 1)

    # -> first, compute X_i^{-1/2} X_j X_i^{-1/2} for 1<=i<j<= n
    C = np.linalg.cholesky(X)
    C_inv = np.linalg.inv(C)
    C_mul = C_inv[mask[0], :, :]  # avoids duplicating the following --
    X_mul = X[mask[1], :, :]  # computations and does not mess up autograd
    A = np.einsum('mij,mjk,mlk->mil', C_mul, X_mul, C_mul)

    # -> then, compute the matrix logarithm and its squared Frobenius norm
    A_log = multilog(A)
    pdists = (A_log**2).sum(axis=(1, 2))

    return pdists if squared else np.sqrt(pdists)


def pdists_vec_to_sym(v, n):
    r"""Aranges the :math:`\frac{n (n - 1)}{2}` distances in a symmetric matrix
    with zeros on the diagonal.
    """
    x = np.zeros(shape=(n, n))
    x[np.triu_indices(n, 1)] = v
    x = x + x.T

    return x


def sample_init_points(n, d):
    r"""Function used to initialize the optimization by sampling a matrix of
    uniform numbers between 0 and 1 and then adjusting their eigenvalues to be
    positive by taking the absolute value and adding 1.
    """
    import numpy as np

    X = np.random.rand(n, d, d)
    X = (X + multitrans(X)) / 2
    X = multipsd(X, lambda W: np.abs(W) + 1)

    return X


def main():
    r"""Main entry point in the graph embedding procedure."""
    args = config_parser().parse_args()

    g_pdists = load_pdists(args)
    n = g_pdists.shape[0]
    d = args.manifold_dim

    # we are actually using only the upper diagonal part
    g_pdists = g_pdists[np.triu_indices(n, 1)]
    g_sq_pdists = g_pdists**2

    # read the graph
    # the distortion cost
    def distortion_cost(X):
        man_sq_pdists = manifold_pdists(X, squared=True)

        return np.sum(np.abs(man_sq_pdists / g_sq_pdists - 1))

    # the manifold, problem, and solver
    manifold = PositiveDefinite(d, k=n)
    problem = Problem(manifold=manifold, cost=distortion_cost, verbosity=2)
    linesearch = ReduceLROnPlateau(
            start_lr=2e-2, patience=10, threshold=1e-4, factor=0.1, verbose=1)
    solver = ConjugateGradient(linesearch=linesearch, maxiter=1000)

    # solve it
    with Timer('training') as t:
        X_opt = solver.solve(problem, x=sample_init_points(n, d))

    # the distortion achieved
    man_pdists = manifold_pdists(X_opt)
    print('Average distortion: ', average_distortion(g_pdists, man_pdists))
    man_pdists_sym = pdists_vec_to_sym(man_pdists, n, 1e12)
    print('MAP: ', mean_average_precision(
            g, man_pdists_sym, diag_adjusted=True))


if __name__ == '__main__':
    sys.exit(main())
