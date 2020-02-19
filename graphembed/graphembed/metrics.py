import numpy as np
import scipy.stats
import torch

from graphembed.utils import squareform1


def spearmanr(x, y):
    """Mimics :func:`scipy.stats.spearmanr`."""
    return scipy.stats.spearmanr(x.cpu().numpy(), y.cpu().numpy()).correlation


def pearsonr(x, y):
    """Mimics :func:`scipy.stats.pearsonr`."""
    xm = x - x.mean()
    ym = y - y.mean()
    return xm @ ym / (xm.norm() * ym.norm())


def average_pearsonr(mpdists, gpdists):
    """Computes the correlation factor on a per-node basis and then takes the
    average.
    """
    mpdists = squareform1(mpdists)
    gpdists = squareform1(gpdists)

    mpdists.sub_(mpdists.mean(axis=1))
    gpdists.sub_(gpdists.mean(axis=1))
    rs_num = (mpdists * gpdists).sum(axis=1)
    rs_denom = (mpdists.norm(dim=1) * gpdists.norm(dim=1))

    return (rs_num / rs_denom).mean()


def area_under_curve(vs, step=None):
    r"""The area under graphs of per-node or per-layer metrics."""
    if step is None:
        step = len(vs)
    aucs = []
    for i in range(0, len(vs) // step * step, step):
        auc = 0.5 * np.mean(vs[(i + 1):(i + step)] + vs[i:(i + step - 1)])
        aucs.append(auc)
    return aucs


def average_distortion(mpdists, gpdists):
    r"""The average distortion.

    Parameters
    ----------
    mpdists : torch.Tensor
        Pairwise distances on the manifold.
    gpdists : torch.Tensor
        Pairwise distances on the graph.
    """
    return torch.mean(torch.abs(mpdists - gpdists) / gpdists)


# NOTE: The version from :module:`graphembed.pyx.precision` is much faster. We
# keep this one too in order to track regressions.
def py_mean_average_precision(mpdists, g, pool=None):
    r"""The (local) mean average precision.

    Parameters
    ----------
    mpdists : numpy.ndarray
        Pairwise distances on the manifold, as an (n,n)-shaped symmetric tensor
        with zeros on the diagonal.
    g : networkx.Graph
        The input graph.
    pool : concurrent.futures.Executor, optional
        An executor used to parallelize the per-node precision computation.
    """
    n = mpdists.shape[0]
    assert n == g.number_of_nodes()

    def average_precision(u):
        sorted_nodes = np.argsort(mpdists[u])
        neighs = set(g.neighbors(u))
        n_neighs = len(neighs)
        precision_sum = 0
        n_correct = 0
        for i in range(1, n):
            if sorted_nodes[i] in neighs:
                n_correct += 1
                precision_sum += n_correct / i
                if n_correct == len(neighs):
                    break
        return precision_sum / n_neighs

    if pool is not None:
        ap_scores = pool.map(average_precision, g.nodes())
    else:
        ap_scores = [average_precision(u) for u in g.nodes()]

    return np.mean(ap_scores)
