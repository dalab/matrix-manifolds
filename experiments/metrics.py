import numpy as np


def average_distortion(g_pdists, m_pdists):
    r"""The average distortion used to measure the quality of the embedding.
    See, e.g., [1].

    Parameters
    ----------
    g_pdists : numpy.ndarray
        Pairwise distances on the graph, as an (n*(n-1)//2,)-shaped array.
    m_pdists : numpy.ndarray
        Pairwise distances on the manifold, as an (n*(n-1)//2,)-shaped array.
    """
    return np.mean(np.abs(m_pdists - g_pdists) / g_pdists)


def mean_average_precision(g, m_pdists):
    r"""The MAP as defined in [1]. The complexity is squared in the number of
    nodes.
    """
    n = m_pdists.shape[0]
    assert n == g.number_of_nodes()

    ap_scores = []
    for u in g.nodes():
        sorted_nodes = np.argsort(m_pdists[u])
        neighs = set(g.neighbors(u))
        n_correct = 0.0
        precisions = []
        for i in range(1, n):
            if sorted_nodes[i] in neighs:
                n_correct += 1
                precisions.append(n_correct / i)

        ap_scores.append(np.mean(precisions))

    return np.mean(ap_scores)
