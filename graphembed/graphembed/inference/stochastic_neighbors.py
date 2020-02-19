import torch
from torch.nn.functional import softmax

from .base import StochasticGraphModel
from graphembed.utils import nnm1d2_to_n, triu_mask


class StochasticNeighbor(StochasticGraphModel):

    def __call__(self, theta, ret_margs=False):
        # Put it into an `n x n-1` matrix.
        n = nnm1d2_to_n(theta.shape[0])
        tm = triu_mask(n, n - 1, device=theta.device)
        theta_mat = torch.empty(n, n - 1, out=theta.new(n, n - 1))
        theta_mat.masked_scatter_(tm, theta)
        theta_mat.T.masked_scatter_(~tm.T, theta)

        logz = torch.logsumexp(theta_mat, dim=1).sum()
        if not ret_margs:
            return logz, None

        margs = softmax(theta_mat, dim=1)
        margs = margs.masked_select(tm) + margs.T.masked_select(~tm.T)
        return logz, margs


class StochasticNeighbors(StochasticGraphModel):

    def __init__(self, k):
        if k < 2:
            raise ValueError('k should be greater or equal to 2. For k=1 '
                             'use `StochasticNeighbor` instead.')
        self.k = k

    def __call__(self, theta, ret_margs=False):
        raise NotImplementedError
