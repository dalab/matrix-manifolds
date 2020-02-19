import torch

from .base import StochasticGraphModel
from graphembed.linalg import torch_batch as tb
from graphembed.utils import nnm1d2_to_n, squareform1


class StochasticTrees(StochasticGraphModel):

    def __call__(self, theta, ret_margs=False):
        n_nodes = nnm1d2_to_n(len(theta))
        theta.requires_grad_()  # Needed by `torch.grad` below!

        # Construct the Laplacian.
        L_off = squareform1(torch.exp(theta))
        L_diag = torch.diag(L_off.sum(1))
        L = L_diag - L_off
        L = L[1:, 1:]

        logz = tb.plogdet(L)
        if not ret_margs:
            return logz, None

        margs, = torch.autograd.grad(logz, theta, create_graph=True)
        with torch.no_grad():
            assert abs(margs.sum() - (n_nodes - 1)) < 1e-4
        return logz, margs
