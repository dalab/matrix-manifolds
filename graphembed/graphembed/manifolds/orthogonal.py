import scipy.linalg as sl
import torch

from graphembed.linalg import torch_batch as tb
from .stiefel import Stiefel


class OrthogonalGroup(Stiefel):

    def __init__(self, n, *args, **kwargs):
        super().__init__(n, n, *args, **kwargs)

    def dist(self, x, y, squared=False, keepdim=False):
        if not torch.all(torch.abs(x.det() - y.det()) < 1e-8):
            raise ValueError('The distance function is defined for matrices '
                             'in the same connected component only.')
        xty = tb.xty(x, y)
        eigs = torch.stack([z.eig().eigenvalues for z in xty])
        theta = torch.atan2(eigs[..., 1], eigs[..., 0])
        dist_sq = theta.pow_(2).sum(-1, keepdim=keepdim)
        return dist_sq if squared else dist_sq.sqrt_()


class SpecialOrthogonalGroup(OrthogonalGroup):

    def _log_np(self, x, y):
        p = sl.logm(x.T @ y)
        return x @ p

    def log(self, x, y):
        if x.ndim == 2:
            assert y.ndim == 2
            return torch.as_tensor(self._log_np(x.numpy(), y.numpy()))
        return torch.stack([
                torch.as_tensor(self._log_np(xi.numpy(), yi.numpy()))
                for xi, yi in zip(x, y)
        ])

    def rand_uniform(self, *shape, out=None):
        xs = super().rand_uniform(*shape, out=out)
        ds = xs.det()
        ds.unsqueeze_(-1)
        xs[..., 0].mul_(ds)
        return xs
