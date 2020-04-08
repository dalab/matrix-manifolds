import numpy as np
import torch

from graphembed.manifolds.base import Manifold


class Euclidean(Manifold):

    def __init__(self, *shape):
        self.shape = shape
        if len(shape) == 0:
            raise ValueError("Need shape parameters.")
        elif len(shape) == 1:
            self._name = "Euclidean manifold of {}-vectors".format(*shape)
        elif len(shape) == 2:
            self._name = "Euclidean manifold of {}x{} matrices".format(*shape)
        else:
            self._name = "Euclidean manifold of shape " + str(shape) + " tensors"
        self.dims = tuple(np.arange(-len(shape), 0))

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dim(self):
        return np.prod(self.shape)

    def zero(self, *shape, out=None):
        return torch.zeros(*shape, *self.shape, out=out)

    def zero_vec(self, *shape, out=None):
        return torch.zeros(*shape, *self.shape, out=out)

    def inner(self, x, u, v, keepdim=False):
        return (u * v).sum(self.dims, keepdim=keepdim)

    def proju(self, x, u, inplace=False):
        return u

    def projx(self, x, inplace=False):
        return x

    def exp(self, x, u):
        return x + u

    def log(self, x, y):
        return y - x

    def rand(self, *shape, out=None, ir=1e-2):
        return torch.empty(*shape, *self.shape, out=out).uniform_(-ir, ir)

    def randvec(self, x, norm):
        # vector distributed uniformly on the sphere of radius ``norm`` around x
        u = torch.randn(x.shape, out=torch.empty_like(x))
        u.div_(u.norm(dim=self.dims, keepdim=True)).mul_(norm)
        return u

    def __str__(self):
        return self._name
