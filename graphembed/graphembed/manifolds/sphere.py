import numpy as np
import torch

from graphembed.manifolds.base import Manifold
from graphembed.utils import EPS


class Sphere(Manifold):

    def __init__(self, *shape):
        self.shape = shape
        if len(shape) == 0:
            raise ValueError("Need shape parameters.")
        elif len(shape) == 1:
            self._name = "Sphere manifold of {}-vectors".format(*shape)
        elif len(shape) == 2:
            self._name = "Sphere manifold of {}x{} matrices".format(*shape)
        else:
            self._name = "Sphere manifold of shape " + str(shape) + " tensors"
        self.dims = tuple(np.arange(-len(shape), 0))

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dim(self):
        return np.prod(self.shape) - 1

    def zero(self, *shape, out=None):
        x = torch.zeros(*shape, np.prod(self.shape), out=out)
        x[..., 0] = -1
        return x.reshape(*shape, *self.shape)

    def zero_vec(self, *shape, out=None):
        return torch.zeros(*shape, *self.shape, out=out)

    def inner(self, x, u, v, keepdim=False):
        return (u * v).sum(self.dims, keepdim=keepdim)

    def proju(self, x, u, inplace=False):
        if not inplace:
            u = u.clone()
        return u.addcmul_(-self.inner(None, x, u, keepdim=True), x)

    def projx(self, x, inplace=False):
        if not inplace:
            x = x.clone()
        return x.div_(self.norm(None, x, keepdim=True))

    def exp(self, x, u):
        norm_u = self.norm(None, u, keepdim=True)
        expx_u = x * torch.cos(norm_u) + u * torch.sin(norm_u) / norm_u
        retrx_u = self.retr(x, u)
        cond = norm_u > EPS[x.dtype]
        return torch.where(cond, expx_u, retrx_u)

    def retr(self, x, u):
        return self.projx(x + u, inplace=True)

    def log(self, x, y):
        u = self.proju(x, y - x, inplace=True)
        dist = self.dist(x, y, keepdim=True)
        # If the two points are "far apart", correct the norm.
        cond = dist.gt(EPS[x.dtype])
        return torch.where(cond, u * dist / self.norm(None, u, keepdim=True), u)

    def dist(self, x, y, squared=False, keepdim=False):
        inner = self.inner(None, x, y, keepdim=keepdim)
        inner.data.clamp_(min=-1 + EPS[x.dtype]**2, max=1 - EPS[x.dtype]**2)
        sq_dist = torch.acos(inner)
        sq_dist.data.clamp_(min=EPS[x.dtype])

        return sq_dist.pow(2) if squared else sq_dist

    def rand(self, *shape, out=None, ir=1e-2):
        x = self.zero(*shape, out=out)
        u = self.randvec(x, norm=ir)
        return self.retr(x, u)

    def rand_uniform(self, *shape, out=None):
        return self.projx(
                torch.randn(*shape, *self.shape, out=out), inplace=True)

    def rand_ball(self, *shape, out=None):
        xs_unif = self.rand_uniform(*shape, out=out)
        rs = torch.rand(*shape).pow_(1 / (self.dim + 1))
        rs = rs.reshape(*shape, *((1, ) * len(self.shape)))
        xs_ball = xs_unif.mul_(rs)
        return xs_ball

    def randvec(self, x, norm=1):
        u = torch.randn(x.shape, out=torch.empty_like(x))
        u = self.proju(x, u, inplace=True)  # "transport" ``u`` to ``x``
        u.div_(u.norm(dim=self.dims, keepdim=True)).mul_(norm)  # normalize
        return u

    def __str__(self):
        return self._name
