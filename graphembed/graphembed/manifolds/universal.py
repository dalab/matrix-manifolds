import torch
from torch.nn.functional import softplus

from graphembed.utils import EPS

from .base import Manifold
from .impl import math


class Universal(Manifold, torch.nn.Module):

    def __init__(self, n, c_init=0.01, c_min=0.001, keep_sign_fixed=False):
        super().__init__()
        self.n = n
        self.c_min = c_min
        self.sign = None if not keep_sign_fixed else 1 if c_init > 0 else -1
        self.c = torch.nn.Parameter(torch.Tensor([c_init]))

    @property
    def ndim(self):
        return 1

    @property
    def dim(self):
        return self.n

    def get_c(self):
        if self.sign:
            return self.sign * (self.c_min + softplus(self.c))
        else:
            return self.c.sign() * self.c_min + self.c

    def get_K(self):
        return -self.get_c()

    def get_R(self):
        return 1.0 / torch.sqrt(torch.abs(self.get_c()))

    def zero(self, *shape, out=None):
        return torch.zeros(*shape, self.n, out=out)

    def zero_vec(self, *shape, out=None):
        return torch.zeros(*shape, self.n, out=out)

    def inner(self, x, u, v, keepdim=False):
        return math.inner(x, u, v, c=self.get_c(), keepdim=keepdim)

    def norm(self, x, u, squared=False, keepdim=False):
        norm = math.norm(x, u, keepdim=keepdim)
        if squared:
            norm.pow_(2)
        return norm

    def proju(self, x, u, inplace=False):
        return u

    def projx(self, x, inplace=False):
        x_new = math.project(x, c=self.get_c())
        if inplace:
            x.set_(x_new)
        return x

    def egrad2rgrad(self, x, u):
        return math.egrad2rgrad(x, u, c=self.get_c())

    def exp(self, x, u, project=True):
        res = math.expmap(x, u, c=self.get_c())
        if project:
            return math.project(res, c=self.get_c())
        else:
            return res

    def retr(self, x, u):
        return math.project(x + u, c=self.get_c())

    def log(self, x, y):
        return math.logmap(x, y, c=self.get_c())

    def dist(self, x, y, squared=False, keepdim=False):
        d = math.dist(x, y, c=self.get_c(), keepdim=keepdim)
        if squared:
            d.pow_(2)
        d.data.clamp_(min=EPS[x.dtype])
        return d

    def transp(self, x, y, u):
        return math.parallel_transport(x, y, u, c=self.get_c())

    def rand(self, *shape, out=None, ir=1e-2):
        x = torch.empty(*shape, self.n, out=out).uniform_(-ir, ir)
        return self.projx(x, inplace=True)

    def randvec(self, x, norm=1):
        raise NotImplementedError

    def __str__(self):
        return f'Universal {self.n}-dimensional manifold'
