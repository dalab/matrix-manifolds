import math
import torch

from graphembed.linalg import fast, torch_batch as tb
from graphembed.utils import EPS

from .base import Manifold


class Grassmann(Manifold):

    # NOTE: The 'svd' retraction based on polar decomposition is compatible with
    # the default transprot. Using 'qr' might lead to issues.
    def __init__(self, n, p, retr='svd', requires_grad=True):
        self.n = n
        self.p = p
        self.requires_grad = requires_grad

        # The retraction to use.
        if retr == 'qr':
            self.retr = self.retr_qr_
        elif retr == 'svd':
            self.retr = self.retr_svd_
        else:
            raise ValueError('Unknown retraction type {}'.format(retr))

    def _singular_values_pxp(self, x):
        if self.p == 2:
            return fast.singular_values_2x2(x)
        return tb.svd(x, compute_uv=self.requires_grad).S

    @property
    def ndim(self):
        return 2

    @property
    def dim(self):
        return self.p * (self.n - self.p)

    def zero(self, *shape, out=None):
        return torch.eye(self.n, self.p, out=out).repeat(*shape, 1, 1)

    def zero_vec(self, *shape, out=None):
        return torch.zeros(*shape, self.n, self.p, out=out)

    def inner(self, x, u, v, keepdim=False):
        return (u * v).sum((-2, -1), keepdim=keepdim)

    def proju(self, x, u, inplace=False):
        if not inplace:
            u = u.clone()
        xxtu = torch.einsum('...ij,...kj,...kl->...il', x, x, u)
        return u.sub_(xxtu)

    def projx(self, x, inplace=False):
        # Represent the column space via an orthogonal matrix instead.
        x_new = tb.qr(x).Q
        if inplace:
            x.set_(x_new)
        return x

    def exp(self, x, u):
        us, ss, vs = tb.svd(u, compute_uv=True)
        cos_s = ss.cos()  # Not inplace!
        sin_s = ss.sin_()
        lhs = x @ tb.mvmt(vs, cos_s, vs)
        rhs = tb.mvmt(us, sin_s, vs)
        return lhs.add_(rhs)

    def retr_qr_(self, x, u):
        y = x + u
        q = tb.qr(y).Q
        return q

    def retr_svd_(self, x, u):
        y = x + u
        u, _, v = tb.svd(y, compute_uv=True)
        uvt = tb.xyt(u, v)
        return uvt

    def log(self, x, y):
        ytx = tb.xty(y, x)
        At = tb.transpose(y) - ytx @ tb.transpose(x)
        Bt = torch.solve(At, ytx).solution
        us, ss, vs = tb.svd(tb.transpose(Bt))
        return tb.mvmt(us[..., :self.p],
                       ss[..., :self.p].atan(),
                       vs[..., :self.p])  # yapf: disable

    def dist(self, x, y, squared=False, keepdim=False):
        xty = tb.xty(x, y)
        s = self._singular_values_pxp(xty)
        s.data.clamp_(min=-1 + EPS[x.dtype]**2, max=1 - EPS[x.dtype]**2)
        dist_sq = s.acos().pow(2).sum(dim=-1, keepdim=keepdim)
        return dist_sq if squared else dist_sq.sqrt()

    # NOTE: The following are identical to :class:`graphembed.manifolds.Sphere`.

    def rand(self, *shape, out=None, ir=1e-2):
        x = self.zero(*shape, out=out)
        u = self.randvec(x, norm=ir)
        return self.exp(x, u)

    def rand_uniform(self, *shape, out=None):
        return self.projx(
                torch.randn(*shape, self.n, self.p, out=out), inplace=True)

    def randvec(self, x, norm):
        u = torch.randn(x.shape, out=torch.empty_like(x))
        u = self.proju(x, u)  # "transport" ``u`` to ``x``
        u.div_(u.norm(dim=(-2, -1), keepdim=True)).mul_(norm)
        return u

    def __str__(self):
        return 'Grassmann manifold of {}x{} matrices'.format(self.n, self.p)
