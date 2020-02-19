import torch

from graphembed.linalg import torch_batch as tb
from .base import Manifold


class Stiefel(Manifold):

    def __init__(self, n, p, retr='svd'):
        self.n = n
        self.p = p

        # The retraction to use.
        if retr == 'qr':
            self.retr = self.retr_qr_
        elif retr == 'svd':
            self.retr = self.retr_svd_
        else:
            raise ValueError('Unknown retraction type {}'.format(retr))

    @property
    def ndim(self):
        return 2

    @property
    def dim(self):
        return self.p * self.n - self.p * (self.p + 1) // 2

    def zero(self, *shape, out=None):
        return torch.eye(self.n, self.p, out=out).repeat(*shape, 1, 1)

    def zero_vec(self, *shape, out=None):
        return torch.zeros(*shape, self.n, self.p, out=out)

    # NOTE: We use the embedded Riemannian structure, not the quotient one.
    def inner(self, x, u, v, keepdim=False):
        return (u * v).sum((-2, -1), keepdim=keepdim)

    # Ref: eq (2.4) in https://epubs.siam.org/doi/pdf/10.1137/S0895479895290954
    def proju(self, x, u, inplace=False):
        if not inplace:
            u = u.clone()
        xtu = tb.xty(x, u)
        symxtu = tb.sym(xtu)
        return u.sub_(x @ symxtu)

    def _orthonormalize(self, x):
        q, r = tb.qr(x)
        d = r.diagonal(dim1=-2, dim2=-1).sign_()
        return q.mul_(d.reshape(-1, 1, self.p))

    # Ref: manopt https://www.manopt.org/reference/manopt/manifolds/stiefel/stiefelfactory.html
    def projx(self, x, inplace=False):
        x_new = self._orthonormalize(x)
        if inplace:
            x.set_(x_new)
        return x

    def exp(self, x, u):
        return NotImplementedError

    def retr_qr_(self, x, u):
        return self._orthonormalize(x + u)

    def retr_svd_(self, x, u):
        y = x + u
        u, _, v = tb.svd(y, compute_uv=True)
        uvt = tb.xyt(u, v)
        return uvt

    def log(self, x, y):
        return NotImplementedError

    def dist(self, x, y, squared=False, keepdim=False):
        return NotImplementedError

    def rand(self, *shape, out=None, ir=1e-2):
        x = self.zero(*shape, out=out)
        u = self.randvec(x, norm=ir)
        return self.retr(x, u)

    def rand_uniform(self, *shape, out=None):
        return self.projx(
                torch.randn(*shape, self.n, self.p, out=out), inplace=True)

    def randvec(self, x, norm):
        u = torch.randn(x.shape, out=torch.empty_like(x))
        u = self.proju(x, u)  # "transport" ``u`` to ``x``
        u.div_(u.norm(dim=(-2, -1), keepdim=True)).mul_(norm)
        return u

    def __str__(self):
        return 'Stiefel manifold of {}x{} matrices'.format(self.n, self.p)
