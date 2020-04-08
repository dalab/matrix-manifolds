r"""The manifold of symmetric positive definite matrices.

References
----------
[1]: Pennec, Xavier, Pierre Fillard, and Nicholas Ayache. "A Riemannian
framework for tensor computing." International Journal of computer vision 66.1
(2006): 41-66.
[2]: Sra, Suvrit. "Positive definite matrices and the symmetric Stein
divergence." preprint (2011).
"""
import math

import torch
from torch.autograd import Function

from graphembed.manifolds.base import Manifold
from graphembed.linalg import fast, torch_batch as tb
from graphembed.utils import squareform0, triu_mask


class SymmetricPositiveDefinite(Manifold):

    def __init__(self,
                 n,
                 *,
                 fast_symeig=True,
                 fast_chol=True,
                 use_stein_div=False,
                 wmin=1e-8,
                 wmax=1e8):
        self.n = n
        self.wmin = wmin
        self.wmax = wmax

        # symeig function
        if n == 2 and fast_symeig:
            self.symeig = fast.symeig2x2
        elif n == 3 and fast_symeig:
            self.symeig = fast.symeig3x3
        else:
            self.symeig = self._default_symeig

        # invchol function
        if n == 2 and fast_chol:
            self.chol = fast.cholesky2x2
            self.invchol = fast.invcholesky2x2
        else:
            self.chol = tb.cholesky
            self.invchol = self._default_invchol

        if use_stein_div:
            self.dist = self.stein_div
            self.pdist = self.stein_pdiv

    def _default_invchol(self, x, ret_chol=False):
        l = self.chol(x)
        eye = torch.eye(self.n, out=x.new(self.n, self.n))
        l_inv = torch.triangular_solve(eye, l, upper=False).solution
        if not ret_chol:
            return l_inv, None
        return l_inv, l

    def _default_symeig(self, x):
        return tb.symeig(x, eigenvectors=True).eigenvalues

    @staticmethod
    def to_vec(x):
        r"""The :math:`Vec(\cdot)` mapping from [1, Sec3.5]."""
        n = x.shape[-1]
        fact = x.new(n, n).fill_(math.sqrt(2)).fill_diagonal_(1.0)
        x_vec = fact.mul_(x).masked_select(triu_mask(n, device=x.device))

        return x_vec

    @staticmethod
    def from_vec(x_vec):
        r"""The inverse of the :math:`Vec(\cdot)` mapping from [1, Sec3.5]."""
        x = squareform0(x_vec / math.sqrt(2))
        x.diagonal(dim1=-2, dim2=-1)[:] *= math.sqrt(2)

        return x

    @property
    def ndim(self):
        return 2

    @property
    def dim(self):
        return self.n * (self.n + 1) // 2

    def zero(self, *shape, out=None):
        return torch.eye(self.n, out=out).repeat(*shape, 1, 1)

    def zero_vec(self, *shape, out=None):
        return torch.zeros(*shape, self.n, self.n, out=out)

    def inner(self, x, u, v, keepdim=False):
        # FIXME(ccruceru): This is not currently differentiable.
        assert not x.requires_grad and \
                not u.requires_grad and \
                not v.requires_grad

        l = self.chol(x)
        x_inv_u = torch.cholesky_solve(u, l)
        x_inv_v = torch.cholesky_solve(v, l)
        return tb.trace(x_inv_u @ x_inv_v, keepdim=keepdim)

    def _lult(self, x, u, ret_chol=False):
        l_inv, l = self.invchol(x, ret_chol=ret_chol)
        lult = tb.axat(l_inv, u)
        return lult, l

    def norm(self, x, u, squared=False, keepdim=False):
        lult, _ = self._lult(x, u)
        norm_sq = lult.pow(2).sum((-2, -1), keepdim=keepdim)

        return norm_sq if squared else norm_sq.sqrt()

    def proju(self, x, u, inplace=False):
        u_new = tb.sym(u)
        if not inplace:
            return u_new
        u.set_(u_new)
        return u

    def projx(self, x, inplace=False):
        x_new = tb.symapply(
                tb.sym(x), lambda w: w, wmin=self.wmin, wmax=self.wmax)
        if not inplace:
            return x_new
        x.set_(x_new)
        return x

    def egrad2rgrad(self, x, u):
        return tb.axat(x, self.proju(x, u))

    def exp(self, x, u):
        # TODO(ccruceru): Replace this with :math:`X \exp(X^{-1} U)` once
        # general matrix exponential is supported.
        lult, l = self._lult(x, u, ret_chol=True)
        exp_lult = tb.symexpm(lult)
        expx_u = tb.axat(l, exp_lult)

        return expx_u

    def retr(self, x, u):
        # Computing :math:`X + U + \frac{1}{2} U X^{-1} U` as:
        # :math:`U X^{-1} U = U (L L^T)^{-1} U = (L^{-1} U)^T (L^{-1} U)`.
        l = self.chol(x)
        l_inv_u = torch.triangular_solve(u, l, upper=False).solution
        u_xinv_u = tb.xty(l_inv_u, l_inv_u)
        y = tb.sym(x + u + 0.5 * u_xinv_u)

        return y

    def log(self, x, y):
        lylt, l = self._lult(x, y, ret_chol=True)
        log_lult = tb.spdlogm(lylt)
        logx_y = tb.axat(l, log_lult)

        return logx_y

    def _norm_log(self, x, squared=False, keepdim=False):
        w = self.symeig(x)
        w.data.clamp_(min=self.wmin, max=self.wmax)
        dist_sq = w.log().pow(2).sum(-1, keepdim=keepdim)
        dist_sq.data.clamp_(min=self.wmin)

        return dist_sq if squared else dist_sq.sqrt()

    def dist(self, x, y, squared=False, keepdim=False):
        lylt, _ = self._lult(x, y)
        return self._norm_log(lylt, squared=squared, keepdim=keepdim)

    def pdist(self, x, squared=False):
        assert x.ndim == 3
        n = x.shape[0]
        l_inv, _ = self.invchol(x)
        m = torch.triu_indices(n, n, 1, device=x.device)
        lylt = tb.axat(l_inv[m[0]], x[m[1]])
        return self._norm_log(lylt, squared=squared)

    def stein_div(self, x, y, squared=False, keepdim=False):
        logdet_x = tb.plogdet(x, chol=self.chol, keepdim=keepdim)
        logdet_y = tb.plogdet(y, chol=self.chol, keepdim=keepdim)
        logdet_xpy = tb.plogdet(0.5 * (x + y), chol=self.chol, keepdim=keepdim)
        div = logdet_xpy - 0.5 * (logdet_x + logdet_y)
        div.data.clamp_(min=self.wmin)
        return div if squared else div.sqrt()

    def stein_pdiv(self, x, squared=False):
        div = PairwiseSteinDivergence.apply(x, self.chol)
        div.data.clamp_(min=self.wmin)
        return div if squared else div.sqrt()

    def transp(self, x, y, u):
        # TODO(ccruceru): Implement the correct one once we have general matrix
        # square root.  See :func:`~randvec` below.
        return u

    def rand(self, *shape, out=None, ir=1e-1):
        # FIXME: Make it in-place to use `out` correctly. Currently we only used
        # it to pass device/dtype settings.
        eyes = self.zero(*shape, out=out)
        u = torch.randn(*shape, self.dim, out=eyes.new())
        u.div_(u.norm(dim=-1, keepdim=True)).mul_(ir)
        u = self.from_vec(u)
        return self.exp(eyes, u)

    def randvec(self, x, norm=1):
        # The tangent vector is :math:`X^{1/2} U X^{1/2}`, where :math:`U` is a
        # symmetric matrix uniformly sampled from the unit sphere (in the
        # corresponding vector space of symmetric matrices).
        # This corresponds to the parallel transport of ``U`` from the identity
        # matrix to ``X``.
        shape = x.shape[:-2] + (self.dim, )
        u = torch.randn(shape, out=x.new(shape))
        u.div_(u.norm(dim=-1, keepdim=True)).mul_(norm)
        u = self.from_vec(u)
        x_sqrt = tb.spdsqrtm(x, wmin=self.wmin, wmax=self.wmax)
        return tb.axat(x_sqrt, u)

    def seccurv(self, x, u, v):
        l = self.chol(x)
        x_inv_u = torch.cholesky_solve(u, l)
        x_inv_v = torch.cholesky_solve(v, l)
        prod = x_inv_u @ x_inv_v
        lb = prod - tb.transpose(prod)  # Lie bracket
        rt = lb.pow_(2).sum((-2, -1), keepdim=True).mul_(-0.25)  # Riem. tensor
        assert all(rt < self.wmin)
        rt.data.clamp_(max=-self.wmin)

        u_norm_sq = tb.trace(x_inv_u @ x_inv_u, keepdim=True)
        v_norm_sq = tb.trace(x_inv_v @ x_inv_v, keepdim=True)
        uv = tb.trace(prod, keepdim=True).pow_(2)
        denom = u_norm_sq.mul_(v_norm_sq).sub_(uv)
        assert all(denom > -self.wmin)
        denom.data.clamp_(min=self.wmin)

        return rt.div_(denom).flatten()

    def __str__(self):
        return "Manifold of {n}x{n} positive definite matrices".format(n=self.n)


class PairwiseSteinDivergence(Function):
    r"""The pairwise Stein divergence function. This can be used as a more
    efficient alternative to the canonical squared distance on the manifold.
    See _[2].

    It is implemented as a separate function to avoid computing the
    log-determinants (in the forward pass) and the inverses (in the backward
    pass) of each involved matrix multiple times.
    """

    @staticmethod
    def forward(ctx, x, chol):
        assert x.ndim == 3 and x.shape[-1] == x.shape[-2]
        n = x.shape[0]
        m = torch.triu_indices(n, n, 1, device=x.device)

        # compute :math:`\log \lvert (X + Y) / 2 \rvert` for all x, y
        xpy = 0.5 * (x[m[0]] + x[m[1]])
        xpy_l = chol(xpy)
        lhs = 2 * xpy_l.diagonal(dim1=-2, dim2=-1).abs().log().sum(1)

        # compute :math:`\log \lvert X \rvert` for all x
        x_l = chol(x)
        logdet_l = 2 * x_l.diagonal(dim1=-2, dim2=-1).abs().log().sum(1)
        rhs = -0.5 * (logdet_l[m[0]] + logdet_l[m[1]])

        ctx.save_for_backward(xpy_l, x_l, m)
        return lhs + rhs

    @staticmethod
    def backward(ctx, grad):
        xpy_l, x_l, m = ctx.saved_tensors
        n, d, _ = x_l.shape

        # TODO(ccruceru): Use `torch.cholesky_inverse` once the batched version
        # is implemented: https://github.com/pytorch/pytorch/issues/7500
        xpy_inv = torch.cholesky_solve(tb.eye_like(xpy_l), xpy_l)
        x_inv = torch.cholesky_solve(tb.eye_like(x_l), x_l)

        # Places tensors of shape (n,...) into symmetric (n,n,...)-tensors.
        def triu(x):
            shape = (n, n) + x.shape[1:]
            y = torch.zeros(shape, out=x.new(*shape))
            y[m[0], m[1], ...] = x
            return y + y.transpose(0, 1)

        grad_mat = 0.5 * (triu(xpy_inv) - x_inv)  # bcast: (n,n,d,d) - (_,n,d,d)
        grad_x = (triu(grad).view(n, n, 1, 1) * grad_mat).sum(0)

        return grad_x, None
