from collections import namedtuple

import numpy as np
import torch
from torch.autograd import Function


def eye_like(x):
    r"""Returns an identity matrix expanded to the same shape as `x`.

    .. warn:
        It should be used in read-only mode as it uses
        `torch.Tensor.~expand_as`.
    """
    assert x.ndim >= 2
    n, m = x.shape[-2:]
    return torch.eye(n, m=m, out=x.new(n, m)).expand_as(x)


def transpose(x):
    r"""Returns the transpose for each matrix in a batch."""
    return x.transpose(dim0=-2, dim1=-1)


def sym(x):
    r"""Returns the symmetrized version of each matrix in a batch."""
    return 0.5 * (x + transpose(x))


def axat(a, x):
    r"""Returns the product :math:`A X A^\top` for each pair of matrices in a
    batch. This should give the same result as :py:`A @ X @ A.T`.
    """
    return torch.einsum("...ij,...jk,...lk->...il", a, x, a)


def trace(x, keepdim=False):
    r"""Returns the trace for each matrix in a batch."""
    traces = x.diagonal(dim1=-2, dim2=-1).sum(-1)
    return traces.view(-1, 1, 1) if keepdim else traces


def cholesky(x, *args, **kwargs):
    r"""A wrapper over `torch.cholesky` that deals with the CUDA maximum batch
    size limitation.
    """
    if x.device.type == 'cpu':
        return torch.cholesky(x, *args, **kwargs)

    shape = x.shape
    batch_size = np.prod(shape[:-2])
    cuda_batch_size = (1 << 18) - 4

    if batch_size <= cuda_batch_size:
        return torch.cholesky(x, *args, **kwargs)

    x = x.view(-1, shape[-1], shape[-1])
    acc = torch.empty_like(x)
    for i in range(0, batch_size, cuda_batch_size):
        indices = torch.arange(
                i,
                min(batch_size, i + cuda_batch_size),
                device=x.device,
                dtype=torch.int64)
        acc[indices] = torch.cholesky(x[indices], *args, **kwargs)

    return acc.view(shape)


def symsqtrace(x, keepdim=False):
    r"""Returns the trace of the squared of a symmetric matrix."""
    # TODO(ccruceru): Consider if symmetrizing the gradient yields better
    # numerical stability (due to skew between the upper/lower diagonal parts).
    return x.pow(2).sum((-2, -1), keepdim=keepdim)


def xty(x, y):
    r"""Computes :math:`X^T Y`."""
    return torch.einsum('...ji,...jk->...ik', x, y)


def xyt(x, y):
    r"""Computes :math:`X Y^T`."""
    return torch.einsum('...ij,...kj->...ik', x, y)


def mvmt(u, w, v):
    r""""The multiplication `u @ diag(w) @ v.T`. The name stands for
    matrix/vector/matrix-transposed.
    """
    return torch.einsum("...ij,...j,...kj->...ik", u, w, v)


def cpuoffload_op_(op, x, make_ret, cpu_offload=True, *args, **kwargs):
    r"""Template function for linear algebra wrapper functions that are slower
    on GPU than on CPU which offloads the computation to the latter.
    """
    if not cpu_offload:
        return op(x, *args, **kwargs)

    device = x.device
    rets = op(x.cpu(), *args, **kwargs)
    return make_ret(*(r.to(device) for r in rets))


SvdOut = namedtuple('tb_svd', ['U', 'S', 'V'])


def svd(x, cpu_offload=True, *args, **kwargs):
    r"""SVD wrapper for CPU offloading."""
    return cpuoffload_op_(
            torch.svd, x, SvdOut, cpu_offload=cpu_offload, *args, **kwargs)


QrOut = namedtuple('tb_qr', ['Q', 'R'])


def qr(x, cpu_offload=True, *args, **kwargs):
    r"""QR wrapper for CPU offloading."""
    return cpuoffload_op_(
            torch.qr, x, QrOut, cpu_offload=cpu_offload, *args, **kwargs)


SymeigOut = namedtuple('tb_symeig', ['eigenvalues', 'eigenvectors'])


def symeig(x, cpu_offload=True, *args, **kwargs):
    r"""Symeig wrapper for CPU offloading."""
    return cpuoffload_op_(
            torch.symeig,
            x,
            SymeigOut,
            cpu_offload=cpu_offload,
            *args,
            **kwargs)


def hgie(w, v):
    r"""The inverse of :py:`torch.symeig` for batched matrices. The name "hgie"
    is simply the string "eigh" reversed.
    """
    return mvmt(v, w, v)


def symapply(x, f, *, wmin=None, wmax=None):
    r"""Template function acting on stacked symmetric matrices that applies a
    given analytic function on them via eigenvalue decomposition.
    """
    w, v = symeig(x, eigenvectors=True)
    if wmin is not None or wmax is not None:
        w.data.clamp_(min=wmin, max=wmax)

    return hgie(f(w), v)


def spdlogm(x, *, wmin=None, wmax=None):
    r"""Returns the matrix logarithm for each symmetric positive definite matrix
    in a batch.
    """
    return symapply(x, torch.log, wmin=wmin, wmax=wmax)


def symexpm(x, *, wmin=None, wmax=None):
    r"""Returns the matrix exponential for each symmetric matrix in a batch."""
    return symapply(x, torch.exp, wmin=wmin, wmax=wmax)


def spdsqrtm(x, *, wmin=None, wmax=None):
    r"""Returns the matrix square root for each symmetric matrix in a batch."""
    return symapply(x, torch.sqrt, wmin=wmin, wmax=wmax)


class PLogDet(Function):
    r"""Log-determinant function for positive definite matrices."""

    @staticmethod
    def forward(ctx, x, chol, keepdim):
        l = chol(x)
        ctx.save_for_backward(l)
        return 2 * l.diagonal(dim1=-2, dim2=-1).abs().log().sum(-1, keepdim)

    @staticmethod
    def backward(ctx, g):
        l, = ctx.saved_tensors
        n = l.shape[-1]
        # TODO: Use cholesky_inverse once pytorch/pytorch/issues/7500 is solved.
        grad_x = g.view(*l.shape[:-2], 1, 1) * torch.cholesky_solve(
                torch.eye(n, out=l.new(n, n)), l)

        return grad_x, None, None


def plogdet(x, chol=cholesky, keepdim=False):
    r"""Forwards the call to the autograd Function above but with support for
    positional arguments.
    """
    return PLogDet.apply(x, chol, keepdim)
