r"""Defines functions that compute eigenvalues for small matrices.

Notes
-----
We do not use both parts of the symmetric matrices where possible. This yields
upper-diagonal gradients. Ideally we should write the backward functions
ourselves and symmetrize them. See also _[3], _[4].

Notice that just taking the sum of the two and dividing by two will only
amplify the skew between them.

References
----------
.. [1] http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/
.. [2] https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3%C3%973_matrices
.. [3] https://github.com/pytorch/pytorch/issues/18825#issuecomment-480955226
.. [4] https://github.com/pytorch/pytorch/issues/22807
"""
import math
import torch

import graphembed.linalg.torch_batch as tb


def det2x2(X, keepdim=False):
    assert X.shape[-2:] == (2, 2)
    det = X[..., 0, 0] * X[..., 1, 1] - X[..., 0, 1] * X[..., 1, 0]
    return det.view(-1, 1, 1) if keepdim else det


def det3x3(X, keepdim=False):
    assert X.shape[-2:] == (3, 3)
    m1 = det2x2(X[..., 1:, 1:])
    m2 = det2x2(torch.stack([X[..., 1:, 0], X[..., 1:, 2]], axis=-1))
    m3 = det2x2(X[..., 1:, :2])
    det = X[..., 0, 0] * m1 - X[..., 0, 1] * m2 + X[..., 0, 2] * m3
    return det.view(-1, 1, 1) if keepdim else det


def symdet3x3(X, keepdim=False):
    assert X.shape[-2:] == (3, 3)
    x00 = X[..., 0, 0]
    x01 = X[..., 0, 1]
    x02 = X[..., 0, 2]
    x11 = X[..., 1, 1]
    x12 = X[..., 1, 2]
    x22 = X[..., 2, 2]
    det = x00 * x11 * x22 + 2 * x01 * x02 * x12 - \
            x11 * x02**2 - x00 * x12**2 - x22 * x01**2
    return det.view(-1, 1, 1) if keepdim else det


def symeig2x2(X, eps=1e-8):
    r"""Computes the eigenvalues of a symmetric 2x2 matrix. See _[1]."""
    # NOTE: We duplicate determinant and trace computation logic here to make it
    # as self-contained as possible.
    assert X.shape[-2:] == (2, 2)
    a = X[..., 0, 0]
    b = X[..., 1, 1]
    c = X[..., 0, 1]
    det = a * b - c**2
    half_trace = 0.5 * (a + b)
    delta = half_trace**2 - det
    delta.data.clamp_(min=eps)
    rhs_term = delta.sqrt()

    eig1 = half_trace - rhs_term
    eig2 = half_trace + rhs_term

    return torch.stack([eig1, eig2], axis=-1)


# TODO(ccruceru): Check if using Taylor series expansion for
# :math:`cos(acos/3 + 2k pi/ 3)` can give accurate, fast, and stable results.
def symeig3x3(X, eps=1e-8):
    r"""Computes the eigenvalues of a symmetric 3x3 matrix. See _[2]."""
    assert X.shape[-2:] == (3, 3)

    q = tb.trace(X, keepdim=True) / 3
    Y = X - q * torch.eye(3, out=X.new(3, 3)).expand_as(X)
    p = torch.sqrt(tb.symsqtrace(Y, keepdim=True) / 6)
    p.data.clamp_(min=eps)
    r = symdet3x3(Y, keepdim=True) / (2 * p.pow(3) + eps)
    r.data.clamp_(min=-1 + eps, max=1 - eps)
    phi = torch.acos(r) / 3

    eig1 = q + 2 * p * torch.cos(phi)
    eig2 = q + 2 * p * torch.cos(phi + 2 * math.pi / 3)
    eig3 = 3 * q - eig1 - eig2

    return torch.stack([eig2, eig3, eig1], axis=-1).squeeze()


def cholesky2x2(X, eps=1e-8):
    assert X.shape[-2:] == (2, 2)
    shape = X.shape[:-2] + (1, 1)
    x00 = X[..., 0, 0].view(shape)
    x00.data.clamp_(min=eps)
    x11 = X[..., 1, 1].view(shape)
    x01 = X[..., 0, 1].view(shape)
    a = x00.sqrt()
    b = x01 / a
    c = (x11 - b**2 + eps).sqrt()

    row1 = torch.cat([a, torch.zeros_like(a)], axis=a.ndim - 1)
    row2 = torch.cat([b, c], axis=a.ndim - 1)
    return torch.cat([row1, row2], axis=a.ndim - 2)


def invcholesky2x2(X, ret_chol=False, eps=1e-8):
    assert X.shape[-2:] == (2, 2)
    shape = X.shape[:-2] + (1, 1)
    x00 = X[..., 0, 0].view(shape)
    x00.data.clamp_(min=eps)
    x11 = X[..., 1, 1].view(shape)
    x01 = X[..., 0, 1].view(shape)
    a = x00.sqrt()
    b = x01 / a
    c = (x11 - b**2 + eps).sqrt()
    det = a * c
    det.data.clamp_(min=eps)

    zeros = torch.zeros_like(c)
    row1 = torch.cat([c, zeros], axis=a.ndim - 1)
    row2 = torch.cat([-b, a], axis=a.ndim - 1)
    l_inv = torch.cat([row1, row2], axis=a.ndim - 2) / det
    if not ret_chol:
        return l_inv, None

    row1 = torch.cat([a, zeros], axis=a.ndim - 1)
    row2 = torch.cat([b, c], axis=a.ndim - 1)
    l = torch.cat([row1, row2], axis=a.ndim - 2)

    return l_inv, l


# Reference: https://www.lucidar.me/en/mathematics/singular-value-decomposition-of-a-2x2-matrix/
def singular_values_2x2(x, eps=1e-8):
    r"""Returns the singular values of 2x2 matrices."""
    a = x[..., 0, 0]
    b = x[..., 0, 1]
    c = x[..., 1, 0]
    d = x[..., 1, 1]

    S1 = a**2 + b**2 + c**2 + d**2
    S2 = (a**2 + b**2 - c**2 - d**2)**2 + 4 * (a * c + b * d)**2
    S2.data.clamp_(min=eps)
    S2 = torch.sqrt(S2)

    s1 = 0.5 * (S1 + S2)
    s1.data.clamp_(min=eps)
    s2 = 0.5 * (S1 - S2)
    s2.data.clamp_(min=eps)

    ss = x.new(*x.shape[:-2], 2)
    ss[..., 0] = torch.sqrt(s1)
    ss[..., 1] = torch.sqrt(s2)

    return ss


# TODO(ccruceru): This is just experimental and not tested properly.
def spdeig(x, iters):
    for _ in range(iters):
        u = tb.cholesky(x, upper=True)
        x = u @ tb.transpose(u)
    return x.diagonal(dim1=-2, dim2=-1)
