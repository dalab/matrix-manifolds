from itertools import product

import numpy as np
import pytest
import torch

from graphembed.manifolds import Sphere
from graphembed.modules import ManifoldParameter
from graphembed.optim import RiemannianSGD, RiemannianAdam
from utils import assert_allclose


@pytest.mark.parametrize('n,make_optim',
                         product([3, 4, 5], [
                                 lambda xs: RiemannianSGD([xs], lr=1e-1),
                                 lambda xs: RiemannianAdam([xs], lr=1e-1),
                         ]))
def test_dominant_eigenvector(seed, rand_sym, n, make_optim):
    torch.set_default_dtype(torch.float64)

    man = Sphere(n)
    A = rand_sym(1, n)[0]
    x = ManifoldParameter(man.rand(1)[0], manifold=man)
    optim = make_optim(x)

    for i in range(200):
        optim.zero_grad()
        loss = -torch.einsum('i,ij,j', x, A, x)
        loss.backward()
        optim.step()

        assert_allclose(1.0, x.detach().norm(), atol=1e-4)

    x_opt = x.squeeze().detach()
    eigv_exp = A.symeig(eigenvectors=True).eigenvectors[:, -1]
    quotient = (x_opt / eigv_exp).abs()
    assert_allclose(quotient, np.ones(n), atol=1e-4)

    eig = (A @ x_opt).norm()
    eig_exp = A.symeig().eigenvalues[-1]
    assert_allclose(eig, eig_exp, atol=1e-4)
