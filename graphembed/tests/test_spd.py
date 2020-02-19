import numpy as np
import pytest
import torch

from graphembed.manifolds import SymmetricPositiveDefinite as SPD
from utils import assert_allclose


def test_dim():
    assert 3 == SPD(2).dim
    assert 6 == SPD(3).dim
    assert 10 == SPD(4).dim


@pytest.mark.parametrize('d', range(2, 10))
def test_unit_distance(d, seed):
    spd = SPD(d)
    u_vec = torch.randn(spd.dim)
    u = SPD.from_vec(u_vec / u_vec.norm())
    x = torch.eye(d)
    assert_allclose(1.0, spd.norm(x, u), atol=1e-4)
    y = spd.exp(x, u)
    assert_allclose(1.0, spd.dist(x, y), atol=1e-4)


@pytest.mark.parametrize('d', range(2, 10))
def test_exp_log(seed, rand_spd, rand_sym, d):
    spd = SPD(d)
    x = rand_spd(10, d)
    u = rand_sym(10, d)
    y = spd.exp(x, u)
    assert_allclose(u, spd.log(x, y), atol=1e-4)
    assert_allclose(spd.norm(x, u), spd.dist(x, y), atol=1e-4)


@pytest.mark.parametrize('d,n', [
        (2, 1000 if not torch.cuda.is_available() else 10000),
        (3, 1000 if not torch.cuda.is_available() else 10000),
        (4, 1000 if not torch.cuda.is_available() else 50),
])
def test_no_nan_dists(seed, rand_spd, d, n):
    spd = SPD(d)
    x = rand_spd(n, d)
    assert not torch.isnan(spd.pdist(x)).any()


@pytest.mark.parametrize('d', range(1, 10))
def test_distance_formulas(seed, rand_spd, d):
    spd = SPD(d)
    x, y = rand_spd(2, d)
    ref_dist = spd.dist(x, y)

    # compute :math:`Y^{-1} X` and take its eigenvalues (we have to use
    # `torch.eig` for this as the resulting matrix might not be symmetric)
    d1 = torch.solve(y, x)[0].eig()[0][:, 0].log_().pow_(2).sum().sqrt_()
    assert_allclose(ref_dist, d1, atol=1e-4)

    d2 = torch.solve(x, y)[0].eig()[0][:, 0].log_().pow_(2).sum().sqrt_()
    assert_allclose(ref_dist, d2, atol=1e-4)


@pytest.mark.parametrize('d', range(2, 10))
def test_inner_norm(seed, d):
    spd = SPD(d)
    xs = spd.rand(100, ir=1.0, out=torch.empty(100, d, d, dtype=torch.float64))
    us = spd.randvec(xs)
    assert_allclose(spd.inner(xs, us, us)**0.5, spd.norm(xs, us), atol=1e-4)


@pytest.mark.parametrize('d', range(2, 10))
def test_gradient(seed, d):
    spd = SPD(d)
    x, y = spd.rand(2, ir=1.0, out=torch.empty(2, d, d, dtype=torch.float64))
    x.requires_grad_()
    dist = 0.5 * spd.dist(x, y, squared=True)
    grad_e = torch.autograd.grad(dist, x)[0]
    grad = spd.egrad2rgrad(x, grad_e)
    assert_allclose(grad.detach(), -spd.log(x.detach(), y), atol=1e-4)


@pytest.mark.parametrize('d', range(2, 10))
def test_stein_pdiv(seed, d):
    spd = SPD(2)
    xs = spd.rand(10, ir=1.0, out=torch.empty(10, d, d, dtype=torch.float64))
    pdivs = spd.stein_pdiv(xs)
    m = torch.triu_indices(10, 10, 1)
    ref_pdivs = spd.stein_div(xs[m[0]], xs[m[1]])
    assert_allclose(ref_pdivs, pdivs, atol=1e-4)
