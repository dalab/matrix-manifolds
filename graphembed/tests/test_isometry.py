from itertools import product
import math
import pytest
import torch

from graphembed.manifolds import (SymmetricPositiveDefinite as SPD, Lorentz,
                                  Sphere)
from graphembed.manifolds.lorentz import ldot
from utils import assert_allclose

# The multiplication by :math:`sqrt{2}` corresponds to positioning the points on
# the hyperboloid with :math:`x0^2 - x1^2 - x2^2 = 2` which has constant
# sectional curvature :math:`-1/2`.
sspd2_hyp_radius_ = math.sqrt(2)


def sspd2_to_h2(x):
    r"""It maps SPD matrices of unit determinant to the Hyperbolic space of
    constant sectional curvature :math:`-1/2`.
    """
    a = x[..., 0, 0]
    b = x[..., 1, 1]
    c = x[..., 0, 1]

    x0 = (a + b).mul_(0.5)
    x1 = (a - b).mul_(0.5)
    x2 = c

    y = torch.stack([x0, x1, x2], axis=-1)
    y.div_(torch.sqrt(-ldot(y, y, keepdim=True)))
    return y


def sphere_to_hyperboloid(x):
    y = x.clone()
    y0 = y.narrow(-1, 0, 1)
    y0.mul_(-1)
    y.narrow(-1, 1, y.shape[-1] - 1).div_(y0)
    y0.reciprocal_()
    return y


def hyperboloid_to_sphere(x):
    y = x.clone()
    y0 = y.narrow(-1, 0, 1)
    y.narrow(-1, 1, y.shape[-1] - 1).div_(y0)
    y0.mul_(-1).reciprocal_()
    return y


@pytest.mark.parametrize('n', [100])
def test_sspd2_to_h2(seed, n):
    spd = SPD(2)
    lorentz = Lorentz(3)

    x = spd.rand(n, ir=1.0)
    x.div_(x.det().sqrt_().reshape(-1, 1, 1))  # unit determinant
    assert_allclose(x.det(), torch.ones(n), atol=1e-4)
    assert_allclose(x, spd.projx(x), atol=1e-4)

    y = sspd2_to_h2(x)
    hyp_dists = sspd2_hyp_radius_ * lorentz.pdist(y)
    assert_allclose(spd.pdist(x), hyp_dists, atol=1e-4)


@pytest.mark.parametrize('n,d', product([100], [0.5, 1, 2]))
def test_sspd2_to_h2_nonconst_factor(seed, n, d):
    spd = SPD(2)
    lorentz = Lorentz(3)

    x = spd.rand(n, ir=1.0)
    x.div_(x.det().sqrt_().reshape(-1, 1, 1))  # unit determinant
    x.mul_(d)  # d**2 determinant
    dets = torch.empty(n).fill_(d**2)
    assert_allclose(x.det(), dets, atol=1e-4)
    assert_allclose(x, spd.projx(x), atol=1e-4)

    y = sspd2_to_h2(x)
    hyp_dists = sspd2_hyp_radius_ * lorentz.pdist(y)

    # The determinant essentially does not affect the curvatures, they are all
    # isometric to the 2-dimensional hyperbolic space of -1/2 constant sectional
    # curvature.
    assert_allclose(spd.pdist(x), hyp_dists, atol=1e-4)


def h2_to_sspd2(x):
    shape = x.shape[:-1] + (1, 1)
    a = (x[..., 0] + x[..., 1]).view(shape)
    b = (x[..., 0] - x[..., 1]).view(shape)
    c = x[..., 2].view(shape)

    row1 = torch.cat([a, c], axis=-1)
    row2 = torch.cat([c, b], axis=-1)
    return torch.cat([row1, row2], axis=-2)


@pytest.mark.parametrize('n,d', product([100], [0.5, 1.0, 2.0]))
def test_h2_to_sspd2(seed, n, d):
    spd = SPD(2)
    lorentz = Lorentz(3)

    x = lorentz.rand(n, ir=1.0).mul_(d)
    y = h2_to_sspd2(x)
    assert_allclose(sspd2_to_h2(y), x / d, atol=1e-4)
    hyp_dists = sspd2_hyp_radius_ * lorentz.pdist(x / d)
    assert_allclose(spd.pdist(y), hyp_dists, atol=1e-4)


@pytest.mark.parametrize('n', [10])
def test_sph_hyp_mapping(seed, n):
    hyp = Lorentz(3)
    sph = Sphere(3)

    x = sph.rand(n, ir=1e-2, out=torch.empty(n, 3, dtype=torch.float64))
    sph_dists = sph.dist(sph.zero(n, out=x.new()), x)
    y = sphere_to_hyperboloid(x)
    hyp_dists = hyp.dist(hyp.zero(n, out=x.new()), y)
    assert_allclose(sph_dists, hyp_dists, atol=1e-4)


@pytest.mark.parametrize('n', [10])
def test_hyp_sph_mapping(seed, n):
    hyp = Lorentz(3)
    sph = Sphere(3)

    x = hyp.rand(n, ir=1e-2, out=torch.empty(n, 3, dtype=torch.float64))
    hyp_dists = hyp.dist(hyp.zero(n, out=x.new()), x)
    y = hyperboloid_to_sphere(x)
    sph_dists = sph.dist(sph.zero(n, out=x.new()), y)
    assert_allclose(sph_dists, hyp_dists, atol=1e-4)
