from itertools import product
import math

import pytest
import scipy.linalg as sl
import torch

from graphembed.manifolds import Grassmann, SpecialOrthogonalGroup, Stiefel
from utils import assert_allclose


@pytest.mark.parametrize('n,p', product(range(5, 10), [2, 3, 4]))
def test_log_gras(n, p):
    gras = Grassmann(n, p)
    x = gras.rand_uniform(100, out=torch.empty(10, n, p, dtype=torch.float64))
    y = gras.rand_uniform(100, out=x.new(10, n, p))
    assert_allclose(gras.dist(x, y), gras.norm(x, gras.log(x, y)), atol=1e-4)


@pytest.mark.parametrize('n', [3, 4, 5])
def test_log_so(seed, n):
    so = SpecialOrthogonalGroup(n)
    x = so.rand_uniform(100, out=torch.empty(100, n, n, dtype=torch.float64))
    y = so.rand_uniform(100, out=x.new(100, n, n))
    assert_allclose(so.dist(x, y), so.norm(x, so.log(x, y)), atol=1e-4)


@pytest.mark.parametrize('n,p', product(range(5, 10), [2, 3, 4]))
def test_gradient_gras(seed, n, p):
    gras = Grassmann(n, p)
    x, y = gras.rand_uniform(2, out=torch.empty(2, n, p, dtype=torch.float64))
    x.requires_grad_()
    dist = 0.5 * gras.dist(x, y, squared=True)
    grad_e = torch.autograd.grad(dist, x)[0]
    grad = gras.egrad2rgrad(x, grad_e)
    assert_allclose(grad.detach(), -gras.log(x.detach(), y), atol=1e-4)


def compute_angle(man, x, y, z, eps=1e-4):
    u = man.log(x, y)
    v = man.log(x, z)
    uv_inner = man.inner(x, u, v, keepdim=True)
    u_norm = man.norm(x, u, keepdim=True)
    v_norm = man.norm(x, v, keepdim=True)

    cos_theta = uv_inner / u_norm / v_norm
    cos_theta.clamp_(-1.0 + eps, 1.0 - eps)

    return cos_theta.acos()


@pytest.mark.parametrize('n,p', product(range(5, 10), [2, 3, 4]))
def test_sum_of_angles_grassmann(seed, n, p):
    float64 = torch.empty(100, n, p, dtype=torch.float64)
    gras = Grassmann(n, p)
    xs = gras.rand_uniform(100, out=torch.empty_like(float64))
    ys = gras.rand_uniform(100, out=torch.empty_like(float64))
    zs = gras.rand_uniform(100, out=torch.empty_like(float64))

    theta1 = compute_angle(gras, xs, ys, zs)
    theta2 = compute_angle(gras, ys, xs, zs)
    theta3 = compute_angle(gras, zs, xs, ys)
    sum_angles = theta1 + theta2 + theta3
    assert torch.all(sum_angles >= math.pi)


def logm_dist(xs, ys):
    return torch.as_tensor([sl.norm(sl.logm(x.T @ y)) for x, y in zip(xs, ys)])


@pytest.mark.parametrize('n', [3, 4, 5])
def test_ortho_dist_same_as_logm(seed, n):
    torch.set_default_dtype(torch.float64)

    ortho = SpecialOrthogonalGroup(n)
    xs = ortho.rand_uniform(100)
    ys = ortho.rand_uniform(100)

    assert_allclose(logm_dist(xs, ys), ortho.dist(xs, ys), atol=1e-4)
