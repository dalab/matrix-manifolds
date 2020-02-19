import math
import numpy as np
import pytest
import scipy.linalg as sl
import torch

from graphembed.linalg import fast, torch_batch as tb
from utils import assert_allclose


@pytest.mark.parametrize('n,d', [(10, 2), (10, 3), (10, 5), (20, 10)])
def test_logm(seed, rand_spd, n, d):
    x = rand_spd(n, d)
    logx_ref = np.empty((n, d, d))
    for i in range(n):
        logx_ref[i] = sl.logm(x[i].cpu().numpy())
    assert_allclose(tb.spdlogm(x), logx_ref, atol=1e-4)


@pytest.mark.parametrize('n,d', [(10, 2), (10, 3), (10, 5)])
def test_expm(seed, rand_sym, n, d):
    x = rand_sym(n, d)
    expx_ref = np.empty((n, d, d))
    for i in range(n):
        expx_ref[i] = sl.expm(x[i].cpu().numpy())
    assert_allclose(tb.symexpm(x), expx_ref, atol=1e-4)


@pytest.mark.parametrize('n,d', [(10, 2), (10, 3), (10, 5), (20, 10)])
def test_sqrtm(seed, rand_spd, n, d):
    x = rand_spd(n, d)
    sqrtx_ref = np.empty((n, d, d))
    for i in range(n):
        sqrtx_ref[i] = sl.sqrtm(x[i].cpu().numpy())
    assert_allclose(tb.spdsqrtm(x), sqrtx_ref, atol=1e-4)


@pytest.mark.parametrize('n,d', [(10, 2), (10, 3), (10, 5), (20, 10)])
def test_symsqtrace(seed, rand_sym, n, d):
    x = rand_sym(n, d)
    traces_np = np.trace((x @ x).cpu().numpy(), axis1=1, axis2=2)
    assert_allclose(tb.symsqtrace(x), traces_np, atol=1e-4)


def test_symeig2x2_eye():
    eyes = torch.eye(2).expand(10, -1, -1)
    ones = torch.ones(2).expand(10, -1)
    eigs = fast.symeig2x2(eyes)
    assert_allclose(eigs, ones, atol=1e-4)


def test_symeig3x3_eye():
    eyes = torch.eye(3).expand(10, -1, -1)
    ones = torch.ones(3).expand(10, -1)
    eigs = fast.symeig3x3(eyes)
    assert_allclose(eigs, ones, atol=1e-4)


@pytest.mark.parametrize('n', range(10, 20))
def test_fast_symeig2x2(seed, rand_sym, n):
    x = rand_sym(n, 2)
    assert_allclose(fast.symeig2x2(x), torch.symeig(x).eigenvalues, atol=1e-4)


@pytest.mark.parametrize('n', range(10, 20))
def test_fast_symeig3x3(seed, rand_sym, n):
    x = rand_sym(n, 3)
    assert_allclose(fast.symeig3x3(x), torch.symeig(x).eigenvalues, atol=1e-4)


# TODO(ccruceru): Find a solution to the non-symmetric gradients that does not
# involve registering a backward hook (which caused performance issues when
# performed inside the function).
# See also https://github.com/pytorch/pytorch/pull/23018


@pytest.mark.parametrize('n,d,eig', [
        (100, 2, fast.symeig2x2),
        (100, 3, fast.symeig3x3),
])
def test_eig_gradients(seed, rand_sym, n, d, eig):
    x1 = rand_sym(n, d).requires_grad_()
    s1 = eig(x1).pow(2).sum()
    s1.backward()
    x2 = x1.detach().clone().requires_grad_()
    s2 = torch.symeig(x2, eigenvectors=True).eigenvalues.pow(2).sum()
    s2.backward()
    assert_allclose(tb.sym(x1.grad), tb.sym(x2.grad), atol=1e-4)


@pytest.mark.parametrize('n', range(10, 20))
def test_cholesky2x2(seed, rand_spd, n):
    x = rand_spd(n, 2)
    assert_allclose(x.cholesky(), fast.cholesky2x2(x), atol=1e-4)


@pytest.mark.parametrize('n', range(10, 20))
def test_cholesky2x2_grad(seed, rand_spd, n):
    x1 = rand_spd(n, 2).requires_grad_()
    s1 = fast.cholesky2x2(x1).pow(2).sum()
    s1.backward()
    x2 = x1.detach().clone().requires_grad_()
    s2 = x2.cholesky().pow(2).sum()
    s2.backward()
    assert_allclose(tb.sym(x1.grad), x2.grad, atol=1e-4)


@pytest.mark.parametrize('n', range(10, 20))
def test_invcholesky2x2(seed, rand_spd, n):
    x = rand_spd(n, 2)
    assert_allclose(
            x.cholesky().inverse(), fast.invcholesky2x2(x)[0], atol=1e-4)


@pytest.mark.parametrize('n', range(10, 20))
def test_invcholesky2x2_grad(seed, rand_spd, n):
    x1 = rand_spd(n, 2).requires_grad_()
    s1 = fast.invcholesky2x2(x1)[0].pow(2).sum()
    s1.backward()
    x2 = x1.detach().clone().requires_grad_()
    s2 = x2.cholesky().inverse().pow(2).sum()
    s2.backward()
    assert_allclose(tb.sym(x1.grad), x2.grad, atol=1e-4)


@pytest.mark.parametrize('n', [500])
def test_singular_values_2x2(seed, n):
    x = torch.rand(n, 2, 2, dtype=torch.float64)
    assert_allclose(
            fast.singular_values_2x2(x),
            tb.svd(x, compute_uv=False).S,
            atol=1e-4)
