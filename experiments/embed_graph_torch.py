import argparse
import math
import os
import sys
import time

import geoopt
from geoopt.manifolds.spd.multi import *
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import gradcheck, Function
from torch.utils.tensorboard import SummaryWriter

from embed_graph import load_pdists, config_parser as base_parser, pdists_vec_to_sym
from metrics import average_distortion, mean_average_precision
from util import Timer

# TODO(ccruceru): Split this into multiple modules with better organized
# options.


def config_parser(parser=None):
    if not parser:
        parser = argparse.ArgumentParser(
                description='Parse graph embedding with PyTorch.')
    parser.add_argument(
            '--stein_div',
            action='store_true',
            help='Whether the Stein divergence should be used as a (squared) '
            'distance function instead')
    parser.add_argument(
            '--min_max_scale',
            action='store_true',
            help='Whether min-max scaling should be used instead of learning '
            'the scaling factor.')
    parser.add_argument(
            '--detect_anomalies',
            action='store_true',
            help='Whether anomaly detection in PyTorch should be enabled.')
    parser.add_argument(
            '--verbose',
            action='store_true',
            help='Whether we should run in verbose mode.')
    return parser


class PairwiseSteinDivergence(Function):

    @staticmethod
    def forward(ctx, x):
        n = x.shape[0]
        mask = torch.triu_indices(n, n, 1)

        xpy = 0.5 * (x[mask[0]] + x[mask[1]])
        xpy_l = torch.cholesky(xpy)
        lhs = 2 * xpy_l.diagonal(dim1=1, dim2=2).log().sum(1)

        x_l = torch.cholesky(x)
        logdet_l = 2 * x_l.diagonal(dim1=1, dim2=2).log().sum(1)
        rhs = -0.5 * (logdet_l[mask[0]] + logdet_l[mask[1]])

        ctx.save_for_backward(xpy_l, x_l, mask)
        return lhs + rhs

    @staticmethod
    def backward(ctx, grad):
        xpy_l, x_l, mask = ctx.saved_tensors
        n, d, _ = x_l.shape

        def triu(x):
            shape = (n, n) + x.shape[1:]
            y = torch.zeros(shape, out=x.new(*shape))
            y[mask[0], mask[1], ...] = x
            return y + y.transpose(0, 1)

        def eye_like(x):
            d = x.shape[-1]
            return torch.eye(d, out=x.new(d, d)).expand_as(x)

        # TODO(ccruceru): Use `torch.cholesky_inverse` once the batched version
        # is implemented: https://github.com/pytorch/pytorch/issues/7500
        xpy_inv = torch.cholesky_solve(eye_like(xpy_l), xpy_l)
        x_inv = torch.cholesky_solve(eye_like(x_l), x_l)

        grad_mat = 0.5 * (triu(xpy_inv) - x_inv)  # (n,n,d,d) - (_,n,d,d)
        # it's important that this multiplication zeros out the diagonal!
        grad_x = (triu(grad).view(n, n, 1, 1) * grad_mat).sum(0)

        return grad_x


manifold_sq_pdists_stein = PairwiseSteinDivergence.apply
gradcheck(
        manifold_sq_pdists_stein,
        # NOTE: The check doesn't work on matrices with non-zero off-diagonal
        # because of how torch.cholesky works. See also
        # https://github.com/pytorch/pytorch/issues/18825
        torch.as_tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.0], [0.0, 0.5]]],
                        dtype=torch.float64).requires_grad_(),
        eps=1e-6,
        atol=1e-4)


def sqrtm2x2(X, eps=1e-6):
    r"""Computes the square root of a 2x2 matrix. See [1].

    References
    ----------
    [1]: https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
    """
    n = X.shape[0]
    a = X[:, 0, 0].reshape(n, 1, 1)
    b = X[:, 1, 1].reshape(n, 1, 1)
    c = X[:, 0, 1].reshape(n, 1, 1)
    dets = a * b - c**2
    sqrt_dets = (dets + eps).sqrt()
    new_a = a + sqrt_dets
    new_b = b + sqrt_dets
    traces = a + b
    ts = (traces + 2 * sqrt_dets + eps).sqrt()
    S = torch.cat((torch.cat((new_a, c), 2), torch.cat((c, new_b), 2)), 1) / ts

    return S


def eig2x2(X, eps=1e-6):
    r"""Computes the eigenvalues of a symmetric 2x2 matrix.  See [1].

    References
    ----------
    [1]: http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/

    """
    # TODO(ccruceru): Do we gain something from checking the diagonal case?
    a = X[:, [0], [0]]
    b = X[:, [1], [1]]
    c = X[:, [0], [1]]
    dets = a * b - c**2
    half_traces = 0.5 * (a + b)
    rhs_term = (half_traces**2 - dets + eps).sqrt()

    return torch.cat([half_traces + rhs_term, half_traces - rhs_term], axis=1)


def eig3x3(X, eps=1e-6):
    r"""Computes the eigenvalues of a symmetric 3x3 matrix.  See [2].

    [2]: https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3%C3%973_matrices
    """
    n, d, _ = X.shape

    q = multitrace(A, keepdim=True) / 3
    Y = X - q * torch.eye(d, out=X.new(d, d)).expand(n, -1, -1)
    p = torch.sqrt(multitrace(Y.matrix_power(2)) / 6 + eps)
    r = torch.det(Y) / (2 * p.pow(3))
    r.data.clamp_(min=-1 + eps, max=1 + eps)
    phi = torch.acos(r) / 3

    eig1 = q + 2 * p * torch.cos(phi)
    eig2 = q + 2 * p * torch.cos(phi + 2 * math.pi / 3)
    eig3 = 3 * q - eig1 - eig2

    return torch.cat([eig1, eig2, eig3], axis=1)


def eig(X):
    d = X.shape[-1]
    if d == 2:
        return eig2x2(X)
    # TODO(ccruceru): https://github.com/pytorch/pytorch/pull/22909.
    # elif d == 3:
    #     return eig3x3(X)

    # eigenvectors for backward
    return torch.symeig(X, eigenvectors=True).eigenvalues


def manifold_sq_pdists(X):
    # We implement it manually rather than calling spd.dist() to avoid computing
    # the cholesky decomposition and its inverse several times.
    n, d, _ = X.shape
    mask = torch.triu_indices(n, n, 1)
    eye = torch.eye(d, out=X.new(d, d))

    # First, compute the matrix inside log for all pairs of points.
    # TODO(ccruceru): Check if manually computing the square root is beneficial.
    L = torch.cholesky(X)
    L_inv = torch.triangular_solve(eye, L, upper=False).solution
    A = multiAXAt(L_inv[mask[0]], X[mask[1]])

    # Then, compute the sum of squared logarithms of eigenvalues.
    # NOTE: This is equivalent to computing the matrix logarithm and then taking
    # its squared Frobenius norm:
    #       A_log = multilog(A)
    #       pdists = A_log.pow(2).sum((1, 2))
    W = eig(A)
    W.data.clamp_(min=X.manifold.wmin, max=X.manifold.wmax)
    pdists = W.log().pow(2).sum(-1)

    return pdists


def sample_init_points(n, d):
    r"""Function used to initialize the optimization by sampling a matrix of
    uniform numbers between 0 and 1 and then adjusting their eigenvalues to be
    positive by taking the absolute value and adding 1.
    """
    X = torch.rand(n, d, d)
    X = multisym(X)
    X = multisymapply(X, lambda W: W.abs() + 1)

    return X


def min_max_scale(t, min, max):
    t_std = (t - t.min()) / (t.max() - t.min())
    return min + t_std * (max - min)


def main():
    args = config_parser(base_parser()).parse_args()
    if args.detect_anomalies:
        torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)

    # load graph pdists
    g_pdists, g = load_pdists(args)
    g_pdists = torch.Tensor(g_pdists)
    n = g_pdists.shape[0]
    d = args.manifold_dim

    # masks used to get the upper diagonal part of (i) pairwise distances
    # matrices, and (ii) the embedding matrices themselves
    mask = torch.triu_indices(n, n, 1)
    e_mask = torch.triu_indices(d, d)

    # we are only using the upper diagonal part
    g_pdists = g_pdists[mask[0], mask[1]]
    # scale if needed
    if args.min_max_scale:
        g_pdists = min_max_scale(g_pdists, 1, 10)
    g_sq_pdists = g_pdists.pow(2)
    # keep a numpy copy for computing metrics
    g_pdists_np = g_pdists.cpu().numpy()

    # embedding initializations
    X_init = sample_init_points(n, d)
    # put them on GPU if available
    if torch.cuda.is_available():
        with Timer('copying data to GPU'):
            X_init = X_init.pin_memory().cuda()
            g_sq_pdists = g_sq_pdists.pin_memory().cuda()

    # the embedding paramters we optimize for
    spd = geoopt.SymmetricPositiveDefinite(wmin=1e-8, wmax=1e8)
    X = geoopt.ManifoldParameter(X_init, manifold=spd)

    # the distance function
    dist_fn = manifold_sq_pdists_stein if args.stein_div else manifold_sq_pdists

    # setup the optimizer
    # TODO(ccruceru): Investigate the momentum issues.
    optim = geoopt.optim.RiemannianSGD([X], lr=0.5)
    lr_scheduler = ReduceLROnPlateau(
            optim, patience=20, factor=0.5, min_lr=1e-8, verbose=args.verbose)

    # training settings
    writer = SummaryWriter()
    n_epochs = 1500
    save_every_epochs = 10

    def criterion(epoch):
        mdists = dist_fn(X)
        l1 = (mdists / g_sq_pdists - 1.0).abs().sum()
        eps = 1.0 / (epoch + 1)
        l2 = (g_sq_pdists / (mdists + eps) - 1.0).abs().sum()
        return (l1 + l2) / n

    def run_epoch(epoch):
        optim.zero_grad()
        loss = criterion(epoch)
        loss.backward()
        optim.step()
        lr_scheduler.step(loss)

        return loss

    def compute_metrics():
        with torch.no_grad():
            man_pdists_np = dist_fn(X).sqrt().cpu().numpy()
        ad = average_distortion(g_pdists_np, man_pdists_np)
        if g is None:
            return ad, None

        # TODO(ccruceru): Make sure this is correct. Try to reproduce the
        # result from the ref. paper on 10D Euclidean manifold.
        man_pdists_sym = pdists_vec_to_sym(man_pdists_np, n)
        mean_ap = mean_average_precision(g, man_pdists_sym)
        return ad, mean_ap

    with Timer('training'):
        for epoch in range(n_epochs):
            # early break if we reached the minimum learning rate
            if optim.param_groups[0]['lr'] <= 2 * lr_scheduler.min_lrs[0]:
                break
            start = time.time()
            loss = run_epoch(epoch)
            stop = time.time()
            if epoch % save_every_epochs != 0:
                continue

            # show it
            if args.verbose:
                print('epoch {:5}, loss {:.10f}, time {}'.format(
                        epoch, loss.item(), stop - start))

            # monitoring
            with torch.no_grad():
                logw = eig(X).log()
                ks = logw.max(1).values - logw.min(1).values
                ad, mean_ap = compute_metrics()
            writer.add_scalar('loss', loss, epoch)
            writer.add_histogram('log_lambda', logw.flatten(), epoch)
            writer.add_histogram('log_k_X', ks, epoch)
            writer.add_embedding(X[:, e_mask[0], e_mask[1]], global_step=epoch)
            # metrics
            writer.add_scalar('avg_distortion', ad, epoch)
            if mean_ap:
                writer.add_scalar('mAP', mean_ap, epoch)

    torch.save(X, os.path.join(writer.get_logdir(), 'x_opt.pt'))

    # final metrics
    ad, mean_ap = compute_metrics()
    print('Average distortion: ', ad)
    if mean_ap:
        print('mAP: ', mean_ap)


if __name__ == '__main__':
    sys.exit(main())
