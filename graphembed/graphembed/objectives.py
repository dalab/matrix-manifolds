import abc
import random

import torch

from graphembed.utils import squareform1


class ObjectiveFunction:

    @abc.abstractmethod
    def __call__(self, gdists, mdists, *, epoch, alpha):
        pass


class QuotientLoss(ObjectiveFunction):

    def __init__(self, inc_l1=True, inc_l2=True):
        self.inc_l1 = inc_l1
        self.inc_l2 = inc_l2
        if not inc_l1 and not inc_l2:
            raise ValueError('At least one of the terms must be included.')

    def __call__(self, gdists, mdists, *, epoch, alpha):
        gdists = gdists * alpha
        loss = 0
        if self.inc_l1:
            loss += (mdists / gdists - 1.0).abs().sum()
        if self.inc_l2:
            eps = 1.0 / (epoch + 1)
            loss += (gdists / (mdists + eps) - 1.0).abs().sum()

        return loss

    def __str__(self):
        return 'quotient_loss'


class StressLoss(ObjectiveFunction):

    def __call__(self, gdists, mdists, *, epoch=None, alpha=None):
        return torch.pow(mdists - gdists, 2).sum()

    def __str__(self):
        return 'stress_loss'


class KLDiveregenceLoss(ObjectiveFunction):

    def __init__(self, inference_model, inclusive=True):
        from graphembed.inference import StochasticNeighbor, StochasticTrees
        if inference_model == 'sne':
            self.inference_fn = StochasticNeighbor()
        elif inference_model == 'sste':
            self.inference_fn = StochasticTrees()
        else:
            raise ValueError(
                    f'Inference model {inference_model} not supported.')
        self.inclusive = inclusive

    def __call__(self, gdists, mdists, *, epoch=None, alpha):
        theta_x = -alpha * gdists
        theta_z = -mdists
        if not self.inclusive:
            theta_x, theta_z = theta_z, theta_x

        return self._kl_div(theta_x, theta_z)

    def _kl_div(self, theta_x, theta_z):
        A_x, margs_x = self.inference_fn(theta_x, ret_margs=True)
        A_z, _ = self.inference_fn(theta_z, ret_margs=False)

        return A_z - A_x - margs_x @ (theta_z - theta_x)

    def __str__(self):
        return 'kl_loss'


class CurvatureRegularizer(ObjectiveFunction):
    # FIXME: Does not work in batched mode.

    def __init__(self, g, lambda_reg):
        self.n_nodes = g.number_of_nodes()
        self.adj_list = [list(g.neighbors(i)) for i in range(self.n_nodes)]
        self.lambda_reg = lambda_reg

    def __call__(self, gdists, mdists, *, epoch=None, alpha=None):
        dists = squareform1(mdists)
        ms, bs, cs = self._sample_nodes(device=dists.device)
        am = dists.gather(dim=1, index=ms)
        ab = dists.gather(dim=1, index=bs)
        ac = dists.gather(dim=1, index=cs)
        bc = dists[bs, cs]
        reg_terms = am + 0.25 * bc - 0.5 * (ab + ac)
        return -self.lambda_reg * reg_terms.abs().sum()

    def _sample_nodes(self, device):
        ms = torch.randperm(self.n_nodes, dtype=torch.long, device=device)
        bs = torch.empty(self.n_nodes, dtype=torch.long, device=device)
        cs = torch.empty(self.n_nodes, dtype=torch.long, device=device)
        for i in range(self.n_nodes):
            bs[i], cs[i] = random.choices(self.adj_list[ms[i]], k=2)
        return ms.unsqueeze_(-1), bs.unsqueeze_(-1), cs.unsqueeze_(-1)

    def __str__(self):
        return 'curvature_regularizer'


class PearsonRLoss(ObjectiveFunction):

    def __call__(self, x, y, **kwargs):
        xm = x - x.mean()
        ym = y - y.mean()
        return -xm @ ym / (xm.norm() * ym.norm())

    def __str__(self):
        return 'pearson_r_loss'


class Sum(ObjectiveFunction):

    def __init__(self, *fns):
        self.fns = fns

    def __call__(self, *args, **kwargs):
        return sum([fn(*args, **kwargs) for fn in self.fns])

    def __str__(self):
        return '__'.join([str(f) for f in self.fns])
