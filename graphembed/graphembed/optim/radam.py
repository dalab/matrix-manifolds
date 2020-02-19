import logging
import torch

from graphembed.manifolds import Euclidean
from graphembed.modules import ManifoldParameter
from graphembed.utils import EPS

logger = logging.getLogger(__name__)
_default_manifold = Euclidean(1)


class RiemannianAdam(torch.optim.Optimizer):

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 nc=False,
                 max_grad_norm=None,
                 exact=False):
        if nc and betas[1] is not None:
            logger.warning('beta1=%.5f will be ignored because `nc` is True',
                           betas[1])
        defaults = dict(
                lr=lr,
                betas=betas,
                nc=nc,
                max_grad_norm=max_grad_norm,
                exact=exact)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                self._step(group)

        return loss

    def _step(self, group):
        lr = group['lr']
        beta1, beta2 = group['betas']
        max_grad_norm = group['max_grad_norm']

        for x in group['params']:
            grad = x.grad
            if grad is None:
                continue
            state = self.state[x]

            # initialize the state
            if len(state) == 0:
                state['step'] = 1
                # exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(x)
                # exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(x)

            if isinstance(x, ManifoldParameter) and x.manifold is not None:
                manifold = x.manifold
            else:
                manifold = _default_manifold
            retr = manifold.exp if group['exact'] else manifold.retr

            # get the Riemannian gradient
            grad = manifold.egrad2rgrad(x, grad)

            # do gradient clipping if needed
            grad_norm = manifold.norm(x, grad, keepdim=True)
            if max_grad_norm is not None:
                grad.mul_(torch.clamp(max_grad_norm / grad_norm, max=1.0))

            # make local variables for easy access
            step = state['step']
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']

            # the varying beta when using AdamNc
            if group['nc']:
                beta2 = 1 - 1 / step

            # actual step
            exp_avg.mul_(beta1).add_(1 - beta1, grad)
            exp_avg_sq.mul_(beta2).add_(1 - beta2, grad_norm.pow_(2))
            denom = exp_avg_sq.sqrt().add_(EPS[x.dtype])
            alpha = lr * (1 - beta2**step)**0.5 / (1 - beta1**step)

            direction = denom.div_(exp_avg).reciprocal_().mul_(-alpha)
            new_x = retr(x, direction)
            exp_avg_new = manifold.transp(x, new_x, exp_avg)

            # update the x and the state
            x.set_(new_x)
            exp_avg.set_(exp_avg_new)
            state['step'] += 1
