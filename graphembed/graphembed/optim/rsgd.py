import torch
from torch.optim.optimizer import required

from graphembed.manifolds import Euclidean
from graphembed.modules import ManifoldParameter

_default_manifold = Euclidean(1)


class RiemannianSGD(torch.optim.Optimizer):

    def __init__(self,
                 params,
                 lr=required,
                 momentum=0,
                 dampening=0,
                 max_grad_norm=None,
                 exact=False):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        defaults = dict(
                lr=lr,
                momentum=momentum,
                dampening=dampening,
                max_grad_norm=max_grad_norm,
                exact=exact)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        with torch.no_grad():  # no grad for repr/exp map and parallel transport
            for group in self.param_groups:
                self._step(group)

        return loss

    def _step(self, group):
        lr = group['lr']
        momentum = group['momentum']
        dampening = group['dampening']
        max_grad_norm = group['max_grad_norm']

        for x in group['params']:
            grad = x.grad
            if grad is None:
                continue
            state = self.state[x]

            # initialize the state
            if len(state) == 0 and momentum > 0:
                state["momentum_buffer"] = grad.clone()

            if isinstance(x, ManifoldParameter) and x.manifold is not None:
                manifold = x.manifold
            else:
                manifold = _default_manifold
            retr = manifold.exp if group['exact'] else manifold.retr

            # get the Riemannian gradient
            grad = manifold.egrad2rgrad(x, grad)

            # do gradient clipping if needed
            if max_grad_norm is not None:
                grad_norm = manifold.norm(x, grad, keepdim=True)
                grad.mul_(torch.clamp(max_grad_norm / grad_norm, max=1.0))

            # momentum-driven step; adjusts the momentum buffer too
            if momentum > 0:
                momentum_buffer = state['momentum_buffer']
                momentum_buffer.mul_(momentum).add_(1 - dampening, grad)
                new_x = retr(x, -lr * momentum_buffer)
                new_momentum_buffer = manifold.transp(x, new_x, momentum_buffer)

                x.set_(new_x)
                momentum_buffer.set_(new_momentum_buffer)

            # simple RSGD step
            else:
                x.set_(retr(x, -lr * grad))
