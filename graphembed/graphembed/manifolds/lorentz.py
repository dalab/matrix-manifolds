import torch
from torch.autograd import Function

from graphembed.manifolds.base import Manifold
from graphembed.manifolds.sphere import Sphere
from graphembed.utils import EPS


class Lorentz(Manifold):

    def __init__(self, n):
        self.n = n
        self.sphere = Sphere(self.n - 1)

    @staticmethod
    def to_poincare_ball(x):
        d = x.shape[-1] - 1
        return x.narrow(-1, 1, d) / (x.narrow(-1, 0, 1) + 1)

    @property
    def ndim(self):
        return 1

    @property
    def dim(self):
        return self.n - 1

    def zero(self, *shape, out=None):
        x = torch.zeros(*shape, self.n, out=out)
        x[..., 0] = 1
        return x

    def zero_vec(self, *shape, out=None):
        return torch.zeros(*shape, self.n, out=out)

    def inner(self, x, u, v, keepdim=False):
        return ldot(u, v, keepdim=keepdim)

    def proju(self, x, u, inplace=False):
        if not inplace:
            u = u.clone()
        return u.addcmul_(ldot(x, u, keepdim=True).expand_as(x), x)

    def projx(self, x, inplace=False):
        if not inplace:
            x = x.clone()
        t = 1 + x.narrow(-1, 1, self.n - 1).pow(2).sum(-1, keepdim=True)
        t.sqrt_()
        x.narrow(-1, 0, 1).copy_(t)
        return x

    def egrad2rgrad(self, x, u, inplace=False):
        if not inplace:
            u = u.clone()
        # Apply the inverse Riemannian metric first!
        u.narrow(-1, 0, 1).mul_(-1)
        return self.proju(x, u, inplace=True)

    def exp(self, x, u):
        un = ldot(u, u, keepdim=True)
        un.clamp_(min=0).sqrt_().clamp_(min=EPS[x.dtype])  # norm of ``u``
        return x.clone().mul_(un.cosh()).addcdiv_(un.sinh() * u, un)

    def log(self, x, y):
        xy = ldot(x, y, keepdim=True).clamp_(max=-1)
        denom = torch.sqrt(xy * xy - 1).clamp_(min=EPS[x.dtype])
        num = acosh(-xy)
        num.clamp_(min=EPS[x.dtype])
        u = num.div_(denom) * torch.addcmul(y, xy, x)
        return self.proju(x, u)

    def dist(self, x, y, squared=False, keepdim=False):
        d = -ldot(x, y)
        d.data.clamp_(min=1)
        dist = acosh(d)
        dist.data.clamp_(min=EPS[x.dtype])
        return dist.pow(2) if squared else dist

    def transp(self, x, y, u):
        xy = ldot(x, y, keepdim=True).expand_as(x)
        uy = ldot(u, y, keepdim=True).expand_as(x)
        return u + uy / (1 - xy) * (x + y)

    def rand(self, *shape, out=None, ir=1e-2):
        x = torch.empty(*shape, self.n, out=out).uniform_(-ir, ir)
        return self.projx(x, inplace=True)

    def randvec(self, x, norm=1):
        shape = x.shape[:-1]
        dirs = self.sphere.rand_uniform(*shape, out=x.new())
        dirs = torch.cat([torch.zeros(*shape, 1), dirs], dim=-1)
        vs = dirs.mul_(norm)
        zero = self.zero(*shape, out=x.new())
        us = self.transp(zero, x, vs)
        return us

    def __str__(self):
        return 'Lorentzian space of dimension {}'.format(self.n)


class LorentzDot(Function):

    @staticmethod
    def forward(ctx, u, v, keepdim):
        ctx.save_for_backward(u, v)
        ctx.keepdim = keepdim
        uv = u * v
        uv.narrow(-1, 0, 1).mul_(-1)
        return uv.sum(-1, keepdim=keepdim)

    @staticmethod
    def backward(ctx, g):
        u, v = ctx.saved_tensors
        if not ctx.keepdim:
            g = g.unsqueeze(-1).expand_as(u)
        g = g.clone()
        g.narrow(-1, 0, 1).mul_(-1)
        return g * v, g * u, None


def ldot(u, v, keepdim=False):
    return LorentzDot.apply(u, v, keepdim)


class Acosh(Function):

    @staticmethod
    def forward(ctx, x):
        z = torch.sqrt(x * x - 1)
        ctx.save_for_backward(z)
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z.data.clamp_(min=EPS[z.dtype])
        z = g / z
        return z, None


acosh = Acosh.apply
