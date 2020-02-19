import abc
import torch

from graphembed.utils import EPS


class Manifold(metaclass=abc.ABCMeta):

    @abc.abstractproperty
    def ndim(self):
        pass

    @abc.abstractproperty
    def dim(self):
        pass

    @abc.abstractmethod
    def zero(self, *shape, out=None):
        pass

    @abc.abstractmethod
    def zero_vec(self, *shape, out=None):
        pass

    @abc.abstractmethod
    def inner(self, x, u, v, keepdim=False):
        pass

    def norm(self, x, u, squared=False, keepdim=False):
        norm_sq = self.inner(x, u, u, keepdim)
        norm_sq.data.clamp_(EPS[u.dtype])
        return norm_sq if squared else norm_sq.sqrt()

    @abc.abstractmethod
    def proju(self, x, u, inplace=False):
        pass

    @abc.abstractmethod
    def projx(self, x, inplace=False):
        pass

    def egrad2rgrad(self, x, u):
        return self.proju(x, u)

    @abc.abstractmethod
    def exp(self, x, u):
        pass

    def retr(self, x, u):
        return self.exp(x, u)

    @abc.abstractmethod
    def log(self, x, y):
        pass

    def dist(self, x, y, squared=False, keepdim=False):
        return self.norm(x, self.log(x, y), squared, keepdim)

    def pdist(self, x, squared=False):
        assert x.ndim == self.ndim + 1
        n = x.shape[0]
        m = torch.triu_indices(n, n, 1, device=x.device)
        return self.dist(x[m[0]], x[m[1]], squared=squared, keepdim=False)

    def transp(self, x, y, u):
        return self.proju(y, u)

    @abc.abstractmethod
    def rand(self, *shape, out=None):
        pass

    def rand_uniform(self, *shape, out=None):
        raise NotImplementedError

    @abc.abstractmethod
    def randvec(self, x, norm=1):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass
