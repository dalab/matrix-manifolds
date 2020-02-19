import abc

import torch
from torch.nn.functional import softplus

from graphembed.monitor import ManifoldParameterMonitoring as MPMonitor


class ManifoldParameter(torch.nn.Parameter):

    def __new__(cls, data=None, manifold=None, requires_grad=True):
        if data is None:
            data = torch.Tensor()
        instance = torch.Tensor._make_subclass(cls, data, requires_grad)
        instance.manifold = manifold
        return instance

    def proj_(self):
        self.manifold.projx(self, inplace=True)

    def __repr__(self):
        return "Parameter on {} containing:\n".format(
                self.manifold) + torch.Tensor.__repr__(self)


class EmbeddingBase(torch.nn.Module):

    @abc.abstractproperty
    def device(self):
        pass

    @abc.abstractproperty
    def curvature_params(self):
        pass

    def burnin(self, value=True):
        # Do not optimize curvature parameters during burnin.
        for p in self.curvature_params:
            p.requires_grad_(not value)


class ManifoldEmbedding(EmbeddingBase):

    def __init__(self, n, manifolds):
        self.n = n
        self.n_components = len(manifolds)
        self.manifolds = manifolds
        self.monitors = [MPMonitor(i, m) for i, m in enumerate(manifolds)]

        super().__init__()  # Initialize parent before adding parameters.
        self.xs = torch.nn.ParameterList([
                # Relies on default placement!  Use `~.to(device=device)`.
                ManifoldParameter(data=manifold.rand(n), manifold=manifold)
                for manifold in manifolds
        ])
        # Initialized to 0.5 because: softplus(0.5) ~ 1.
        self.scales = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.tensor(0.5)) for _ in manifolds])

    @property
    def device(self):
        return self.xs[0].device

    @property
    def curvature_params(self):
        return self.scales

    @torch.no_grad()
    def perturb(self, norm):
        for x, man in zip(self.xs, self.manifolds):
            x.set_(man.retr(x, man.randvec(x, norm)))

    @torch.no_grad()
    def stabilize(self):
        for x in self.xs:
            x.proj_()

    @torch.no_grad()
    def add_stats(self, writer, epoch):
        for i in range(self.n_components):
            self.monitors[i](self.xs[i], writer, epoch)
            writer.add_scalar(f'scale{i}', self.scales[i], epoch)

    def compute_dists(self, i=None):
        return sum([
                softplus(s) * man.pdist(x if i is None else x[i], squared=True)
                for x, s, man in zip(self.xs, self.scales, self.manifolds)
        ])

    def __len__(self):
        return self.n


class BatchedObjective(torch.nn.Module):

    def __init__(self, objective_fn, dataset, embedding):
        super().__init__()
        self.objective_fn = objective_fn
        self.dataset = dataset
        self.embedding = embedding

    def forward(self, indices, *args, **kwargs):
        return self.objective_fn(
                self.dataset[indices].to(self.embedding.device),
                self.embedding.compute_dists(indices), *args, **kwargs)
