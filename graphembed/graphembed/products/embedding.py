import torch

from graphembed.manifolds import Universal
from graphembed.modules import EmbeddingBase, ManifoldParameter
from graphembed.monitor import ManifoldParameterMonitoring as Monitor


class Embedding(EmbeddingBase):

    def __init__(self, n, ds, r_max=5.0, **kwargs):
        super().__init__()
        self.n = n
        self.ds = ds
        self.r_max = r_max

        # parameters
        self.manifolds = torch.nn.ModuleList(
                [Universal(d, **kwargs) for d in self.ds])
        self.xs = torch.nn.ParameterList([
                # Relies on default placement!  Use `~.to(device=device)`.
                ManifoldParameter(data=man.rand(n), manifold=man)
                for man in self.manifolds
        ])

        # monitoring
        self.monitors = [Monitor(i, m) for i, m in enumerate(self.manifolds)]

    @property
    def device(self):
        return self.xs[0].device

    @property
    def curvature_params(self):
        for man in self.manifolds:
            yield man.c

    @torch.no_grad()
    def stabilize(self):
        for x in self.xs:
            # Norm constraint.
            norm = x.norm(p=2, dim=-1, keepdim=True)
            norm.div_(self.r_max).clamp_(min=1)
            x.div_(norm)

            # Project onto the manifold (for numerical stability).
            x.proj_()

    @torch.no_grad()
    def add_stats(self, writer, epoch):
        for i in range(len(self.manifolds)):
            self.monitors[i](self.xs[i], writer, epoch)
            writer.add_scalar(f'curv{i}', self.manifolds[i].get_K(), epoch)

    def compute_dists(self, indices=None):
        return sum([
                man.pdist(x if indices is None else x[indices], squared=True)
                for man, x in zip(self.manifolds, self.xs)
        ])

    def __len__(self):
        return self.n
