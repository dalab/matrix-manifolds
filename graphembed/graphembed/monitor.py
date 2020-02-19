import matplotlib.pyplot as plt

from graphembed.manifolds import (Euclidean, Lorentz, SymmetricPositiveDefinite,
                                  Sphere, Universal)
from graphembed.utils import PLT_MUTEX


class ManifoldParameterMonitoring:

    def __init__(self, identifier, manifold):
        self.identifier = identifier
        self.manifold = manifold

        # Poor man's sum types in Python.
        if isinstance(manifold, SymmetricPositiveDefinite):
            self._f = self._spd
        elif isinstance(manifold, Euclidean):
            self._f = self._eucl
        elif isinstance(manifold, Lorentz):
            self._f = self._hyp
        elif isinstance(manifold, Sphere):
            self._f = self._sphere
        elif isinstance(manifold, Universal):
            self._f = self._eucl
        else:
            self._f = self._do_nothing

    def __call__(self, x, writer, epoch):
        # Add the grad norms.
        # FIXME: Gradients cleared up under no_grad()?
        # grad_norms = self.manifold.norm(x, x.grad)
        # writer.add_histogram(self._tag('grad_norm'), grad_norms, epoch)
        # Add the manifold-specific stats.
        return self._f(x, writer, epoch)

    def _do_nothing(self, x, writer, epoch):
        pass

    def _spd(self, x, writer, epoch):
        w = self.manifold.symeig(x)
        w.data.clamp_(self.manifold.wmin, self.manifold.wmax)
        logw = w.log()  # log-eigenvalues
        logks = logw.max(1).values - logw.min(1).values  # log-condition numbers
        writer.add_histogram(self._tag('log_lambda'), logw.flatten(), epoch)
        writer.add_histogram(self._tag('log_k_X'), logks, epoch)

    def _eucl(self, x, writer, epoch):
        if self.manifold.dim == 2:
            with PLT_MUTEX:
                self._add_2d_embedding(x.cpu().detach().numpy(), writer, epoch)
        elif self.manifold.dim == 3:
            with PLT_MUTEX:
                self._add_3d_embedding(x.cpu().detach().numpy(), writer, epoch)

    def _hyp(self, x, writer, epoch):
        if self.manifold.dim == 2:
            y = Lorentz.to_poincare_ball(x.detach())
            self._add_2d_embedding(
                    y.cpu().detach().numpy(), writer, epoch, circle=True)
        elif self.manifold.dim == 3:
            y = Lorentz.to_poincare_ball(x.detach())
            self._add_3d_embedding(
                    y.cpu().detach().numpy(), writer, epoch, sphere=True)

    def _sphere(self, x, writer, epoch):
        if self.manifold.dim == 1:
            with PLT_MUTEX:
                self._add_2d_embedding(
                        x.cpu().detach().numpy(), writer, epoch, circle=True)
        elif self.manifold.dim == 2:
            with PLT_MUTEX:
                self._add_3d_embedding(
                        x.cpu().detach().numpy(), writer, epoch, sphere=True)

    def _add_2d_embedding(self, x, writer, epoch, circle=False):
        plt.scatter(x[:, 0], x[:, 1])
        if circle:
            plt.xlim(-1.05, 1.05)
            plt.ylim(-1.05, 1.05)
        writer.add_figure(self._tag('embedding'), plt.gcf(), epoch)

    def _add_3d_embedding(self, x, writer, epoch, sphere=False):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2])
        if sphere:
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_zlim(-1.05, 1.05)
        writer.add_figure(self._tag('embedding'), fig, epoch)

    def _tag(self, t):
        return f'{t}_comp{self.identifier}'
