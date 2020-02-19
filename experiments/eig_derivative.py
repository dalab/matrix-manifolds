import geoopt
import torch
from torch.autograd import grad, gradcheck, Function


class EigDiff(Function):

    @staticmethod
    def forward(ctx, x):
        ws, vs = torch.symeig(x, eigenvectors=True)
        ctx.save_for_backward(ws, vs)
        return ws.pow(2).sum()

    @staticmethod
    def backward(ctx, grad):
        ws, vs = ctx.saved_tensors
        grad_x = 2 * torch.einsum('ij,j,kj->ik', vs, ws, vs) * grad

        return torch.triu(grad_x) + torch.triu(grad_x.T, 1)


eig = EigDiff.apply

# check grad by finite differences
x = geoopt.manifolds.spd.multisym(
        torch.rand(3, 3, dtype=torch.float64, requires_grad=True))
gradcheck(eig, x, eps=1e-6, atol=1e-4)
