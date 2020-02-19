import torch

n = (1 << 16) - 1
x = torch.rand(n, 2, 2, device='cuda').requires_grad_()

a = x[:, [0], [0]]
b = x[:, [1], [1]]
s = (a + b).sum()
s.backward()
assert torch.allclose(x.grad, torch.eye(2, out=x.new(2, 2)))

n = (1 << 19) - 8
x = torch.rand(n, 2, 2, device='cuda')
x = (x + x.transpose(-2, -1)).div_(2).add_(torch.eye(2, out=x.new(2, 2)))
y = x.detach().requires_grad_()
l = y.cholesky()
logdet = 2 * l.diagonal(dim1=1, dim2=2).log().sum()
logdet.backward()
assert torch.allclose(y.grad, y.inverse())
