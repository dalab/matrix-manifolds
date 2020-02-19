import torch

mat = torch.randn(4, 4, dtype=torch.float64)
mat = (mat @ mat.transpose(-1, -2)).div_(2).add_(torch.eye(4, dtype=torch.float64))
mat = mat.detach().clone().requires_grad_(True)
mat_clone = mat.detach().clone().requires_grad_(True)

# Way 1
chol_mat = mat.cholesky()
logdet1 = 2 * chol_mat.diagonal().log().sum()

# Way 2
w, _ = mat_clone.symeig(eigenvectors=True)
logdet2 = w.log().sum()

print('Are these both log(det(A))?', bool(logdet1 - logdet2 < 1e-8))

logdet1.backward()
logdet2.backward()

inv_mat = mat.inverse()
print('Does Way 1 yield A^{-1}?', bool(torch.norm(mat.grad - inv_mat) < 1e-8))
print('Does Way 2 yield A^{-1}?', bool(torch.norm(mat_clone.grad - inv_mat) < 1e-8))
