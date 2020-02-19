import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sl
import torch

from ..gen.gen_spd_graph import gen_ortho

n = 1
xs = gen_ortho(n, 3).numpy()
ys = gen_ortho(n, 3).numpy()

us = np.empty((n, 3, 3))
for i, (x, y) in enumerate(zip(xs, ys)):
    us[i] = sl.logm(x.T @ y)

steps = np.linspace(0, 1, 10)
for i, (x, u) in enumerate(zip(xs, us)):
    lhs = np.empty((len(steps), 2))
    rhs = np.empty((len(steps), 2))
    for j, t in enumerate(steps):
        y = torch.as_tensor(x @ sl.expm(t * u))
        eigs = y.eig().eigenvalues
        lhs[j] = eigs[0].numpy()
        rhs[j] = eigs[1].numpy()

    plt.plot(lhs[:, 0], lhs[:, 1])
    plt.show()
