import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import torch

from graphembed.manifolds import Grassmann, Sphere

num_points_geodesic = 20
num_points_great_circle = 200

grass = Grassmann(3, 2)
sph = Sphere(3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def add_circle(x, color):
    p1, p2 = x.T
    u = sph.log(p1, p2)
    ts = torch.linspace(0, 2 * np.pi, num_points_great_circle)
    us = u * ts.reshape(-1, 1)
    xs = p1.repeat(num_points_great_circle, 1)
    ys = sph.exp(xs, us)
    ax.scatter(ys[:, 0], ys[:, 1], ys[:, 2], color=color)


x, y = grass.rand_uniform(2)
u = grass.log(x, y)
xs = x.repeat(num_points_geodesic, 1, 1)
us = u * torch.linspace(0, 1.0, num_points_geodesic).reshape(-1, 1, 1)
ys = grass.exp(xs, us)

for i, y in enumerate(ys):
    add_circle(y, color=plt.cm.gnuplot(i / num_points_geodesic))

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
plt.tight_layout()
plt.savefig('plot.pdf')
