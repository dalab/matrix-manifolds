import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import torch

from graphembed.manifolds import SymmetricPositiveDefinite as SPD

spd = SPD(2)
x, y = spd.rand(2, ir=3.0)
u = spd.log(x, y)

n = 100
ys = spd.exp(x.repeat(n, 1, 1), torch.linspace(0, 1.0, n).reshape(n, 1, 1) * u)


def add_ellipsis(x, offset):
    ws, us = x.symeig(eigenvectors=True)
    rad = torch.atan2(us[0][1], us[0][0])
    degs = np.rad2deg(rad)
    ellipse = Ellipse(xy=(offset, 0), width=ws[0], height=ws[1], angle=degs)

    max_x = max(rad.cos().abs() * ws[0], rad.sin().abs() * ws[1]) / 2
    max_y = max(rad.sin().abs() * ws[0], rad.cos().abs() * ws[1]) / 2
    return ellipse, max_x, max_y


fig, ax = plt.subplots()
max_width = 0
max_height = 0
min_width = 0
for i, y in enumerate(ys):
    offset = i * 0.1
    e, width, height = add_ellipsis(y, offset=offset)
    e.set_facecolor(plt.cm.gnuplot(i / n))
    e.set_alpha(0.8)
    ax.add_artist(e)
    min_width = min(min_width, -width)
    max_width = max(max_width, width + offset)
    max_height = max(max_height, height)

plt.axis('off')
plt.xlim(1.01 * min_width, 1.01 * max_width)
plt.ylim(-1.01 * max_height, 1.01 * max_height)
plt.show()
plt.tight_layout()
fig.savefig('plot.pdf')
