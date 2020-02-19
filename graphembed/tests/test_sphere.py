import math
import numpy as np
import torch

from graphembed.manifolds import Sphere
from utils import assert_allclose


def test_sphere_dist_poles():
    man = Sphere(5)
    x = torch.zeros(5)
    x[0] = 1.0
    y = torch.zeros(5)
    y[0] = -1.0
    assert_allclose(man.dist(x, y), math.pi)
