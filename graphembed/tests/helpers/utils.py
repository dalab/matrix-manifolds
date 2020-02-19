import numpy as np
import torch


def assert_allclose(x, y, **kwargs):
    if isinstance(x, torch.Tensor):
        x = x.cpu()
    if isinstance(y, torch.Tensor):
        y = y.cpu()
    np.testing.assert_allclose(x, y, **kwargs)
