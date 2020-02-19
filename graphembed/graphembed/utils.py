import glob
import logging
import math
from multiprocessing import Lock
import os
import re
import time

import torch

logger = logging.getLogger(__name__)

EPS = {torch.float32: 1e-8, torch.float64: 1e-8}

# The mutex used to sync access to matplotlib in case several instances of the
# training engine are used in parallel.
PLT_MUTEX = Lock()


def nnp1d2_to_n(m):
    r"""Computes n from :math:`n (n + 1) / 2`."""
    n = math.floor(math.sqrt(2 * m))
    assert n * (n + 1) // 2 == m
    return n


def nnm1d2_to_n(m):
    r"""Computes n from :math:`n (n - 1) / 2`."""
    n = math.ceil(math.sqrt(2 * m))
    assert n * (n - 1) // 2 == m
    return n


def triu_mask(n, m=None, *, d=0, device=None):
    r"""Returns a binary mask with ones above the ``d``th diagonal of an n x m
    matrix.

    NOTE: This should be more memory-efficient than using ``torch.triu_indices``
    which uses LongTensors.
    """
    if not m:
        m = n
    return torch.ones(
            n, m, dtype=torch.uint8, device=device).triu_(diagonal=d).bool()


def squareform1(x_vec):
    r"""Converts a vector-form distance vector to a square-form matrix, and
    vice-versa. See :func:`scipy.spatial.distance.squareform`.
    """
    # The 'tovector' case.
    if x_vec.ndim >= 2 and x_vec.shape[-2] == x_vec.shape[-1]:
        m = triu_mask(x_vec.shape[-1], d=1, device=x_vec.device)
        return x_vec.masked_select(m).view(*x_vec.shape[:-2], -1)

    # The 'tomatrix' case.
    else:
        n = nnm1d2_to_n(x_vec.shape[-1])
        shape = x_vec.shape[:-1] + (n, n)
        x = torch.zeros(shape, out=x_vec.new(shape))
        m = triu_mask(n, d=1, device=x_vec.device)
        x.masked_scatter_(m, x_vec)
        x.transpose(dim0=-2, dim1=-1).masked_scatter_(m, x_vec)

        return x


def squareform0(x_vec):
    r"""Converts a vector-form upper-triangular part of a square matrix to its
    square form, and vice-versa. Works the same as :func:`.squareform` but
    including the diagonal elements.
    """
    # The 'tovector' case.
    if x_vec.ndim >= 2 and x_vec.shape[-2] == x_vec.shape[-1]:
        m = triu_mask(x_vec.shape[-1], device=x_vec.device)
        return x_vec.masked_select(m).view(*x_vec.shape[:-2], -1)

    # The 'tomatrix' case.
    else:
        n = nnp1d2_to_n(x_vec.shape[-1])
        shape = x_vec.shape[:-1] + (n, n)
        x = torch.empty(shape, out=x_vec.new(shape))
        m = triu_mask(n, device=x_vec.device)
        x.masked_scatter_(m, x_vec)
        x.transpose(dim0=-2, dim1=-1).masked_scatter_(m, x_vec)

        return x


def basename_numeric_order(path):
    n = [int(s) for s in re.findall(r'\d+', os.path.basename(path))]
    return n[-1]


def latest_path_by_basename_numeric_order(pattern):
    paths = glob.glob(pattern)
    if not paths:
        return None
    return max(paths, key=basename_numeric_order)


def check_mkdir(path, increment=False):
    r"""Only creates a directory if it does not exist already.  Emits an
    warning if it exists. When 'increment' is true, it creates a directory
    nonetheless by incrementing an integer at its end.
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        if increment:
            trailing_int = 0
            while os.path.isdir(path):
                basename = os.path.basename(path)
                split = basename.split('_')
                if split[-1].isdigit():
                    basename = '_'.join(split[:-1])
                path = os.path.join(
                        os.path.dirname(path),
                        basename + '_{}'.format(trailing_int))
                trailing_int += 1
            os.makedirs(path)
            logger.info('Created the directory (%s) instead', path)
        else:
            logger.warning('The given path already exists (%s)', path)

    return path


class Timer:

    def __init__(self, msg, precision=4, loglevel=logging.DEBUG):
        self.msg = msg
        self.precision = precision
        self.loglevel = loglevel

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        msg = 'time({}): {:.{precision}f}s'.format(
                self.msg, self.interval, precision=self.precision)
        logger.log(self.loglevel, msg)
