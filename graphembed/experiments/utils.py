import os

MANIFOLD_IDENTIFIERS = ['euc', 'sph', 'hyp', 'spd', 'spdstein', 'grass', 'so']


def build_manifold(*names):
    from graphembed.manifolds import (Euclidean, Grassmann, Lorentz,
                                      SymmetricPositiveDefinite,
                                      SpecialOrthogonalGroup, Sphere)

    factors = []
    for name in names:
        parts = name.split('_')
        identifier = parts[0]
        if identifier in ['euc', 'sph', 'hyp', 'so', 'spd', 'spdstein']:
            n = int(parts[1])
        elif identifier in ['grass']:
            n1 = int(parts[1])
            n2 = int(parts[2])
        else:
            raise ValueError(f'Unkown manifold identifier {identifier}')

        if identifier == 'euc':
            man = Euclidean(n)
        elif identifier == 'sph':
            man = Sphere(n)
        elif identifier == 'hyp':
            man = Lorentz(n)
        elif identifier == 'so':
            man = SpecialOrthogonalGroup(n)
        elif identifier == 'spd':
            man = SymmetricPositiveDefinite(n)
        elif identifier == 'spdstein':
            man = SymmetricPositiveDefinite(n, use_stein_div=True)
        elif identifier == 'grass':
            man = Grassmann(n1, n2)

        factors.append(man)

    return factors


def manifold_label_for_display(*names):
    counts = {}
    for name in names:
        entry = tuple(name.split('_'))
        if entry[0] == 'grass':
            entry = ('gr', entry[1], entry[2])
        if entry in counts:
            counts[entry] += 1
        else:
            counts[entry] = 1

    def _label(entry):
        identifier = entry[0]
        if identifier in ['hyp', 'sph']:
            dim = int(entry[1]) - 1
            return '{}({})'.format(identifier.upper(), dim)
        elif identifier == 'gr':
            return '{}({},{})'.format(identifier.upper(), entry[2], entry[1])
        else:
            return '{}({})'.format(identifier.upper(), ','.join(entry[1:]))

    return ' x '.join(
            _label(e) if c == 1 else '{}^{}'.format(_label(e), c)
            for e, c in counts.items())


def get_color_for_manifold(name):
    if name.startswith('euc'):
        return 'tab:blue'
    elif name.startswith('spdstein'):
        return 'tab:pink'
    elif name.startswith('spd') or \
            name in ('grass_3_1', 'grass_4_1', 'grass_4_2'):
        return 'tab:green'
    elif name in ('grass_5_1'):
        return 'tab:purple'
    elif name.startswith('hyp') or name.startswith('sph'):
        return 'tab:orange'
    elif name in ('so_3', ):
        return 'tab:pink'
    return None


def manifold_label_for_paths(*names):
    return '+'.join(name for name in names)


def manifold_factors_from_path_label(label):
    return label.split('+')


def make_run_id(**kwargs):
    return ', '.join(f'{k}={v}' for k, v in kwargs.items())


def make_exp_dir(*args):
    from graphembed.utils import check_mkdir
    save_dir = os.path.join(*args)
    check_mkdir(save_dir, increment=False)
    return save_dir


def fullpath_list(d, only_dirs=True, only_files=False):
    for entry in os.listdir(d):
        fullpath = os.path.join(d, entry)
        if only_dirs and os.path.isdir(fullpath):
            yield fullpath
        if only_files and os.path.isfile(fullpath):
            yield fullpath


def set_seeds(seed):
    import numpy
    import random
    import torch
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
