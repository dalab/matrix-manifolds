from concurrent.futures import ProcessPoolExecutor
import os
import sys

import torch

from graphembed.modules import ManifoldEmbedding
from experiments.agg_angle_ratios import sample_angle_ratios_and_save
from experiments.utils import build_manifold, fullpath_list


def main():
    torch.set_default_dtype(torch.float64)

    with ProcessPoolExecutor(max_workers=args.num_cpus) as pool:
        futures = []
        for dim_dir in fullpath_list(args.root_dir, only_dirs=True):
            for man_dir in fullpath_list(dim_dir, only_dirs=True):
                man_name = os.path.basename(man_dir)
                if args.manifolds and man_name not in args.manifolds:
                    continue
                f = pool.submit(grid_fn, man_dir, man_name)
                futures.append(f)
        for f in futures:
            f.result(None)


def grid_fn(exp_dir, man_name):
    man = build_manifold(man_name)

    # Load the embedding.
    xs = torch.load(os.path.join(exp_dir, 'xs.pt'))
    emb = ManifoldEmbedding(len(xs), man)
    emb.burnin(True)
    for x in emb.xs:
        x.requires_grad_(False)
    emb.xs[0].set_(xs)

    # Sample the angle ratios.
    sample_angle_ratios_and_save(emb, args.force, args.num_random_triangles,
                                 exp_dir)


def parse_args():
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser(
            description='Angle ratios aggregator for compact manifolds.')
    parser.add_argument(
            '--root_dir',
            type=str,
            required=True,
            help='The root dir of the generated graphs hierarchy.')
    parser.add_argument(
            '--manifolds',
            nargs='*',
            type=str,
            help='The manifolds to restrict to.')
    parser.add_argument(
            '--num_random_triangles',
            type=int,
            default=int(1e4),
            help='The number of triangles to subsample.')
    parser.add_argument(
            '--force',
            action='store_true',
            help='Whether existing experiments should be re-run')
    parser.add_argument(
            '--num_cpus',
            type=int,
            default=multiprocessing.cpu_count(),
            help='The number of CPUs to use.')
    return parser.parse_args()


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())
