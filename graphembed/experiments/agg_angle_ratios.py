import logging
import logging.config
logging.config.fileConfig('logging.conf')

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import math
import os
import random
import sys
import traceback

import torch
import numpy as np

from .utils import (build_manifold, fullpath_list,
                    manifold_factors_from_path_label)


def main():
    torch.set_default_dtype(torch.float64)
    # FIXME: There seems to be a PyTorch synchronization issue when applying it
    # on Grassman(5, 1) which makes it stuck somewhere in log.
    # with ThreadPoolExecutor(max_workers=args.n_cpus) as ppool:
    with ProcessPoolExecutor(max_workers=args.n_cpus) as ppool:
        futures = []
        # Go through the whole directory structure.
        ds_dirs = list(fullpath_list(args.root_dir))
        random.shuffle(ds_dirs)
        for ds_dir in ds_dirs:
            ds_name = os.path.basename(ds_dir)
            if args.datasets and ds_name not in args.datasets:
                continue
            for flipp_dir in fullpath_list(ds_dir):
                flipp = os.path.basename(flipp_dir).split('_')[1]
                loss_fn_dirs = list(fullpath_list(flipp_dir))
                random.shuffle(loss_fn_dirs)
                for loss_fn_dir in loss_fn_dirs:
                    loss_fn_str = os.path.basename(loss_fn_dir)
                    man_dirs = list(fullpath_list(loss_fn_dir))
                    random.shuffle(man_dirs)
                    for man_dir in man_dirs:
                        man_name = os.path.basename(man_dir)
                        if args.manifolds and man_name not in args.manifolds:
                            continue
                        man_factors = build_manifold(
                                *manifold_factors_from_path_label(man_name))
                        for ds_gen_dir in fullpath_list(man_dir):
                            gen_id = os.path.basename(ds_gen_dir)
                            for optim_dir in fullpath_list(ds_gen_dir):
                                optim_name = os.path.basename(optim_dir)
                                emb_file = os.path.join(optim_dir,
                                                        'best_embedding.pth')
                                if not os.path.isfile(emb_file):
                                    logging.warning('Embedding not found: %s',
                                                    emb_file)
                                    continue

                                emb = load_embedding(emb_file, man_factors)

                                # Submit it for processing.
                                f = ppool.submit(sample_angle_ratios_and_save,
                                                 emb, args.force,
                                                 args.num_random_triangles,
                                                 optim_dir)
                                futures.append(f)
        # Wait for the results.
        for f in futures:
            f.result()


def sample_angle_ratios_and_save(emb, force, num_random_triangles, save_dir):
    if not force and check_agg_file_exists(save_dir):
        logging.warning('Skipping previously processed experiment: %s',
                        save_dir)
        return

    try:
        samples = sample_angle_ratios(emb, num_random_triangles, save_dir)
        np.save(os.path.join(save_dir, 'angle_ratios.npy'), samples)
        logging.warning('Finished processing %s', save_dir)
    except Exception as e:
        logging.error('Failed to sample for (%s): %s', save_dir, str(e))
        traceback.print_exc()


def sample_angle_ratios(emb, num_random_triangles, exp_dir):
    n_nodes = len(emb)

    samples = []
    error_counter = 0
    for l in range(num_random_triangles):
        i, j, k = random.sample(range(n_nodes), 3)
        if i == j or i == k or j == k:
            continue
        try:
            theta1 = compute_angle(emb, i, j, k)
            theta2 = compute_angle(emb, j, i, k)
            theta3 = compute_angle(emb, k, i, j)
        except Exception as e:
            error_counter += 1
            continue

        theta_sum = theta1 + theta2 + theta3
        ratio = (theta_sum - np.pi) / np.pi / 2
        if not math.isnan(ratio):
            samples.append(ratio)
        else:
            error_counter += 1
    logging.warning('Failed to sample %d/%d for %s', error_counter,
                    num_random_triangles, exp_dir)

    return np.sort(samples)


def compute_angle(emb, i, j, k, eps=1e-4):
    manifolds = emb.manifolds
    n_factors = len(manifolds)

    xs = [emb.xs[l][i] for l in range(n_factors)]
    ys = [emb.xs[l][j] for l in range(n_factors)]
    zs = [emb.xs[l][k] for l in range(n_factors)]

    us = prod_log(manifolds, xs, ys)
    vs = prod_log(manifolds, xs, zs)

    uv_inner = prod_inner(manifolds, xs, us, vs, keepdim=True)
    u_norm = prod_norm(manifolds, xs, us, keepdim=True)
    v_norm = prod_norm(manifolds, xs, vs, keepdim=True)

    cos_theta = uv_inner / u_norm / v_norm
    cos_theta = np.clip(cos_theta, -1.0 + eps, 1.0 - eps)
    return np.arccos(cos_theta)


def prod_log(manifolds, xs, ys):
    return [man.log(x, y) for man, x, y in zip(manifolds, xs, ys)]


def prod_norm(manifolds, xs, us, keepdim=False):
    inner = prod_inner(manifolds, xs, us, us, keepdim=keepdim)
    inner.clamp_(min=0).sqrt_()
    return inner


def prod_inner(manifolds, xs, us, vs, keepdim=False):
    return sum([
            man.inner(x, u, v, keepdim=keepdim)
            for man, x, u, v in zip(manifolds, xs, us, vs)
    ])


def check_agg_file_exists(exp_dir):
    return os.path.isfile(os.path.join(exp_dir, 'angle_ratios.npy'))


def load_embedding(emb_file, man_factors):
    from graphembed.modules import ManifoldEmbedding

    emb_state_dict = torch.load(emb_file, map_location='cpu')
    n_nodes = emb_state_dict['xs.0'].shape[0]
    embedding = ManifoldEmbedding(n_nodes, man_factors)
    embedding.load_state_dict(emb_state_dict)
    # No gradients.
    embedding.burnin(True)
    for x in embedding.xs:
        x.requires_grad_(False)
    # Do a projection.
    embedding.stabilize()

    return embedding


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Angle ratios aggregator.')
    parser.add_argument('--root_dir', type=str, help='The root directory.')
    parser.add_argument(
            '--datasets',
            nargs='+',
            type=str,
            help='The datasets to restrict to.')
    parser.add_argument(
            '--manifolds', nargs='*', type=str, help='The manifolds.')
    parser.add_argument(
            '--num_random_triangles',
            type=int,
            default=int(1e4),
            help='The number of triangles to subsample.')
    parser.add_argument(
            '--force',
            action='store_true',
            help='Whether existing experiments should be re-run')
    parser.add_argument('--n_cpus', type=int, help='The number of CPUs to use.')
    return parser.parse_args()


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())
