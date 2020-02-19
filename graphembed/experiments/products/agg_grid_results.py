import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch

from graphembed.data import load_graph_pdists
from graphembed.pyx import FastPrecision
from graphembed.products import Embedding as UniversalEmbedding
from graphembed.manifolds import Euclidean
from graphembed.metrics import area_under_curve, average_distortion, pearsonr
from graphembed.modules import ManifoldEmbedding as Embedding
from graphembed.utils import latest_path_by_basename_numeric_order, Timer

from ..utils import fullpath_list

MAX_NUM_FACTORS = 6
HEADERS = [
        'dataset',
        'loss_fn',
        'n_factors',

        # curvatures
        'c1_mean', 'c1_std',
        'c2_mean', 'c2_std',
        'c3_mean', 'c3_std',
        'c4_mean', 'c4_std',
        'c5_mean', 'c5_std',
        'c6_mean', 'c6_std',

        # first few f1 scores
        'f1_1_mean', 'f1_1_std',
        'f1_2_mean', 'f1_2_std',
        'f1_3_mean', 'f1_3_std',
        'f1_4_mean', 'f1_4_std',
        'f1_5_mean', 'f1_5_std',
        'f1_10_mean', 'f1_10_std',

        # area under the F1 curve, up to 5, 10, and all layers
        'auc_5',
        'auc_10',
        'auc_total',

        # pearson R
        'r_mean', 'r_std',

        # distortion
        'dist_mean', 'dist_std',
]  # yapf: disable


def main():
    torch.set_default_dtype(torch.float64)

    csv_file = open('results.csv', 'w')
    csv_writer = csv.DictWriter(csv_file, delimiter=';', fieldnames=HEADERS)
    csv_writer.writeheader()

    for ds_dir in fullpath_list(args.root_dir):
        ds_name = os.path.basename(ds_dir)
        if args.datasets and ds_name not in args.datasets:
            continue
        # Load the dataset graph in order to compute F1 scores.
        gpdists, g = load_graph_pdists(
                os.path.join('../data', ds_name + '.edges.gz'),
                cache_dir='.cached_pdists')
        gpdists.div_(gpdists.max())  # Normalization important for distortion!
        n_nodes = g.number_of_nodes()
        with Timer('constructing FastPrecision'):
            fp = FastPrecision(g)
        for loss_fn_dir in fullpath_list(ds_dir):
            loss_fn = os.path.basename(loss_fn_dir)
            for n_factors_dir in fullpath_list(loss_fn_dir):
                n_factors = os.path.basename(n_factors_dir)
                n_factors = 0 if n_factors == 'baseline' else int(n_factors)
                run_dirs = list(fullpath_list(n_factors_dir))

                num_pdists = n_nodes * (n_nodes - 1) // 2
                n_runs = len(run_dirs)
                all_pdists = np.ndarray(shape=(num_pdists * n_runs))
                all_cs = np.zeros(shape=(n_runs, MAX_NUM_FACTORS))
                pearson_rs = np.ndarray(shape=(n_runs))
                distortions = np.ndarray(shape=(n_runs))

                for i, run_dir in enumerate(run_dirs):
                    # Load the embedding.
                    pattern = os.path.join(run_dir, 'embedding_*.pth')
                    path = latest_path_by_basename_numeric_order(pattern)
                    emb = load_embedding(path)

                    # The sorted curvatures.
                    if isinstance(emb, UniversalEmbedding):
                        cs = np.sort([-c.item()
                                      for c in emb.cuvature_params]).flatten()
                        all_cs[i, :n_factors] = cs

                    # Compute the manifold pairwise distances.
                    mpdists = emb.compute_dists(None)
                    mpdists.sqrt_()
                    indices = np.arange(i * num_pdists, (i + 1) * num_pdists)
                    all_pdists[indices] = mpdists.numpy()

                    # Compute the pearson R.
                    pearson_rs[i] = pearsonr(gpdists, mpdists)

                    # Compute the average distortion
                    distortions[i] = average_distortion(mpdists, gpdists)

                # Compute the F1 scores.
                with Timer('computing F1 scores'):
                    f1_means, f1_stds = fp.layer_mean_f1_scores(
                            all_pdists, n_runs)

                # Aggregate the metrics
                # - Pearson R
                r_mean = np.mean(pearson_rs)
                r_std = np.std(pearson_rs)
                # - average distortion
                dist_mean = np.mean(distortions)
                dist_std = np.std(distortions)

                # Average the curvatures.
                c_means = np.mean(all_cs, axis=0)
                c_stds = np.std(all_cs, axis=0)

                entry = {
                        'dataset': ds_name,
                        'loss_fn': loss_fn,
                        'n_factors': n_factors,

                        'c1_mean': c_means[0], 'c1_std': c_stds[0],
                        'c2_mean': c_means[1], 'c2_std': c_stds[1],
                        'c3_mean': c_means[2], 'c3_std': c_stds[2],
                        'c4_mean': c_means[3], 'c4_std': c_stds[3],
                        'c5_mean': c_means[4], 'c5_std': c_stds[4],
                        'c6_mean': c_means[5], 'c6_std': c_stds[5],

                        'f1_1_mean': f1_means[0], 'f1_1_std': f1_stds[0],
                        'f1_2_mean': f1_means[1], 'f1_2_std': f1_stds[1],
                        'f1_3_mean': f1_means[2], 'f1_3_std': f1_stds[2],
                        'f1_4_mean': f1_means[3], 'f1_4_std': f1_stds[3],
                        'f1_5_mean': f1_means[4], 'f1_5_std': f1_stds[4],
                        'f1_10_mean': f1_means[9], 'f1_10_std': f1_stds[9],

                        'auc_5': area_under_curve(f1_means[:5])[0],
                        'auc_10': area_under_curve(f1_means[:10])[0],
                        'auc_total': area_under_curve(f1_means)[0],

                        'r_mean': r_mean, 'r_std': r_std,

                        'dist_mean': dist_mean, 'dist_std': dist_std,
                }  # yapf: disable
                csv_writer.writerow(entry)

    csv_file.close()


def load_embedding(f):
    emb_state = torch.load(f, map_location='cpu')
    n_nodes, dim = emb_state['xs.0'].shape
    if 'scales.0' in emb_state:
        emb = Embedding(n_nodes, [Euclidean(dim)])
        emb.load_state_dict(emb_state)
    else:
        n_factors = len(emb_state) // 2
        emb = UniversalEmbedding(n_nodes, [dim] * n_factors)
        try:
            emb.load_state_dict(emb_state)
        except:
            i, j = 0, 0
            with torch.no_grad():
                for key, tensor in emb_state.items():
                    if key.startswith('xs'):
                        emb.xs[i].set_(tensor)
                        i += 1
                    elif key.startswith('c'):
                        emb.manifolds[j].c.set_(tensor)
                        j += 1
    emb.burnin(True)
    for x in emb.xs:
        x.requires_grad_(False)
    return emb


def parse_args():
    parser = argparse.ArgumentParser(
            description='Product embedding aggregator.')
    parser.add_argument('--root_dir', type=str, help='The root directory.')
    parser.add_argument(
            '--datasets',
            nargs='+',
            type=str,
            help='The datasets to restrict to.')
    return parser.parse_args()


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())
