import logging
import logging.config
logging.config.fileConfig('logging.conf')

import argparse
import csv
import glob
import os
import sys
import traceback

import numpy as np
import torch

from graphembed.data import load_graph_pdists
from graphembed.pyx import FastPrecision
from graphembed.metrics import area_under_curve, average_distortion, pearsonr
from graphembed.modules import ManifoldEmbedding
from graphembed.utils import nnm1d2_to_n, Timer

from .utils import (build_manifold, fullpath_list,
                    manifold_factors_from_path_label,
                    manifold_label_for_display)

MAX_NUM_FACTORS = 4
HEADERS = [
        # DIMENSIONS
        'dataset', 'flip_probability', 'loss_fn', 'manifold', 'dim',

        # MEASURES
        'optimizers',
        'loss', 'loss_std',
        'dist', 'dist_std',
        'r', 'r_std',
        # first few f1 scores
        'f1_1', 'f1_1_std',
        'f1_2', 'f1_2_std',
        'f1_3', 'f1_3_std',
        'f1_4', 'f1_4_std',
        'f1_5', 'f1_5_std',
        # area under the F1 curve, up to 5, 10, and all layers
        'auc_5', 'auc_10', 'auc_total',

        # curvatures
        's1', 's1_std',
        's2', 's2_std',
        's3', 's3_std',
        's4', 's4_std',

        # NOTE: Add new fields only in the last position to avoid breaking the
        # Excel processing.
] # yapf: disable


def main():
    torch.set_default_dtype(torch.float64)

    csv_file = open(args.results_file, 'w')
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
        with Timer('constructing FastPrecision'):
            fp = FastPrecision(g)
        for flipp_dir in fullpath_list(ds_dir):
            flipp = os.path.basename(flipp_dir).split('_')[1]
            for loss_fn_dir in fullpath_list(flipp_dir):
                loss_fn_str = os.path.basename(loss_fn_dir)
                for man_dir in fullpath_list(loss_fn_dir):
                    man_name = os.path.basename(man_dir)
                    factor_names = manifold_factors_from_path_label(man_name)
                    man_factors = build_manifold(*factor_names)
                    man_label = manifold_label_for_display(*factor_names)

                    def add_dimensions(partial_entry):
                        partial_entry.update({
                                'dataset': ds_name,
                                'flip_probability': flipp,
                                'loss_fn': loss_fn_str,
                                'manifold': man_label,
                                'dim': sum([m.dim for m in man_factors]),
                        })  # yapf: disable
                        return partial_entry

                    # So I don't get confused again, note that this is OK w.r.t
                    # to flipp because when it is 0, there's only an 'orig' dir.
                    try:
                        partial_entry = process_exp_dir(man_dir, man_factors,
                                                        gpdists, fp)
                    except Exception as e:
                        logging.error('Failed to run for (%s): %s', man_dir,
                                      str(e))
                        continue
                    entry = add_dimensions(partial_entry)
                    csv_writer.writerow(entry)
    csv_file.close()


# TODO(ccruceru): Consider weighing the AUC with the ratio of nodes on the
# corresponding layer.

# TODO(ccruceru): Include in the F1@1 only nodes with degree less than some
# percentage of the total number of nodes in the graph.


def process_exp_dir(exp_dir, man_factors, gpdists, fp):
    run_dirs = list(fullpath_list(exp_dir))
    num_runs = len(run_dirs)
    num_pdists = len(gpdists)
    num_nodes = nnm1d2_to_n(num_pdists)
    num_factors = len(man_factors)

    all_losses = np.ndarray(shape=(num_runs))
    all_optims = []
    all_pdists = np.ndarray(shape=(num_pdists * num_runs))
    all_scales = np.zeros(shape=(num_runs, MAX_NUM_FACTORS))
    pearson_rs = np.ndarray(shape=(num_runs))
    distortions = np.ndarray(shape=(num_runs))

    for i, run_dir in enumerate(run_dirs):
        # Load the embedding.
        emb_state_dict, loss, optim_name = load_best_embedding(run_dir)
        embedding = ManifoldEmbedding(num_nodes, man_factors)
        embedding.load_state_dict(emb_state_dict)

        all_losses[i] = loss
        all_optims.append(optim_name)

        # Compute the manifold pairwise distances.
        mpdists = embedding.compute_dists(None).detach().sqrt_()
        indices = np.arange(i * num_pdists, (i + 1) * num_pdists)
        all_pdists[indices] = mpdists.numpy()

        # The scaling factors.
        all_scales[i, :num_factors] = np.sort(
                [s.item() for s in embedding.curvature_params]).flatten()

        # Compute the other metrics.
        pearson_rs[i] = pearsonr(mpdists, gpdists).item()

        # Compute the average distortion.
        distortions[i] = average_distortion(mpdists, gpdists).item()

    # Compute the F1 scores.
    with Timer('computing F1 scores'):
        f1_means, f1_stds = fp.layer_mean_f1_scores(mpdists.numpy())

    # Average the scaling factors.
    scale_means = np.mean(all_scales, axis=0)
    scale_stds = np.std(all_scales, axis=0)

    return {
            'optimizers': ','.join(all_optims),
            'loss': all_losses.mean(), 'loss_std': all_losses.std(),

            'dist': distortions.mean(), 'dist_std': distortions.std(),
            'r': pearson_rs.mean(), 'r_std': pearson_rs.std(),

            'f1_1': f1_means[0], 'f1_1_std': f1_stds[0],
            'f1_2': f1_means[1], 'f1_2_std': f1_stds[1],
            'f1_3': f1_means[2], 'f1_3_std': f1_stds[2],
            'f1_4': f1_means[3], 'f1_4_std': f1_stds[3],
            'f1_5': f1_means[4], 'f1_5_std': f1_stds[4],

            'auc_5': area_under_curve(f1_means[:5])[0],
            'auc_10': area_under_curve(f1_means[:10])[0],
            'auc_total': area_under_curve(f1_means)[0],

            's1': scale_means[0], 's1_std': scale_stds[0],
            's2': scale_means[1], 's2_std': scale_stds[1],
            's3': scale_means[2], 's3_std': scale_stds[2],
            's4': scale_means[3], 's4_std': scale_stds[3],
    }  # yapf: disable


def load_best_embedding(exp_dir):
    best_optim, best_loss = get_best_optim(exp_dir)
    path = os.path.join(exp_dir, best_optim, 'best_embedding.pth')
    best_embedding = torch.load(path, map_location='cpu')

    return best_embedding, best_loss, best_optim


def get_best_optim(exp_dir):
    best_optim = None
    best_loss = None
    for optim_dir in fullpath_list(exp_dir):
        optim_name = os.path.basename(optim_dir)
        loss_files = glob.glob(os.path.join(optim_dir, 'best_loss_*'))
        if len(loss_files) != 1:
            logging.warning(
                    'More than one (or no) loss file(s) found in %s. '
                    'Skipping it.', optim_dir)
            continue
        loss_file = loss_files[0]

        with open(loss_file, 'r') as f:
            loss = float(f.read())
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_optim = optim_name
    if best_loss is None:
        raise ValueError('Failed to get best optim for {}'.format(exp_dir))

    return best_optim, best_loss


def parse_args():
    parser = argparse.ArgumentParser(
            description='Graph embedding run aggregator.')
    parser.add_argument('--root_dir', type=str, help='The root directory.')
    parser.add_argument(
            '--results_file',
            type=str,
            default='./results.csv',
            help='The output CSV file.')
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
