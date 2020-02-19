import argparse
import csv
import glob
import os
import sys

import torch

from graphembed.data import load_graph_pdists
from graphembed.metrics import average_distortion, pearsonr, spearmanr
from graphembed.modules import ManifoldEmbedding

from .agg_grid_results import load_best_embedding
from .utils import (build_manifold, fullpath_list,
                    manifold_factors_from_path_label,
                    manifold_label_for_display)

HEADERS = [
        # DIMENSIONS
        'dataset', 'loss_fn', 'manifold', 'dim',

        # MEASURES
        'loss',
        'optimizer',
        'distortion',
        'pearson_r',
        'spearman_r',

        # the list of scaling factors
        'scaling_factors',

        # NOTE: Add new fields only in the last position to avoid breaking the
        # Excel processing.
] # yapf: disable


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
        gpdists, _ = load_graph_pdists(
                os.path.join('../data/dissimilarities', ds_name + '.npy'))
        gpdists.div_(gpdists.max())  # Normalization important for distortion!
        flipp_dir = os.path.join(ds_dir, 'flipp_0.0000')
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
                            'loss_fn': loss_fn_str,
                            'manifold': man_label,
                            'dim': sum([m.dim for m in man_factors]),
                    })  # yapf: disable
                    return partial_entry

                partial_entry = process_exp_dir(
                        os.path.join(man_dir, 'orig'), man_factors, gpdists)
                entry = add_dimensions(partial_entry)
                csv_writer.writerow(entry)
    csv_file.close()


def process_exp_dir(exp_dir, man_factors, gpdists):
    # Load the embedding.
    emb_state_dict, loss, optim_name = load_best_embedding(exp_dir)
    n_nodes = emb_state_dict['xs.0'].shape[0]
    embedding = ManifoldEmbedding(n_nodes, man_factors)
    embedding.load_state_dict(emb_state_dict)

    # The scaling factors.
    scaling_facts = []
    for i in range(len(man_factors)):
        cs_id = f'scales.{i}'
        if cs_id in emb_state_dict:
            scaling_facts.append(emb_state_dict[cs_id])
    scaling_facts_str = ','.join(['{:.2f}'.format(s) for s in scaling_facts])

    # Compute the pairwise distances on the manifold.
    mpdists = embedding.compute_dists(None).detach().sqrt_()

    # Compute the metrics.
    distortion = average_distortion(mpdists, gpdists).item()
    pearson_r = pearsonr(mpdists, gpdists).item()
    spearman_r = spearmanr(mpdists, gpdists).item()

    return {
            'loss': loss,
            'optimizer': optim_name,
            'distortion': distortion,
            'pearson_r': pearson_r,
            'spearman_r': spearman_r,

            'scaling_factors': scaling_facts_str,
    }  # yapf: disable


def parse_args():
    parser = argparse.ArgumentParser(
            description='Graph embedding run aggregator.')
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
