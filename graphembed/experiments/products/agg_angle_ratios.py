from concurrent.futures import ProcessPoolExecutor
import os
import sys

import torch

from graphembed.utils import latest_path_by_basename_numeric_order

from .agg_grid_results import load_embedding
from ..agg_angle_ratios import sample_angle_ratios_and_save, parse_args
from ..utils import fullpath_list


def main():
    torch.set_default_dtype(torch.float64)
    with ProcessPoolExecutor(max_workers=args.n_cpus) as ppool:
        futures = []
        for ds_dir in fullpath_list(args.root_dir):
            ds_name = os.path.basename(ds_dir)
            for loss_fn_dir in fullpath_list(ds_dir):
                loss_fn = os.path.basename(loss_fn_dir)
                for n_factors_dir in fullpath_list(loss_fn_dir):
                    try:
                        n_factors = int(os.path.basename(n_factors_dir))
                    except:
                        continue  # Ignore the Euclidean baseline.
                    for run_dir in fullpath_list(n_factors_dir):
                        # Load the embedding.
                        pattern = os.path.join(run_dir, 'embedding_*.pth')
                        path = latest_path_by_basename_numeric_order(pattern)
                        emb = load_embedding(path)

                        # Submit it for processing.
                        f = ppool.submit(sample_angle_ratios_and_save, emb,
                                         run_dir)
                        futures.append(f)
        # Wait for the results.
        for f in futures:
            f.result()


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())
