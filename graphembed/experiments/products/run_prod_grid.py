import logging
import logging.config
logging.config.fileConfig('logging.conf')

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import torch

from graphembed.data import load_graph_pdists, GraphDataset
from graphembed.manifolds import Euclidean
from graphembed.modules import ManifoldEmbedding as Embedding
from graphembed.objectives import (CurvatureRegularizer, KLDiveregenceLoss,
                                   QuotientLoss, Sum)
from graphembed.optim import RiemannianAdam
from graphembed.products import Embedding as UniversalEmbedding, TrainingEngine
from graphembed.pyx import FastPrecision
from graphembed.utils import check_mkdir


def main():
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Default torch settings.
    torch.set_default_dtype(torch.float64)
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)

    with ThreadPoolExecutor(max_workers=2) as tpool:
        for ds_path in args.datasets:
            ds_name = os.path.basename(ds_path).split('.')[0]

            # Load the graph.
            gpdists, g = load_graph_pdists(ds_path, cache_dir='.cached_pdists')
            n_nodes = g.number_of_nodes()
            ds = GraphDataset(gpdists if n_nodes < 5000 else gpdists.to('cpu'))
            fp = FastPrecision(g)

            for loss_fn_str in args.loss_fns:
                loss_fn, alpha = build_loss_fn(loss_fn_str)

                # Run the Euclidean baseline.
                if args.run_baseline:
                    for run_id in range(args.n_runs):
                        # Set the random seeds.
                        set_seeds(run_id)
                        # Create the output directory.
                        output_dir = make_exp_dir(args.save_dir, ds_name,
                                                  loss_fn_str, 'baseline',
                                                  str(run_id))
                        # Run the experiment.
                        exp_run_eucl(ds_name, ds, loss_fn, alpha, tpool, fp,
                                     output_dir)

                # Add the curvature regularizer if needed.
                # TODO(ccruceru): Put this into the naming of the loss function.
                if args.lambda_reg is not None:
                    loss_fn = Sum(loss_fn,
                                  CurvatureRegularizer(g, args.lambda_reg))

                # Run the products.
                for n_fact in args.factors:
                    for run_id in range(args.n_runs):
                        # Set the random seeds.
                        set_seeds(run_id)
                        # Create the output directory.
                        output_dir = make_exp_dir(args.save_dir, ds_name,
                                                  loss_fn_str, str(n_fact),
                                                  str(run_id))
                        # Run the experiment.
                        exp_run(ds_name, ds, loss_fn, alpha, n_fact, tpool, fp,
                                output_dir)


def exp_run(ds_name, ds, loss_fn, alpha, n_fact, tpool, fp, output_dir):
    man_args = dict(c_init=0.01, c_min=0.001, keep_sign_fixed=False)
    dims = [args.dim // n_fact] * n_fact
    emb = UniversalEmbedding(len(ds), dims, r_max=5, **man_args)

    optim = RiemannianAdam([
            dict(params=emb.xs, lr=0.05, exact=True),
            dict(params=emb.curvature_params, lr=0.05),
    ])
    training_engine = TrainingEngine(
            embedding=emb,
            optimizer=optim,
            objective_fn=loss_fn,
            alpha=alpha,
            n_epochs=5000,
            batch_size=4096,
            burnin_epochs=10,
            burnin_lower_lr=args.burnin_lower_lr,
            burnin_higher_lr=args.burnin_higher_lr,
            val_every_epochs=args.eval_every,
            save_every_epochs=1000,
            lazy_metrics={
                'Layer_Mean_F1': lambda p: \
                        tpool.submit(fp.layer_mean_f1_scores, p),
            },
            save_dir=output_dir)
    training_engine(ds)


def exp_run_eucl(ds_name, ds, loss_fn, alpha, tpool, fp, output_dir):
    emb = Embedding(len(ds), [Euclidean(args.dim)])
    optim = RiemannianAdam([
            dict(params=emb.xs, lr=0.05, exact=True),
    ])
    training_engine = TrainingEngine(
            embedding=emb,
            optimizer=optim,
            objective_fn=loss_fn,
            alpha=alpha,
            n_epochs=5000,
            batch_size=4096,
            burnin_epochs=10,
            burnin_lower_lr=args.burnin_lower_lr,
            burnin_higher_lr=args.burnin_higher_lr,
            val_every_epochs=args.eval_every,
            save_every_epochs=1000,
            lazy_metrics={
                'Layer_Mean_F1': lambda p: \
                        tpool.submit(fp.layer_mean_f1_scores, p),
            },
            save_dir=output_dir)
    training_engine(ds)


def build_loss_fn(loss_str):
    loss, param = loss_str.split('_')
    if loss == 'sne':
        return KLDiveregenceLoss(loss), float(param)
    elif loss == 'dist':
        inc_l2 = int(param) == 2
        return QuotientLoss(inc_l2=inc_l2), 1.0
    raise ValueError(f'Could not parse loss {loss_str}')


def make_exp_dir(*args):
    save_dir = os.path.join(*args)
    check_mkdir(save_dir, increment=False)
    return save_dir


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Graph embedding driver.')
    parser.add_argument('--dim', type=int, help='The manifolds dimension.')
    parser.add_argument(
            '--n_runs', type=int, help='The number of runs per configuration.')
    parser.add_argument('--datasets', nargs='+', type=str, help='The datasets.')
    parser.add_argument(
            '--loss_fns',
            nargs='+',
            type=str,
            help='The loss function encodings.')
    parser.add_argument(
            '--factors',
            nargs='+',
            type=int,
            help='The number of factors to use in the product.')
    parser.add_argument(
            '--lambda_reg',
            type=float,
            help='The lambda to use with the curvature regularizer.')
    parser.add_argument(
            '--run_baseline',
            action='store_true',
            help='Whether Euclidean baselines should be run.')
    parser.add_argument(
            '--burnin_lower_lr',
            action='store_true',
            help='Use a lower LR during burnin.')
    parser.add_argument(
            '--burnin_higher_lr',
            action='store_true',
            help='Use a higher LR during burnin.')
    parser.add_argument(
            '--eval_every',
            type=int,
            default=500,
            help='How often to run evaluation.')
    parser.add_argument(
            '--save_dir',
            type=str,
            help='The directory where the results will be saved.')
    parser.add_argument(
            '--verbose', action='store_true', help='Show the logs of each run.')
    args = parser.parse_args()
    if args.burnin_lower_lr and args.burnin_higher_lr:
        raise ValueError('`burnin_lower_lr` and `burnin_higher_lr` are '
                         'mutually exclusive.')
    if not args.factors and not args.run_baseline:
        raise ValueError('At least one of the factors or the baseline must be '
                         'run')
    if not args.factors:  # Allow running only the baseline!
        args.factors = []
    return args


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())
