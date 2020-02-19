import logging
import logging.config
logging.config.fileConfig('logging.conf')

from concurrent.futures import ThreadPoolExecutor
import glob
import os
import sys
import traceback

import networkx as nx
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')

from graphembed.optim import RiemannianAdam, RiemannianSGD
from graphembed.utils import nnm1d2_to_n

from .utils import (build_manifold, make_exp_dir, make_run_id,
                    manifold_label_for_paths, manifold_label_for_display,
                    set_seeds)

optimizers = [
        ('adam', lambda emb: RiemannianAdam([
                dict(params=emb.xs, lr=0.01, exact=True, max_grad_norm=100),
                dict(params=emb.curvature_params, lr=0.01)
        ])),
        ('sgd', lambda emb: RiemannianSGD([
                dict(params=emb.xs, lr=0.01, exact=True, max_grad_norm=20),
                dict(params=emb.curvature_params, lr=1e-4, max_grad_norm=500),
        ])),
        ('adam-no-curv', lambda emb: RiemannianAdam([
                dict(params=emb.xs, lr=0.01, exact=True, max_grad_norm=100),
        ])),
]


def main():
    if not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)
    # Fix the random seeds.
    set_seeds(args.random_seed)

    # Default torch settings.
    torch.set_default_dtype(torch.float64)
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)

    with ThreadPoolExecutor(max_workers=4) as tpool:
        for ds_path in args.datasets:
            ds_name = os.path.basename(ds_path).split('.')[0]
            for flipp in args.flip_probabilities:
                for ds_gen_id, g, ds, fp in \
                        ds_generator(ds_path, args.num_runs, flipp):
                    for loss_fn_str in args.loss_fns:
                        for factor_names in args.manifold:
                            for optim_name, make_optim in optimizers:
                                _exp_run(ds_name, flipp, ds_gen_id, ds, fp,
                                         loss_fn_str, factor_names, optim_name,
                                         make_optim, tpool)
                    # Save the random graphs to disk for later inspection.
                    if flipp != 0:
                        filename = os.path.join(args.save_dir, ds_name,
                                                f'flipp_{flipp:.4f}',
                                                f'{ds_gen_id}.edges.gz')
                        # Make sure we do not overwrite it if it has been run in
                        # a prevoius execution of this script.
                        if not os.path.isfile(filename):
                            nx.write_edgelist(g, filename)


def _exp_run(ds_name, flipp, ds_gen_id, ds, fp, loss_fn_str, factor_names,
             optim_name, make_optim, tpool):
    loss_fn, alpha = build_loss_fn(loss_fn_str)
    man_factors = build_manifold(*factor_names)

    save_dir = make_exp_dir(
            args.save_dir, ds_name, f'flipp_{flipp:.4f}', loss_fn_str,
            manifold_label_for_paths(*factor_names), ds_gen_id, optim_name)
    run_id = make_run_id(
            dataset=ds_name,
            fp=flipp,
            loss_fn=loss_fn_str,
            manifold=manifold_label_for_display(*factor_names),
            version=ds_gen_id,
            optim=optim_name)

    # Check if we should force re-running if the directory exists.
    if not args.force and check_exp_exists(save_dir):
        logging.warning('Skipping existing experiment: %s', run_id)
        return

    logging.warning('Running for (%s)', run_id)
    try:
        exp_run(ds, fp, loss_fn, alpha, man_factors, make_optim, tpool,
                save_dir)
    except Exception as e:
        logging.error('Failed to run for (%s): %s', run_id, str(e))
        traceback.print_exc()


def check_exp_exists(exp_dir):
    return os.path.isfile(os.path.join(exp_dir, 'best_embedding.pth')) and \
            len(glob.glob(os.path.join(exp_dir, 'best_loss_*'))) > 0


def exp_run(ds, fp, loss_fn, alpha, man_factors, make_optim, tpool, save_dir):
    from graphembed.modules import ManifoldEmbedding
    from graphembed.train import TrainingEngine

    embedding = ManifoldEmbedding(len(ds), man_factors)
    optimizer = make_optim(embedding)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=50, threshold=1e-4, min_lr=1e-5)

    lazy_metrics = None
    if fp:
        lazy_metrics = {
                'Layer_Mean_F1': lambda p: \
                        tpool.submit(fp.layer_mean_f1_scores, p),
        }
    training_engine = TrainingEngine(
            embedding=embedding,
            optimizer=optimizer,
            objective_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            min_lr=1e-5,  # Early breaking.
            alpha=alpha,
            n_epochs=3000,
            batch_size=512,
            burnin_epochs=10,
            burnin_lower_lr=True,  # Use lower learning rate during burnin.
            stabilize_every_epochs=5,
            val_every_epochs=args.eval_every,
            lazy_metrics=lazy_metrics,
            save_dir=save_dir)
    training_engine(ds)


def ds_generator(ds_path, num_runs, flipp=None):
    from graphembed.data import load_graph_pdists, GraphDataset
    from graphembed.pyx import FastPrecision

    def _load(flipp):
        gpdists, g = load_graph_pdists(
                ds_path, cache_dir='.cached_pdists', flip_probability=flipp)
        assert args.no_fp or g is not None
        n_nodes = nnm1d2_to_n(len(gpdists))
        ds = GraphDataset(gpdists if n_nodes < 5000 else gpdists.to('cpu'))
        fp = None if args.no_fp else FastPrecision(g)
        return g, ds, fp

    if not flipp or flipp == 0:
        assert num_runs == 1
        yield ('orig', *_load(flipp=None))
    else:
        for i in range(num_runs):
            yield (f'gen_{i}', *_load(flipp=flipp))


def build_loss_fn(loss_str):
    from graphembed.objectives import (KLDiveregenceLoss, StressLoss,
                                       QuotientLoss)
    if loss_str == 'stress':
        return StressLoss(), None

    loss, param = loss_str.split('_')
    if loss == 'sne-incl':
        return KLDiveregenceLoss('sne', inclusive=True), float(param)
    elif loss == 'sne-excl':
        return KLDiveregenceLoss('sne', inclusive=False), float(param)
    elif loss == 'sste-incl':
        return KLDiveregenceLoss('sste', inclusive=True), float(param)
    elif loss == 'sste-excl':
        return KLDiveregenceLoss('sste', inclusive=False), float(param)
    elif loss == 'dist':
        inc_l2 = int(param) == 2
        return QuotientLoss(inc_l2=inc_l2), 1.0

    raise ValueError(f'Could not parse loss {loss_str}')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Graph embedding driver.')
    parser.add_argument('--datasets', nargs='+', type=str, help='The datasets.')
    parser.add_argument(
            '--manifold',
            action='append',
            nargs='+',
            help='The components of the (product) manifolds.')
    parser.add_argument(
            '--loss_fns',
            nargs='+',
            type=str,
            help='The loss functions to use.')
    parser.add_argument(
            '--flip_probabilities',
            nargs='*',
            type=float,
            help='The flip probability used to generate several noisy '
            'versions of the input graph')
    parser.add_argument(
            '--num_runs',
            type=int,
            default=1,
            help='The number of noisy graphs to run on.')
    parser.add_argument(
            '--save_dir',
            type=str,
            help='The directory where the results will be saved.')
    parser.add_argument(
            '--no_fp',
            action='store_true',
            help='Do not evaluate the F1@k metric.')
    parser.add_argument(
            '--eval_every',
            type=int,
            help='Once every this many epochs the evaluation will be run.')
    parser.add_argument(
            '--force',
            action='store_true',
            help='Whether existing experiments should be re-run')
    parser.add_argument(
            '--verbose', action='store_true', help='Show the logs of each run.')
    parser.add_argument(
            '--random_seed',
            type=int,
            default=42,
            help='The manual random seed.')
    args = parser.parse_args()

    if not args.manifold:
        raise ValueError('At least a manifold must be specified.')
    if not args.flip_probabilities:
        args.flip_probabilities = [0]
    if not args.save_dir:
        import datetime
        now = datetime.datetime.now()
        args.save_dir = 'runs/grid-{}'.format(now.strftime('%H_%M_%S'))

    return args


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())
