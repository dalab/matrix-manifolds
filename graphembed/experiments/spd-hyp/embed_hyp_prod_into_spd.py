import logging
import logging.config
logging.config.fileConfig('logging.conf')

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import os
import sys

import torch
import matplotlib
matplotlib.use('Agg')

from graphembed.data import load_graph_pdists, GraphDataset
from graphembed.manifolds import Lorentz, SymmetricPositiveDefinite as SPD
from graphembed.modules import ManifoldEmbedding, ManifoldParameter
from graphembed.objectives import KLDiveregenceLoss
from graphembed.optim import RiemannianSGD
from graphembed.pyx import FastPrecision
from graphembed.train import TrainingEngine
from graphembed.utils import latest_path_by_basename_numeric_order

from utils import set_seeds


def main():
    args = parse_args()

    # Fix the random seeds.
    set_seeds(args.random_seed)

    # Default torch settings.
    torch.set_default_dtype(torch.float64)
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)

    # load data
    gpdists, g = load_graph_pdists(args.input_graph, cache_dir='.cached_pdists')
    n_nodes = g.number_of_nodes()
    ds = GraphDataset(gpdists)
    fp = FastPrecision(g)

    # run hyp2
    hyp = Lorentz(3)
    emb = ManifoldEmbedding(n_nodes, [hyp] * args.n_factors)
    for i in range(args.n_factors):
        emb.scales[i] = torch.nn.Parameter(torch.tensor(2.0))
    man_name = '_'.join('hyp2' for _ in range(args.n_factors))
    save_dir = os.path.join(args.save_dir, man_name)
    if args.hyp_snapshot or args.hyp_pretrained:
        logging.info('Loading embedding for %s', man_name)
        load_embedding(emb, save_dir)
    if not args.hyp_pretrained:
        train(ds, fp, emb, args.n_epochs, save_dir)

    # map it to SPD
    spd = SPD(2 * args.n_factors)
    spd_emb = ManifoldEmbedding(n_nodes, [spd])
    save_dir = os.path.join(args.save_dir, 'spd{}'.format(spd.dim))
    if args.spd_snapshot:
        logging.info('Loading embedding for SPD%d', spd.dim)
        load_embedding(spd_emb, save_dir)
    else:
        with torch.no_grad():
            spd_emb.xs[0] = ManifoldParameter(
                    block_diag([
                            h2_to_sspd2(emb.xs[i].mul(math.sqrt(2)))
                            for i in range(args.n_factors)
                    ]),
                    manifold=spd)
        hyp_dists = emb.to('cpu').compute_dists(None)
        spd_dists = spd_emb.compute_dists(None).to('cpu')
        assert torch.allclose(hyp_dists, spd_dists, atol=1e-4)

    # run spd2
    train(ds, fp, spd_emb, args.n_epochs, save_dir, args.n_epochs)


def load_embedding(emb, snapshot_path):
    pattern = os.path.join(snapshot_path, 'embedding_*.pth')
    path = latest_path_by_basename_numeric_order(pattern)
    emb.load_state_dict(torch.load(path))


def train(ds, fp, emb, n_epochs, save_dir, last_step=0):
    with ThreadPoolExecutor(max_workers=4) as tpool:
        training_engine = TrainingEngine(
                embedding=emb,
                optimizer=RiemannianSGD([
                    dict(params=emb.xs, lr=.05, max_grad_norm=100, exact=False),
                    # NOTE: We do not optimize the scaling parameters!
                ]),
                objective_fn=KLDiveregenceLoss(
                    inference_model='sne', inclusive=True),
                alpha=0.10,
                n_epochs=n_epochs,
                stabilize_every_epochs=20,
                val_every_epochs=5,
                lazy_metrics={
                    'Layer_Mean_F1': lambda p: \
                            tpool.submit(fp.layer_mean_f1_scores, p),
                },
                save_every_epochs=n_epochs // 10,
                save_dir=save_dir)
        training_engine(ds, last_step=last_step)


def h2_to_sspd2(x):
    shape = x.shape[:-1] + (1, 1)
    a = (x[..., 0] + x[..., 1]).view(shape)
    b = (x[..., 0] - x[..., 1]).view(shape)
    c = x[..., 2].view(shape)

    row1 = torch.cat([a, c], axis=-1)
    row2 = torch.cat([c, b], axis=-1)
    return torch.cat([row1, row2], axis=-2)


# Stolen from: https://github.com/yulkang/pylabyk
def block_diag(xs):
    if isinstance(xs, (list, tuple)):
        xs = torch.cat([x.unsqueeze(-3) for x in xs], -3)

    d = xs.ndim
    n = xs.shape[-3]
    siz0 = xs.shape[:-3]
    siz1 = xs.shape[-2:]
    xs = xs.unsqueeze(-2)
    eye = attach_dim(torch.eye(n, device=xs.device).unsqueeze(-2), d - 3, 1)
    return (xs * eye).reshape(siz0 + torch.Size(torch.tensor(siz1) * n))


def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
            torch.Size([1] * n_dim_to_prepend) + v.shape +
            torch.Size([1] * n_dim_to_append))


def parse_args():
    parser = argparse.ArgumentParser(
            description='Injecting HYP(2) embedding into SPD(2)')
    parser.add_argument(
            '--n_factors',
            type=int,
            default=1,
            help='The number of 2-dimensional hyperbolic factors to use.')
    parser.add_argument(
            '--input_graph',
            type=str,
            required=True,
            help='The path to the input graph.')
    parser.add_argument(
            '--hyp_pretrained',
            action='store_true',
            help='Whether the lorentz model has already been trained.')
    parser.add_argument(
            '--hyp_snapshot',
            action='store_true',
            help='Whether to continue optimizing hyp model from a snapshot.')
    parser.add_argument(
            '--spd_snapshot',
            action='store_true',
            help='Whether to continue optimizing spd model from a snapshot.')
    parser.add_argument(
            '--n_epochs',
            type=int,
            default=1000,
            help='The number of epochs to train for.')
    parser.add_argument(
            '--save_dir',
            type=str,
            help='The path where the transformed embedding should be saved.')
    parser.add_argument(
            '--random_seed',
            type=int,
            default=42,
            help='The manual random seed.')
    args = parser.parse_args()

    if args.hyp_pretrained and args.hyp_snapshot:
        raise ValueError('The args `hyp_pretrained` and `hyp_snapshot` are '
                         'mutually exclusive: either continue training or not.')

    return args


if __name__ == '__main__':
    sys.exit(main())
