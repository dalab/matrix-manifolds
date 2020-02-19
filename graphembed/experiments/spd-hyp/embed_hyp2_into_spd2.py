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

from utils import set_seeds

# NOTE: This is not entirely superseded by `embed_hyp_prod_into_spd` because
# here we allow the optimization of the curvature of one H2. Once we go to
# products, there is no way to isometrically embed with different radiuses to
# preserve the distances in SPD(n).


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
    emb = ManifoldEmbedding(n_nodes, [Lorentz(3)])
    path = os.path.join(args.save_dir, 'hyp2')
    train(ds, fp, emb, args.n_epochs, path)
    curvature_sq = 1 / emb.scales[0]

    # map it to SSPD
    sspd_emb = ManifoldEmbedding(n_nodes, [SPD(2)])
    sspd_emb.xs[0] = ManifoldParameter(
            h2_to_sspd2(emb.xs[0] / curvature_sq.sqrt()),
            manifold=sspd_emb.manifolds[0])
    sspd_emb.scales[0] = torch.nn.Parameter(1 / curvature_sq / 2)
    assert torch.allclose(
            emb.compute_dists(None), sspd_emb.compute_dists(None), atol=1e-4)

    # run spd2
    path = os.path.join(args.save_dir, 'spd2')
    train(ds, fp, sspd_emb, args.n_epochs, path, args.n_epochs)


def train(ds, fp, emb, n_epochs, save_dir, last_step=0):
    with ThreadPoolExecutor(max_workers=4) as tpool:
        training_engine = TrainingEngine(
                embedding=emb,
                optimizer=RiemannianSGD([
                    dict(params=emb.xs, lr=.05, max_grad_norm=100, exact=False),
                    dict(params=emb.scales, lr=1e-5),
                ]),
                objective_fn=KLDiveregenceLoss(
                    inference_model='sne', inclusive=True, n_nodes=len(ds)),
                alpha=0.10,
                n_epochs=n_epochs,
                stabilize_every_epochs=20,
                val_every_epochs=5,
                lazy_metrics={
                    'Layer_Mean_F1': lambda p: \
                            tpool.submit(fp.layer_mean_f1_scores, p),
                },
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


def parse_args():
    parser = argparse.ArgumentParser(
            description='Injecting HYP(2) embedding into SPD(2)')
    parser.add_argument(
            '--input_graph', type=str, help='The path to the input graph.')
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
    return parser.parse_args()


if __name__ == '__main__':
    sys.exit(main())
