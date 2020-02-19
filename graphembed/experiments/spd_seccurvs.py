import logging
import logging.config
logging.config.fileConfig('logging.conf')

import argparse
import os
import random

import numpy as np
import torch

from graphembed.manifolds import SymmetricPositiveDefinite as SPD

from .agg_grid_results import load_best_embedding

# TODO(ccruceru): Is there a lower bound on the sectional curvaure for the SPD
# manifold?

parser = argparse.ArgumentParser(
        description='Compute SPD sectional curvatures.')
parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='The directory containing the saved embeddings.')
parser.add_argument(
        '--max_neigh_pairs',
        type=int,
        default=int(1e4),
        help='The maximum number of neighbor pairs to compute seccurvs for.')
args = parser.parse_args()

emb_state = torch.load(
        os.path.join(args.input, 'best_embedding.pth'), map_location='cpu')
for comp_id in range(len(emb_state) // 2):
    logging.warning('Processing %s, comp %d', args.input, comp_id)

    xs = emb_state[f'xs.{comp_id}']
    if xs.ndim != 3 and xs.shape[-2] != xs.shape[-1]:
        continue
    if torch.cuda.is_available():
        xs = xs.to('cuda')

    n_nodes, n, _ = xs.shape
    spd = SPD(n)

    def sectional_curvatures(i):
        x = xs[i]

        n_neighs = min(n_nodes, int(np.sqrt(args.max_neigh_pairs)))
        neighs = torch.as_tensor(random.sample(range(n_nodes), n_neighs))
        x_rep = x.repeat(n_neighs, 1, 1)
        x_logs = spd.log(x_rep, xs[neighs])

        x_rep = x_rep.repeat(n_neighs, 1, 1)
        us = x_logs.repeat(n_neighs, 1, 1)
        vs = x_logs.repeat_interleave(n_neighs, dim=0)
        seccurvs = spd.seccurv(x_rep, us, vs)
        return seccurvs

    samples = []
    for i in range(n_nodes):
        samples.append(sectional_curvatures(i))
    samples = torch.cat(samples).cpu().detach().numpy()

    out_file = os.path.join(args.input, f'spd-seccurvs_{comp_id}.npy')
    np.save(out_file, samples)
