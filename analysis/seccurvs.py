import argparse
import itertools
import logging
import multiprocessing
import os
import random
import sys

import networkit as nk
import networkx as nx
import numpy as np

from utils import np_output_filename

parser = argparse.ArgumentParser(
        description='Compute graph sectional curvatures')
parser.add_argument('--input', type=str, required=True, help='The input graph.')
parser.add_argument('--force', action='store_true', help='Re-generate them.')
parser.add_argument(
        '--min_num_nodes',
        type=int,
        default=100,
        help='The minimum number of nodes in the largest connected '
        'component to keep.')
parser.add_argument(
        '--sample_ratio',
        type=float,
        default=0.5,
        help='The percentage of all nodes to use as reference nodes `a`.')
parser.add_argument(
        '--max_neigh_pairs',
        type=int,
        default=int(1e4),
        help='The maximum number of neighbor pairs to compute seccurvs for.')
parser.add_argument(
        '--inherit_filename',
        action='store_true',
        help='Whether the file format of the degrees file should be inheritted '
        'from the input.')
parser.add_argument(
        '--n_cpus',
        type=int,
        default=multiprocessing.cpu_count(),
        help='The number of CPUs used for parallelization.')
args = parser.parse_args()

out_file = np_output_filename(args.input, 'seccurvs', args.inherit_filename)
if os.path.isfile(out_file) and not args.force:
    logging.warning('The sectional curvatures already exist: %s', out_file)
    sys.exit(0)

# load the graph
g = nx.convert_node_labels_to_integers(nx.read_edgelist(args.input))
connected_components = list(nx.connected_components(g))
if len(connected_components) == 0:
    logging.fatal('Empty graph. This is most probably due to a too small a '
                  'distance threshold.')
    sys.exit(1)
if len(connected_components) > 1:
    logging.warning('The input graph has %d connected components. ',
                    len(connected_components))
    g = g.subgraph(max(connected_components, key=len))
    g = nx.convert_node_labels_to_integers(g)
    num_nodes = g.number_of_nodes()
    if num_nodes > args.min_num_nodes:
        logging.warning('Keeping the largest only with %d nodes.', num_nodes)
    else:
        logging.fatal(
                'The largest connected component has %d nodes. Dropping '
                'this scenario.', num_nodes)
        sys.exit(1)
n_nodes = g.number_of_nodes()
num_ref_nodes = int(args.sample_ratio * n_nodes)

# compute the shortest paths
gk = nk.nxadapter.nx2nk(g)
shortest_paths = nk.distance.APSP(gk).run().getDistances()
dists = {u: {} for u in range(n_nodes)}
for i, u in enumerate(g.nodes()):
    for j, v in enumerate(g.nodes()):
        dists[u][v] = int(shortest_paths[i][j])


# compute the sectional curvature samples
def sectional_curvatures(m):
    seccurvs = []
    neighs = list(g[m])
    n_neighs = len(neighs)
    neigh_pairs = [(neighs[i], neighs[j])
                   for i in range(n_neighs)
                   for j in range(i + 1, n_neighs)]
    for b, c in random.sample(neigh_pairs,
                              min(args.max_neigh_pairs, len(neigh_pairs))):
        xis = []
        for a in random.sample(range(n_nodes), num_ref_nodes):
            if a == m: continue
            xi = dists[a][m]**2 + dists[b][c]**2 / 4 - \
                    (dists[a][b]**2 + dists[a][c]**2) / 2
            xi_g = xi / dists[a][m] / 2
            xis.append(xi_g)

        seccurvs.append(np.mean(xis))

    return seccurvs


# parallalize over nodes ``m``
pool = multiprocessing.Pool(args.n_cpus)
seccurvs = pool.map(sectional_curvatures, range(n_nodes))
seccurvs = list(itertools.chain(*seccurvs))

# save them
np.save(out_file, seccurvs)
