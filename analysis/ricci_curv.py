import argparse
import logging
import multiprocessing
import os
import sys

import networkx as nx

from utils import _output_filename
from GraphRicciCurvature.OllivierRicci import ricciCurvature as ricci_curvature

parser = argparse.ArgumentParser(
        description='Compute graph Ollivier-Ricci curvatures')
parser.add_argument('--input', type=str, required=True, help='The input graph.')
parser.add_argument(
        '--alpha',
        type=float,
        default=0.999,
        help='The alpha that determines the approximation of the limit to 1.')
parser.add_argument('--force', action='store_true', help='Re-generate them.')
parser.add_argument(
        '--min_num_nodes',
        type=int,
        default=100,
        help='The minimum number of nodes in the largest connected '
        'component to keep.')
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

out_file = _output_filename(
        args.input, 'ricci', args.inherit_filename, extension='dot')
if os.path.isfile(out_file) and not args.force:
    logging.warning('The .dot Ricci graph already exist: %s', out_file)
    sys.exit(0)

# Load the graph.
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

# Compute the Ricci curvatures as shown in Yau et al.
g = ricci_curvature(
        g,
        alpha=args.alpha,
        method='OTD',
        compute_nc=True,
        verbose=False,
        proc=args.n_cpus)

# Save the .dot graph
nx.nx_pydot.write_dot(g, out_file)
