import argparse
import logging
import os
import sys

import networkx as nx
import numpy as np

import sage.all
from sage.graphs.graph import Graph
import sage.graphs.hyperbolicity as sage_hyp

from utils import np_output_filename

parser = argparse.ArgumentParser(description='Compute delta-hyperbolicities')
parser.add_argument('--input', type=str, required=True, help='The input graph.')
parser.add_argument('--force', action='store_true', help='Re-generate them.')
parser.add_argument(
        '--min_num_nodes',
        type=int,
        required=True,
        help='The minimum number of nodes in the largest connected '
        'component to keep.')
parser.add_argument(
        '--sampling_size',
        type=int,
        default=int(1e8),
        help='The sampling size used in the hyperbolicity distribution.')
parser.add_argument(
        '--inherit_filename',
        action='store_true',
        help='Whether the file format of the degrees file should be inheritted '
        'from the input.')
args = parser.parse_args()

values_file = np_output_filename(args.input, 'hyp-values',
                                 args.inherit_filename)
counts_file = np_output_filename(args.input, 'hyp-counts',
                                 args.inherit_filename)
if os.path.isfile(values_file) and os.path.isfile(counts_file) and \
        not args.force:
    logging.warning('The hyperbolicities already exist: %s', values_file)
    sys.exit(0)

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

sage_graph = Graph(g)
h_dict = sage_hyp.hyperbolicity_distribution(
        sage_graph, sampling_size=args.sampling_size)
values = np.asarray([float(h) for h, _ in h_dict.items()])
counts = np.asarray([int(c * args.sampling_size) for _, c in h_dict.items()])

np.save(values_file, values)
np.save(counts_file, counts)
