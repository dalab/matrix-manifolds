import argparse
import networkx as nx

parser = argparse.ArgumentParser(
        description='Generate torus or cylinder graphs.')
parser.add_argument(
        '--main_cycle_length',
        type=int,
        required=True,
        help='The length of the "main" cycle.')
parser.add_argument(
        '--secondary_cycle_length',
        type=int,
        required=True,
        help='The length of the "secondary" cycle.')
parser.add_argument(
        '--make_torus',
        action='store_true',
        help='Whether the first and last circles should be linked.')
args = parser.parse_args()

g = nx.cycle_graph(args.main_cycle_length)
for i in range(args.secondary_cycle_length - 1):
    gi = nx.cycle_graph(args.main_cycle_length)
    off = g.number_of_nodes()
    gi = nx.relabel_nodes(gi, {u: u + off for u in gi.nodes()})

    g.add_edges_from(gi.edges())
    for u in gi.nodes():
        g.add_edge(u, u - args.main_cycle_length)

# Link the first and last ones.
if args.make_torus and args.secondary_cycle_length > 2:
    n_nodes = g.number_of_nodes()
    off = n_nodes - args.main_cycle_length
    for u in range(args.main_cycle_length):
        g.add_edge(u, u + off)

filename = 'torus{}_n{}_m{}.edges.gz'.format(g.number_of_nodes(),
                                             args.main_cycle_length,
                                             args.secondary_cycle_length)
nx.write_edgelist(g, filename)
