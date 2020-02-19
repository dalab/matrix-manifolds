import argparse
import networkx as nx

parser = argparse.ArgumentParser(description='Generate cycle of cycles graph')
parser.add_argument(
        '--main_cycle_nodes',
        type=int,
        required=True,
        help='The number of nodes on the "main" cycle.')
parser.add_argument(
        '--secondary_cycle_nodes',
        type=int,
        required=True,
        help='The number of nodes on the "secondary" cycle.')
args = parser.parse_args()

g = nx.cycle_graph(args.main_cycle_nodes)
for i in range(args.main_cycle_nodes):
    gi = nx.cycle_graph(args.secondary_cycle_nodes)
    off = g.number_of_nodes()
    gi = nx.relabel_nodes(gi,
                          {u: i if u == 0 else u + off - 1
                           for u in gi.nodes()})
    g.add_edges_from(gi.edges())

f = 'cycle_of_cycles{}_n{}_m{}.edges.gz'.format(
        g.number_of_nodes(), args.main_cycle_nodes, args.secondary_cycle_nodes)
nx.write_edgelist(g, f)
