import argparse
import networkx as nx

parser = argparse.ArgumentParser(description='Generate cycle graph')
parser.add_argument(
        '--cycle_nodes',
        type=int,
        required=True,
        help='The number of cycle nodes.')
parser.add_argument(
        '--tree_depth', type=int, required=True, help='The depth of each tree.')
parser.add_argument(
        '--tree_branching',
        type=int,
        required=True,
        help='The branching factor of each tree.')
args = parser.parse_args()

g = nx.cycle_graph(args.cycle_nodes)
for i in range(args.cycle_nodes):
    tree = nx.balanced_tree(args.tree_branching, args.tree_depth)
    off = g.number_of_nodes()
    tree = nx.relabel_nodes(
            tree, {u: i if u == 0 else u + off - 1
                   for u in tree.nodes()})
    g.add_edges_from(tree.edges())

f = 'treecycle{}_n{}_h{}_r{}.edges.gz'.format(g.number_of_nodes(),
                                              args.cycle_nodes, args.tree_depth,
                                              args.tree_branching)
nx.write_edgelist(g, f)
