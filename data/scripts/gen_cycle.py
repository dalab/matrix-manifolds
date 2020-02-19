import argparse
import networkx as nx

parser = argparse.ArgumentParser(description='Generate cycle graph')
parser.add_argument('--n', type=int, required=True, help='The number of nodes.')
args = parser.parse_args()

g = nx.cycle_graph(args.n)
nx.write_edgelist(g, f'cycle{args.n}.edges.gz')
