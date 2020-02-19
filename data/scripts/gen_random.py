import argparse
import random
import networkx as nx

parser = argparse.ArgumentParser(description='Generate random graph.')
parser.add_argument(
        '--nodes', type=int, required=True, help='The number nodes.')
parser.add_argument(
        '--p', type=float, required=True, help='The link probability.')
args = parser.parse_args()

g = nx.Graph()
for u in range(args.nodes):
    for v in range(u + 1, args.nodes):
        if random.random() < args.p:
            g.add_edge(u, v)
assert nx.number_connected_components(g) == 1

p_str = '{:.4f}'.format(args.p).replace('.', 'p')
name = 'random{}_{}.edges.gz'.format(g.number_of_nodes(), p_str)
nx.write_edgelist(g, name)
