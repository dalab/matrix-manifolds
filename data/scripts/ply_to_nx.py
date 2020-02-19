import argparse
import os

import networkx as nx
from plyfile import PlyData

parser = argparse.ArgumentParser(description='Transform a PLY file to a graph.')
parser.add_argument(
        '--input', type=str, required=True, help='The input PLY file.')
args = parser.parse_args()

ply = PlyData.read(args.input)
g = nx.Graph()
for face in ply['face'].data['vertex_indices']:
    for i in range(len(face) - 1):
        g.add_edge(face[i], face[i + 1])
g = g.subgraph(max(nx.connected_components(g), key=len))
g = nx.convert_node_labels_to_integers(g)

print(g.number_of_nodes(), g.number_of_edges())
filename = os.path.splitext(os.path.basename(args.input))[0] + '.edges.gz'
nx.write_edgelist(g, filename)
