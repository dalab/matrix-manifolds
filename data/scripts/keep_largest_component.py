import argparse
import os
import sys

import networkx as nx


def main():
    args = parse_args()

    if os.path.isdir(args.input):
        d = args.input
        for f in [os.path.join(d, f) for f in os.listdir(d)]:
            process_file(f, args)
    else:
        process_file(args.input, args)


def process_file(f, args):
    g = nx.convert_node_labels_to_integers(nx.read_edgelist(f))
    if not g.is_directed():
        g = g.subgraph(max(nx.connected_components(g), key=len))
        nx.write_edgelist(g, f)


def parse_args():
    parser = argparse.ArgumentParser(
            description='Convert .dot graphs to .edges')
    parser.add_argument(
            '--input',
            type=str,
            required=True,
            help='The .data/ directory containing .dot graphs or an '
            'individual graph file.')
    return parser.parse_args()


if __name__ == '__main__':
    sys.exit(main())
