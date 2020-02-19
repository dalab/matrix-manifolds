import argparse
import logging
import os
import sys

import networkx as nx


def main():
    args = parse_args()
    g = nx.convert_node_labels_to_integers(nx.read_edgelist(args.input_graph))
    edges = list(g.selfloop_edges())
    if edges:
        logging.warning('Removing %d edges from %s', len(edges),
                        args.input_graph)
        g.remove_edges_from(edges)
        nx.write_edgelist(g, args.input_graph)


def parse_args():
    parser = argparse.ArgumentParser(description='Remove self edges')
    parser.add_argument(
            '--input_graph', type=str, required=True, help='The input graph.')
    return parser.parse_args()


if __name__ == '__main__':
    sys.exit(main())
