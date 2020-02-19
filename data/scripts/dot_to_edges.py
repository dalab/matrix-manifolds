import argparse
import os
import sys
from subprocess import check_call

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
    if f.endswith('.dot'):
        g = nx.convert_node_labels_to_integers(nx.nx_pydot.read_dot(f))
        f = os.path.splitext(f)[0] + '.edges'  # fallthrough
        nx.write_edgelist(g, f)

    if f.endswith('.edges'):
        check_call(['gzip', '-f', f])


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
