import sys

import networkx as nx


def main():
    n = 10  # number of nodes in the grid in each dimension
    d = 3  # the number of dimensions of the grid
    g = nx.convert_node_labels_to_integers(nx.grid_graph(dim=[n] * d))
    print(g.number_of_nodes(), g.number_of_edges())

    nx.write_edgelist(g, 'grid_n{}_d{}.edges'.format(n, d))


if __name__ == '__main__':
    sys.exit(main())
