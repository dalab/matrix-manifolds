import sys

import networkx as nx


def main():
    n = 10  # number of nodes in the grid in each dimension
    d = 2  # the number of dimensions of the grid
    r = 2  # branching factor of each tree
    h = 3  # depth of each tree

    g = nx.convert_node_labels_to_integers(nx.grid_graph(dim=[n] * d))
    print(g.number_of_nodes(), g.number_of_edges())

    out_g = g.copy()

    for u in g.nodes(data=False):
        next_u = out_g.number_of_nodes()
        tree = nx.balanced_tree(r, h)
        tree = nx.relabel_nodes(tree, {u: next_u + u for u in tree.nodes()})
        out_g.add_edge(u, next_u)
        out_g.add_edges_from(tree.edges())

    print(out_g.number_of_nodes(), out_g.number_of_edges())
    nx.write_edgelist(out_g, 'gtree_n{}_d{}_r{}_h{}.edges'.format(n, d, r, h))


if __name__ == '__main__':
    sys.exit(main())
