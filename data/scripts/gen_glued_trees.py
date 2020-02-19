import sys

import networkx as nx


def main():
    r = 3
    h = 4
    n = 10

    g = nx.balanced_tree(r, h)
    base_nodes = g.number_of_nodes()

    for i in range(1, n):
        n_nodes = base_nodes * i
        new_g = nx.balanced_tree(r, h)
        new_g = nx.relabel_nodes(new_g, {u: u + n_nodes for u in new_g.nodes()})

        # merge them
        g.add_edges_from(new_g.edges())
        for u in new_g.nodes():
            g.add_edge(u, u - n_nodes)

    nx.write_edgelist(g, f'combtrees{n * base_nodes}_n{n}_r{r}_h{h}.edges')


if __name__ == '__main__':
    sys.exit(main())
