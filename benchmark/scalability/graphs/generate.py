"""Generate the 3-regular graph-state family (rebuilt 2026-11-09).

graphs_<n>_4.qasm: networkx.random_regular_graph(3, n, seed=1000*n + 4), one cz per
edge, written for each vertex u in node order over its neighbours v > u in
adjacency order. Reproduces every graphs_<n>_4.qasm in this folder byte for byte.
n must be even (3n even), so the grid's 75, 125 and 175 use 74, 124 and 174.

    python3 benchmark/scalability/graphs/generate.py 124 174
"""
import os
import sys

import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))


def circuit(n, seed):
    G = nx.random_regular_graph(3, n, seed=seed)
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];"]
    for u in G.nodes:
        for v in G[u]:
            if v > u:
                lines.append(f"cz q[{u}],q[{v}];")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    for n in map(int, sys.argv[1:]):
        path = os.path.join(HERE, f"graphs_{n}_4.qasm")
        with open(path, "w") as f:
            f.write(circuit(n, 1000 * n + 4))
        print("wrote", path)
