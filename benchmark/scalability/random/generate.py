"""Generate the random-circuit scalability family.

Structure-free control for the scalability study. The other families in
``benchmark/scalability/`` all have strong locality or regularity that a placer
can exploit — QFT is all-to-all but in a fixed triangular order, Ising is
nearest-neighbour, adder/cat are near-linear. A random circuit has none of it:
each layer is a fresh uniform perfect matching, so the interaction graph is
dense and unstructured and every stage genuinely needs long-range rearrangement.
That makes it the honest stress test of how the compiler scales when there is no
structure left to find.

Per size ``n``: ``DEPTH`` layers, each a uniformly random perfect matching of
the qubits emitted as CZ gates, separated by a random single-qubit rotation on
every qubit. The 2q gate count is therefore ``DEPTH * floor(n/2)`` — linear in
n, so fidelity-vs-size curves compare like with like.

Seeded per (size, replicate) as ``seed = 1000*n + replicate`` (replicate 0
keeps the original ``seed = n`` for backward compatibility, since
``1000*n + 0 == n`` only when the offset is folded in below), so the family
is reproducible: rerunning this script rewrites byte-identical QASM.

REPLICATES: added 2026-09-XX (5 seeds/size, for error bars), REMOVED
2026-09-14 -- back to ONE circuit per size. Replicate-0 files (``random_n<N>.qasm``, seed=n) are unaffected;
the ``_s1..s4`` files this used to also generate are no longer written by
this script (existing ones on disk are stale and should be deleted
separately, this script does not delete files it doesn't write).

SIZES (2026-09-14): also redone to match qft/adder/graphs' shared grid --
10, 30, 50, 75(*), 100, 150, 200, ..., 1000 (uniform +50 from 150 onward).
(*) no special-cased exception needed here (unlike graphs' n=75, a random
circuit has no regularity constraint on n), so 75 is used as given.
"""

import os
import random

SIZES = [10, 30, 50, 75, 100, 125, 150, 175] + list(range(200, 1001, 50))   # 125, 175 added 2026-11-09
DEPTH = 10
HERE = os.path.dirname(os.path.abspath(__file__))


def circuit(n, depth, rng):
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];"]
    for q in range(n):
        lines.append(f"h q[{q}];")
    for _layer in range(depth):
        order = list(range(n))
        rng.shuffle(order)
        for i in range(0, n - 1, 2):
            lines.append(f"cz q[{order[i]}],q[{order[i + 1]}];")
        for q in range(n):
            lines.append(f"rz({rng.uniform(0, 6.283185307):.6f}) q[{q}];")
            lines.append(f"rx(1.570796) q[{q}];")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    for n in SIZES:
        # seed = n, matching every existing random_n<N>.qasm file on disk --
        # unchanged since before replicates existed, so this stays
        # byte-identical for every size that was already generated.
        path = os.path.join(HERE, f"random_n{n}.qasm")
        with open(path, "w") as f:
            f.write(circuit(n, DEPTH, random.Random(n)))
        print(f"wrote {path}  ({DEPTH * (n // 2)} CZ gates)")
