"""The benchmark suites: each is a folder of .qasm files under benchmark/.

    from benchmark.suites import SUITES, circuits, n_qubits, transpiled

A suite's circuits are its folder's .qasm files in order of qubit count. The
paper uses all five suites below (110 circuits).

``transpiled(b)`` is the path of circuit ``b`` after the Qiskit transpile
every compiler applies (level 3, fixed seed): benchmark/transpiled/<b>, in the
same folder layout as the circuits. ZAC, MQT and PowerMove read these files;
NEAT and ZAP run the same transpile when they load a circuit. They are written
by baselines/run_zac.py (``prepare_qasm2``) and rebuilt only if missing, so
delete a circuit's transpiled file after editing the circuit.
"""

import glob
import os
import re

BENCH = os.path.dirname(os.path.abspath(__file__))
TRANSPILED = os.path.join(BENCH, "transpiled")
SUITES = ["qasmbench", "scalability/graphs", "scalability/random",
          "scalability/adder", "scalability/qft"]


def n_qubits(benchmark):
    """Qubit count from a circuit's register declarations, without transpiling.

    ``benchmark`` is a path under benchmark/ (or absolute). Handles OPENQASM 2.0
    ``qreg q[10];`` and OPENQASM 3.0 ``qubit[10] q;``; multiple quantum
    registers are summed, as Qiskit flattens them.
    """
    path = benchmark if os.path.isabs(benchmark) else os.path.join(BENCH, benchmark)
    total = 0
    with open(path) as f:
        for line in f:
            m = re.match(r"\s*qreg\s+\w+\[(\d+)\]", line)          # QASM 2.0
            if m is None:
                m = re.match(r"\s*qubit\[(\d+)\]\s+\w+", line)     # QASM 3.0
            if m:
                total += int(m.group(1))
    if not total:
        raise ValueError(f"no quantum register declaration found in {path}")
    return total


def circuits(suite):
    """Circuits of ``suite`` as paths under benchmark/ ('qasmbench/bv_n14.qasm'), by qubit count."""
    paths = glob.glob(os.path.join(BENCH, suite, "*.qasm"))
    if not paths:
        raise FileNotFoundError(f"no suite benchmark/{suite}/")
    rel = [os.path.relpath(p, BENCH).replace(os.sep, "/") for p in paths]
    return sorted(rel, key=lambda b: (n_qubits(b), b))


def transpiled(benchmark):
    """Path of the transpiled QASM 2 file of circuit ``benchmark`` (may not exist yet)."""
    return os.path.join(TRANSPILED, benchmark)
