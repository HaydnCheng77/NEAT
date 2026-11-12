"""Compile a benchmark suite with NEAT.

    python3 NEAT/run.py [suite] [--animation]

``suite`` is a folder under benchmark/ (default ``qasmbench``; e.g.
``scalability/graphs``). Each circuit is compiled on its own generated array
(NEAT/architecture.py), simulated (fidelity under the paper's model), checked by
NEAT's AOD verifier (NEAT/verifier.py), and logged to
``results/neat/<suite>/log/<circuit>.json``.

The baseline ZAP has its own driver, baselines/run_zap.py.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from NEAT.neat import Neat                                      # noqa: E402
from NEAT.architecture import build_architecture, validate      # noqa: E402
from benchmark.suites import circuits, n_qubits                 # noqa: E402

# NEAT's scheduler implements ALAP only.
SCHEDULING = "alap_separate"


def architecture_for(benchmark, verbose=True):
    """The array NEAT compiles ``benchmark`` (a path under benchmark/) on."""
    n_q = n_qubits(benchmark)
    arch = build_architecture(n_q)
    if verbose:
        info = validate(arch, n_q)
        print(f"[INFO] generated architecture for n_q={n_q}: "
              f"{info['storage_traps']} storage, {info['pairs']} pairs, "
              f"zone gap {info['zone_gap_um']} um")
    return arch


def main():
    parser = argparse.ArgumentParser(description="Compile a benchmark suite with NEAT.")
    parser.add_argument("suite", nargs="?", default="qasmbench",
                        help="folder under benchmark/, e.g. qasmbench (default), scalability/graphs")
    parser.add_argument("--animation", action="store_true",
                        help="also render MP4 animations (needs ffmpeg)")
    args = parser.parse_args()

    os.chdir(ROOT)      # benchmark/ and results/ are relative to the repository root
    print(f"[RUN] NEAT -> results/neat/{args.suite}/")
    for benchmark in circuits(args.suite):
        Neat(benchmark=benchmark,
             architecture=architecture_for(benchmark),
             initial_mapping=[],
             output_dir=f"neat/{args.suite}",
             scheduling_strategy=SCHEDULING,
             placement_strategy="baseline",
             routing_strategy="baseline",
             ).solve(True, args.animation)


if __name__ == "__main__":
    main()
