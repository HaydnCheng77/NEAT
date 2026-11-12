"""Compile a benchmark suite with the baseline ZAP.

    python3 baselines/run_zap.py [suite] [--animation]

``suite`` is a folder under benchmark/ (default ``qasmbench``; e.g.
``scalability/graphs``). ZAP compiles on a fixed array from
baselines/zap/architectures/, is simulated under the paper's fidelity model,
and logs to ``results/zap/<suite>/log/<circuit>.json``. Unlike run_zac.py /
run_mqt.py / run_powermove.py this needs no conversion: ZAP is the code NEAT
was forked from, so it reads circuits and writes logs exactly as NEAT does.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "baselines")]
from zap.zap import Zap                                         # noqa: E402
from benchmark.suites import circuits                           # noqa: E402

# ZAP's default scheduling strategy.
SCHEDULING = "asap_separate"
ARCH_DIR = os.path.join(ROOT, "baselines", "zap", "architectures")

# The configuration ZAP ran with in the paper, per suite: its fixed array, and
# its placer's ``initial_mapping_parallel_lookahead`` (the number of initial
# stages whose AOD conflicts the initial mapping counts). The lookahead is 0 on
# qasmbench and graphs and ZAP's built-in default of 3 on the other three.
CONFIG = {
    "qasmbench":          ("default.json", 0),
    "scalability/graphs": ("scale_to_500_fixed.json", 0),
    "scalability/random": ("scale_to_500_fixed.json", 3),
    "scalability/adder":  ("scale_to_500_fixed.json", 3),
    "scalability/qft":    ("scale_to_500_fixed.json", 3),
}


def architecture_for(suite):
    """The fixed array ZAP compiles ``suite`` on, with its placer configuration."""
    name, lookahead = CONFIG.get(suite, ("scale_to_500_fixed.json", 3))
    with open(os.path.join(ARCH_DIR, name)) as f:
        arch = json.load(f)
    arch.setdefault("routing", {})["initial_mapping_parallel_lookahead"] = lookahead
    return arch


def main():
    parser = argparse.ArgumentParser(description="Compile a benchmark suite with the baseline ZAP.")
    parser.add_argument("suite", nargs="?", default="qasmbench",
                        help="folder under benchmark/, e.g. qasmbench (default), scalability/graphs")
    parser.add_argument("--animation", action="store_true",
                        help="also render MP4 animations (needs ffmpeg)")
    args = parser.parse_args()

    os.chdir(ROOT)      # benchmark/ and results/ are relative to the repository root
    print(f"[RUN] ZAP -> results/zap/{args.suite}/")
    for benchmark in circuits(args.suite):
        Zap(benchmark=benchmark,
            architecture=architecture_for(args.suite),
            initial_mapping=[],
            output_dir=f"zap/{args.suite}",
            scheduling_strategy=SCHEDULING,
            placement_strategy="baseline",
            routing_strategy="baseline",
            ).solve(True, args.animation)


if __name__ == "__main__":
    main()
