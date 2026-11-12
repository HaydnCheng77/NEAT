"""Recompute fidelity from event traces -- no recompilation.

    python3 experiments/end_to_end/rescore.py results/<compiler>/<suite>/events/<circuit>.events.json.gz
    python3 experiments/end_to_end/rescore.py --suite scalability/qft            # every compiler
    python3 experiments/end_to_end/rescore.py --suite qasmbench --recompute --set T_TR=10

Edit ``PARAMS`` (or pass --set KEY=VALUE) to change the model. By default the
durations recorded in each event are used, which reproduces the scorer every
compiler was scored with when the trace was made. ``--recompute`` re-derives
every duration from the geometry and PARAMS instead (shared timing model for
all compilers; needed when changing a duration constant or the movement law).

Fidelity model (ZAP's, applied identically to every compiler):
  F = F_1q^n_1q * F_2q^n_2q * F_idle^n_crosstalk * F_tr^n_transfers
      * exp(-n_q * T_total / T2)
(2026-10-10: every qubit decoheres for the whole circuit duration, as in
the paper's Eq. (coherence); per-qubit busy time is no longer credited.)
"""
import argparse
import glob
import gzip
import json
import math
import os
import sys

PARAMS = {
    "F_1Q": 0.9997, "F_2Q": 0.995, "F_TR": 0.999, "F_IDLE": None,  # None -> 1-(1-F_2Q)/2
    "T2": 1.5e6, "T_1Q": 52.0, "T_2Q": 0.27, "T_TR": 15.0,   # 2026-10-07: paper Table tab:params (2q was 0.36 us)
    "MOVE_A": 200.0, "MOVE_D0": 110.0,          # t = MOVE_A * sqrt(d / MOVE_D0)
}


def load(path):
    with gzip.open(path, 'rt') if path.endswith('.gz') else open(path) as f:
        return json.load(f)


def _move_t(d, P):
    return P["MOVE_A"] * math.sqrt(d / P["MOVE_D0"]) if d > 0 else 0.0


def score(trace, params=None, recompute=False):
    P = dict(PARAMS, **(params or {}))
    f_idle = P["F_IDLE"] if P["F_IDLE"] is not None else 1 - (1 - P["F_2Q"]) / 2
    n_q = len(trace['init'])
    total = 0.0
    n1 = n2 = ntr = nx = 0
    for e in trace['events']:
        t = e['t']
        if t == 'job':
            ntr += e['transfers']
            if recompute:
                continue          # its "sub" pick/move/drop steps carry the time
        elif t in ('pick', 'drop'):
            if not e.get('sub'):
                ntr += len(e['qs'])
        elif t == '1q':
            n1 += e['n']
        elif t == 'cz':
            n2 += len(e['pairs'])
        elif t == 'xtalk':
            nx += len(e['qs'])
        if not recompute:
            total += e.get('dur', 0.0)
            continue
        # shared timing model
        if t in ('pick', 'drop'):
            total += P["T_TR"] if e['qs'] else 0.0
        elif t == 'move':
            # makespan: the slowest atom
            total += max((_move_t(math.hypot(x1 - x0, y1 - y0), P)
                          for q, x0, y0, x1, y1 in e['moves']), default=0.0)
        elif t == '1q':
            total += P["T_1Q"] if e['n'] else 0.0
        elif t == 'cz':
            total += P["T_2Q"]
    coh = n_q * total / P["T2"]
    c1, c2 = P["F_1Q"] ** n1, P["F_2Q"] ** n2
    ci, ct = f_idle ** nx, P["F_TR"] ** ntr
    fc = math.exp(-coh) if coh < 700 else 0.0
    log10 = -(n1 * math.log10(P["F_1Q"]) + n2 * math.log10(P["F_2Q"]) + nx * math.log10(f_idle)
              + ntr * math.log10(P["F_TR"])) + coh / math.log(10)
    return {"n_qubits": n_q, "n_1q_gate": n1, "n_2q_gate": n2, "n_transfers": ntr,
            "n_crosstalk": nx, "total_duration": total,
            "total_fidelity": c1 * c2 * ci * ct * fc,
            "fidelity_1q_gate": c1, "fidelity_2q_gate": c2, "fidelity_idle": ci,
            "fidelity_handover": ct, "fidelity_decoherence": fc,
            "log_fidelity_decoherence": coh, "neg_log10_fidelity": log10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--suite")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="KEY=VALUE override of PARAMS")
    a = ap.parse_args()
    params = {k: float(v) for k, v in (s.split("=", 1) for s in a.set)}
    paths = list(a.paths)
    if a.suite:
        paths += sorted(glob.glob(f"results/*/{a.suite}/events/*.events.json.gz"))
    tot = {}
    for p in paths:
        r = score(load(p), params, a.recompute)
        comp = p.split(os.sep)[1] if p.startswith("results") else "?"
        name = os.path.basename(p).split(".events")[0]
        tot[comp] = tot.get(comp, 0.0) + r["neg_log10_fidelity"]
        print(f"{comp:14s} {name:18s} -log10F {r['neg_log10_fidelity']:12.4f}  F {r['total_fidelity']:.6e}")
    for c, v in sorted(tot.items()):
        print(f"TOTAL {c:14s} sum -log10F {v:.4f}")


if __name__ == "__main__":
    sys.exit(main())
