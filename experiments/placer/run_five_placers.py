"""Five compilers' placers on the same stage, routed by NEAT's router (2026-10-29).

    python3 experiments/placer/run_five_placers.py <suite> <circuit> [N_STAGES]

Measurement only; the shipped compiler is untouched. Builds on
run_stage_candidates.py (imported: same forks, adapters and cost hooks).

Part 1, per stage. Up to N_STAGES (default 10; the paper uses 100) stages with two-qubit gates,
equally spaced from the first to the last. At each, from NEAT's layout at that
stage (every earlier stage placed by NEAT), the stage is placed by each of
  neat   NEAT's placer (shipped)
  zap    baseline ZAP's Placer (baselines/zap/placer.py, unchanged)
  zac    ZAC's matchings (see run_stage_candidates.py)
  pm     PowerMove's anchor-and-join rule (see run_stage_candidates.py)
  qmap   MQT QMAP's HeuristicPlacer (mqt-qmap 3.8.0, github.com/munich-quantum-toolkit/qmap, the
         RoutingAwareCompiler default): its cost is the sum over AOD
         compatibility groups of sqrt(longest move in the group), a move
         joining the first group it is compatible with; idle atoms in the zone
         go to storage (atom jobs), then gates go to Rydberg pairs (gate jobs),
         a gate with a partner already in the zone offered a reuse option with
         QMAP's reuseLevel (5) off its cost. Ported as the FIRST dive of its
         iterative diving search (each job takes the option that raises the
         cost least, jobs hardest-first), 8 nearest options per job, no
         lookahead term; like the ZAC port, a gate may not take a trap another
         gate's qubit still holds (one layout per stage cannot express QMAP's
         storage round trip)
and each layout is routed by NEAT's router (only that stage forced, later
stages NEAT's). Identical layouts are merged.
Each compiler's own objective scores every layout (lower = better):
  neat   stage Delta T_eqv: waves sum(2 t_trans + t_b) + 2 alpha tau_trans
         + beta tau_idle, idle atoms in the zone charged once (this stage only)
  zap    total distance (um) + 1000 * AOD-incompatible move pairs
  zac    sum over moved atoms of sqrt(d), a gate whose atoms share a row
         costing sqrt(max(d0, d1))
  pm     PowerMove's grouping: moves by increasing distance, each into the
         first compatible group, sum over groups of t(longest move)
  qmap   QMAP's cost: sum over groups (first compatible, longest move first)
         of sqrt(longest move)
Actual cost: that stage's cost after routing (and the whole circuit's), in us
(nats * T2 / n_q).

Part 2, whole placer. The circuit compiled with each placer at EVERY stage
(same starting layout, NEAT's router), scored by the Simulator: -log10 F.

Writes experiments/placer/data/<suite>/<circuit>.json.
"""
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv, _argv = sys.argv[:1], sys.argv
import run_stage_candidates as M                                  # noqa: E402
sys.argv = _argv
from NEAT.architecture import build_architecture              # noqa: E402
from benchmark.suites import n_qubits as _qasm_n_qubits             # noqa: E402
import run_zap                                                     # noqa: E402
from NEAT.simulator import Simulator                     # noqa: E402

OUT = os.path.join(ROOT, "experiments", "placer", os.environ.get("FP_OUT", "data"))
# FP_WHOLE_FROM: copy the whole-circuit block from this data dir instead of
# recompiling it (it does not depend on N_STAGES).
WHOLE_FROM = os.environ.get("FP_WHOLE_FROM")
PLACERS = {"neat": "shipped", "zap": "zap_placer", "zac": "zac_placer",
           "pm": "pm_placer", "qmap": "qmap_placer"}
STAGE_LIMIT = int(os.environ.get("FP_STAGE_LIMIT", "300"))
WHOLE_LIMIT = int(os.environ.get("FP_WHOLE_LIMIT", "1800"))
K_OPT = 8
REUSE_LEVEL = 5.0


# --- QMAP port -------------------------------------------------------------

class _Groups:
    """QMAP's compatibility groups: a move joins the first compatible group."""

    def __init__(self, compatible):
        self.g, self.compatible = [], compatible

    def delta(self, vecs):
        """Cost added by ``vecs`` (sum over groups of sqrt(longest move))."""
        trial = [[list(v), d] for v, d in self.g]
        before = sum(math.sqrt(d) for _, d in trial)
        for v in vecs:
            self._add(trial, v)
        return sum(math.sqrt(d) for _, d in trial) - before

    def add(self, vecs):
        for v in vecs:
            self._add(self.g, v)

    def _add(self, groups, v):
        d = math.dist((v[0], v[2]), (v[1], v[3]))
        for grp in groups:
            if all(self.compatible(v, u) for u in grp[0]):
                grp[0].append(v)
                grp[1] = max(grp[1], d)
                return
        groups.append([[v], d])


def _qmap_place(self, i, start):
    out = list(start)
    vec = lambda q, t: None if out[q] == t else (out[q][0], t[0], out[q][1], t[1])
    groups = _Groups(self._compatible_2d)
    gates = [tuple(g) for g in self.list_full_gates[i] if g[0] != g[1]]
    gq = {q for g in gates for q in g}
    # atom jobs: idle atoms in the zone go to storage
    taken = set(start)
    jobs = []
    for q in (q for q in range(self.n_q) if start[q] in self._ent_set and q not in gq):
        opts = sorted((s for s in self.stg_slm_sites if s not in taken),
                      key=lambda s: math.dist(start[q], s))[:K_OPT]
        jobs.append((q, opts))
    jobs.sort(key=lambda j: -math.dist(start[j[0]], j[1][0]) if j[1] else 0)
    for q, opts in jobs:
        opts = [s for s in opts if s not in taken]
        if not opts:
            opts = sorted((s for s in self.stg_slm_sites if s not in taken),
                          key=lambda s: math.dist(start[q], s))[:K_OPT]
        best = min(opts, key=lambda s: (groups.delta([vec(q, s)]), math.dist(start[q], s)))
        groups.add([vec(q, best)])
        taken.discard(out[q])
        taken.add(best)
        out[q] = best
    # gate jobs
    held = {out[q] for q in gq}
    free = lambda t, own: (t not in taken or t in own) and (t not in held or t in own)
    gjobs = []
    for q0, q1 in gates:
        own = {out[q0], out[q1]}
        mid = ((out[q0][0] + out[q1][0]) / 2, (out[q0][1] + out[q1][1]) / 2)
        cands = sorted((p for p in self.all_pairs if free(p[0], own) and free(p[1], own)),
                       key=lambda p: math.dist(mid, p[0]) + math.dist(mid, p[1]))[:K_OPT]
        gjobs.append(((q0, q1), cands))
    gjobs.sort(key=lambda j: -(math.dist(out[j[0][0]], j[1][0][0]) + math.dist(out[j[0][1]], j[1][0][1]))
               if j[1] else 0)
    for (q0, q1), cands in gjobs:
        own = {out[q0], out[q1]}
        opts = []
        for keep, other in ((q0, q1), (q1, q0)):           # reuse options
            t = out[keep]
            if t in self._ent_set:
                for u in self._twin_map.get(t, ()):
                    if free(u, own):
                        opts.append(({other: u}, -REUSE_LEVEL))
        for p in cands:
            if not (free(p[0], own) and free(p[1], own)):
                continue
            l, r = (p[0], p[1]) if p[0] <= p[1] else (p[1], p[0])
            qa, qb = (q0, q1) if out[q0] <= out[q1] else (q1, q0)
            opts.append(({qa: l, qb: r}, 0.0))
        if not opts:                                       # fall back: any free pair
            p = min((p for p in self.all_pairs if free(p[0], own) and free(p[1], own)),
                    key=lambda p: math.dist(out[q0], p[0]) + math.dist(out[q1], p[1]))
            opts = [({q0: p[0], q1: p[1]}, 0.0)]

        def score(o):
            vs = [v for v in (vec(q, t) for q, t in o[0].items()) if v is not None]
            return groups.delta(vs) + o[1]
        mv, _ = min(opts, key=score)
        groups.add([v for v in (vec(q, t) for q, t in mv.items()) if v is not None])
        for q, t in mv.items():
            taken.discard(out[q])
            taken.add(t)
            out[q] = t
    return out


M.FOREIGN = M.FOREIGN + ("qmap_placer",)
M.PLACE_FNS["qmap_placer"] = _qmap_place
# Part 2 forces every stage: S["force"] == "all"
M._active = lambda: M.S["force"] is not None and (M.S["force"] == "all" or M.S["stage"] == M.S["force"])


# --- each compiler's objective on a finished layout -------------------------

def _zac_cost(placer, start, out):
    """Sum over moved atoms of sqrt(d), a gate whose atoms share a row costing
    sqrt(max(d0, d1)) (vmplacer.place_gate)."""
    gates = [g for g in placer.list_full_gates[M.S["stage"]] if g[0] != g[1]]
    gq = {q for g in gates for q in g}
    d = lambda q: math.dist(start[q], out[q])
    cost = sum(math.sqrt(d(q)) for q in range(placer.n_q) if q not in gq)
    for a, b in gates:
        cost += math.sqrt(max(d(a), d(b))) if start[a][1] == start[b][1] else math.sqrt(d(a)) + math.sqrt(d(b))
    return cost


def objectives(placer, start, out):
    vecs = [(start[q][0], out[q][0], start[q][1], out[q][1])
            for q in range(placer.n_q) if start[q] != out[q]]
    comp = placer._compatible_2d
    d = lambda v: math.dist((v[0], v[2]), (v[1], v[3]))
    conf = sum(not comp(vecs[a], vecs[b]) for a in range(len(vecs)) for b in range(a + 1, len(vecs)))
    pm = []
    for v in sorted(vecs, key=d):
        for grp in pm:
            if all(comp(v, u) for u in grp[0]):
                grp[0].append(v)
                grp[1] = max(grp[1], d(v))
                break
        else:
            pm.append([[v], d(v)])
    qm = _Groups(comp)
    qm.add(sorted(vecs, key=lambda v: -d(v)))
    return {"neat": M.S["pred"]["teqv_stage_c"],
            "zap": sum(map(d, vecs)) + 1000.0 * conf,
            "zac": _zac_cost(placer, start, out),
            "pm": sum(placer.movement_duration(g[1]) for g in pm),
            "qmap": sum(math.sqrt(g[1]) for g in qm.g)}


def _capture(self, i):
    out = M._placing_rec(self, i)
    if M._active() and M.S["force"] != "all":
        M.S["objs"] = objectives(self, self._stage_start_mapping, out)
    return out


M.Placer.placing = _capture


def run_fork(b, a, force, pol, limit):
    M.signal.alarm(limit)
    try:
        return M.compile_once(b, a, force=force, pol=pol)
    finally:
        M.signal.alarm(0)


def whole(b, a, pol):
    """-log10 F of the circuit with ``pol`` placing every stage (Simulator)."""
    M.signal.alarm(WHOLE_LIMIT)
    try:
        M.S.update(stage=-1, force="all" if pol != "shipped" else None, pol=pol,
                   rng=M.random.Random(0), costs=[], pred=None, sig=None, decisions=0,
                   zp=None, arch=a, detail=[])
        z = M.Zap(benchmark=b, architecture=a, initial_mapping=[], output_dir="")
        t0 = time.time()
        rc, _ = z._compile_once("alap_separate", a)
        t = time.time() - t0
    finally:
        M.signal.alarm(0)
    z._merge_transfers(rc)
    ds = float((a.get('routing') or {}).get('min_tone_sep', 2.0) or 0.0)
    legal = M.Verifier(results_code=rc, traps=list(z.stg_slm_sites) + list(z.ent_slm_sites), d_s=ds).legal
    sim = Simulator(results_code=rc, architecture=a)
    lnf = 0.0
    for f in (sim.cir_fidelity_1q_gate, sim.cir_fidelity_2q_gate,
              sim.cir_fidelity_2q_gate_for_idle, sim.cir_fidelity_atom_transfer):
        lnf += math.log(f) if f > 0 else -math.inf
    lnf -= sim.log_fidelity_coherence
    return {"neglog10F": -lnf / math.log(10), "legal": legal, "compile_s": t,
            "duration_us": sim.total_duration}


def main():
    suite, circ = sys.argv[1], sys.argv[2]
    n_stages = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    out_dir = os.path.join(OUT, suite.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{circ}.json")
    if os.path.exists(out):
        return
    b = f"{suite}/{circ}.qasm"
    a = build_architecture(_qasm_n_qubits(b))
    a.setdefault("routing", {})["initial_mapping_parallel_lookahead"] = run_zap.CONFIG[suite][1]   # read only by ZAP's placer
    T0 = time.time()
    base, rc = run_fork(b, a, None, None, WHOLE_LIMIT)
    stages = rc["stages"]["stage"]
    n_q = rc["n_q"]
    t2 = a.get("qubit_spec", {}).get("T2", 1.5e6)
    two_q = [s for s in range(len(base)) if stages[s]["type"] != "1qGate"]
    k = min(n_stages, len(two_q))
    picked = sorted({two_q[round(j * (len(two_q) - 1) / max(1, k - 1))] for j in range(k)})
    recs, failed = [], []
    for s in picked:
        lay, seen = [], {}
        for name, pol in PLACERS.items():
            try:
                costs, crc = run_fork(b, a, s, pol, STAGE_LIMIT)
            except Exception as e:
                failed.append({"stage": s, "placer": name, "error": repr(e)[:200]})
                print(json.dumps({"circuit": circ, **failed[-1]}), flush=True)
                continue
            sig = M.S["sig"]
            if sig in seen:
                lay[seen[sig]]["placers"].append(name)
                continue
            seen[sig] = len(lay)
            lay.append({"placers": [name], "objs": M.S["objs"], "pred": M.S["pred"],
                        "stage_us": costs[s] * t2 / n_q, "total_us": sum(costs) * t2 / n_q,
                        "aod_legal": crc["verification"]["aod_legal"]})
        recs.append({"stage": s, "layouts": lay})
        if os.environ.get("FP_PROGRESS"):                # one line per finished stage (2026-11-07)
            with open(os.environ["FP_PROGRESS"], "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {circ} stage {len(recs)}/{len(picked)}"
                        f" ({time.time() - T0:.0f} s)\n")
    whole_res = {}
    if WHOLE_FROM:
        src = os.path.join(ROOT, "experiments", "placer", WHOLE_FROM, suite.replace("/", "_"), f"{circ}.json")
        if os.path.exists(src):
            whole_res = json.load(open(src))["whole"]
    # FP_SKIP_WHOLE=1 (2026-11-07): skip Part 2 (the placement table reads only Part 1).
    for name, pol in ([] if os.environ.get("FP_SKIP_WHOLE") == "1" else PLACERS.items()):
        if name in whole_res:
            continue
        try:
            whole_res[name] = whole(b, a, pol)
        except Exception as e:
            whole_res[name] = {"error": repr(e)[:200]}
            print(json.dumps({"circuit": circ, "whole": name, "error": repr(e)[:200]}), flush=True)
    rec = {"suite": suite, "circuit": circ, "n_q": n_q, "n_2q_stages": len(two_q),
           "stages": recs, "failed": failed, "whole": whole_res}
    with open(out + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(out + ".tmp", out)
    print(json.dumps({"circuit": circ, "stages": len(recs),
                      "whole": {k: round(v.get("neglog10F", float("nan")), 4) for k, v in whole_res.items()}}),
          flush=True)


if __name__ == "__main__":
    main()
