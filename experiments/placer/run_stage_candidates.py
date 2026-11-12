"""Whole-stage candidate placements: predicted vs routed cost (2026-10-29).

    [T_TRANS=<us>] python3 experiments/placer/run_stage_candidates.py <suite> <circuit> [N_STAGES]

Measurement only. For up to N_STAGES stages spread over the circuit, several
complete placements of that one stage are generated -- every pair-selection
and every evacuation decision of the stage -- each starting from the shipped
layout (all earlier stages shipped). Candidates:
  shipped    Delta T_eqv for pairs, shipped evacuation rule
  zap_pairs  pairs by 1000 * conflicts + distance, shipped evacuation
  dist_pairs pairs by distance, shipped evacuation
  all_stay   shipped pairs, no idle atom leaves
  all_leave  shipped pairs, every idle atom with a free storage site leaves
  rand<i>    each pair decision a uniform pick from the shortlist with prob.
             0.5, each evacuation decision flipped with prob. 0.3 and its
             destination a uniform pick from the 4 best sites with prob. 0.3
  zap_placer baseline ZAP's whole Placer (baselines/zap/placer.py, unchanged)
             run on the stage from the same start layout and home traps
  zac_placer ZAC's two matchings (baselines/zac/zac/placer/vmplacer.py) on this
             architecture: every idle atom in the zone back to storage by a
             min-weight matching on sqrt(d); then gates to free Rydberg pairs by
             a min-weight matching on sqrt(d0) + sqrt(d1) (sqrt(max) for a
             same-row pair), left trap to the qubit further left. Matched over
             all free sites rather than ZAC's search window (the window is a
             speed limit, not part of its objective). ZAC's reuse: a gate
             with a partner already on an entanglement trap whose twin is
             free is offered only that pair, the partner keeping its trap
             (ZAC offers a reused qubit only its current site). Without this
             the joint matching can send two zone atoms to each other's
             traps, a move cycle NEAT's router cannot route (2026-10-29).
             Also, a gate may not take a trap still held by another gate's
             qubit: in ZAC every non-reused atom has gone back to storage
             before the gates are placed, so it never lands on an occupied
             trap. One layout per stage cannot express that round trip, so
             this is the closest rule (stricter than ZAC: traps those atoms
             vacate this stage are not offered either); if it leaves no
             full matching, the unrestricted matching is used
  pm_placer  PowerMove's stage rule (baselines/powermove/mvqc.py) on this
             architecture: every idle atom in the zone drops to the nearest
             free storage trap in its own column (nearest column if the
             column is full); for each gate in circuit order, a partner
             already in the zone stays put and the other moves to its twin
             trap (q1 stays if both are in the zone); a gate with both
             partners in storage takes the free pair nearest to both
Foreign candidates are routed by NEAT's router like the others (SC_POL picks
the candidate list, SC_OUT the output folder).
Each candidate is compiled with ONLY that stage forced (later stages placed by
the shipped placer); the compiler is deterministic, so each fork is exact.

Predicted cost of the stage's final placement, per objective:
  dist   total move distance (um)
  dconf  total distance + 1000 * AOD-incompatible move pairs
  teqv   sum over the placer's waves of (2 t_trans + t_b) + 2 alpha tau_trans
         + sum over idle atoms left in the zone of k_q tau_idle (us)
  teqv_stage  the paper's stage T_eqv: waves + 2 alpha tau_trans + beta tau_idle
         (this stage's exposure only)
  *_c    the same with the waves re-planned from the final vectors alone
         (longest move first, least-increase), so every candidate -- also a
         foreign placer's -- is priced by one wave rule
Actual cost (nats, the Simulator's charge): 'stage' = that stage, including its
idle crosstalk; 'total' = the whole circuit.
Writes experiments/placer/stage_candidates[_t<T>]/<suite>/<circuit>.json.
"""
import json
import math
import os
import random
import signal
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "baselines"))   # for `import zap`
os.chdir(ROOT)

from NEAT.architecture import build_architecture            # noqa: E402
from benchmark.suites import n_qubits as _qasm_n_qubits             # noqa: E402
import run_zap                                                     # noqa: E402
from NEAT.neat import Neat as Zap                                  # noqa: E402
from NEAT.placer import Placer                            # noqa: E402
from NEAT import router as R                               # noqa: E402
from NEAT.transport import compatible_2D                  # noqa: E402
from zap.placer import Placer as ZapPlacer                 # noqa: E402
from NEAT.verifier import Verifier                        # noqa: E402

T_TRANS = os.environ.get("T_TRANS")
OUT = os.path.join(ROOT, "experiments", "placer",
                   os.environ.get("SC_OUT", "stage_candidates") + (f"_t{int(float(T_TRANS))}" if T_TRANS else ""))
POLICIES = os.environ.get("SC_POL", ",".join(
    ["shipped", "zap_pairs", "dist_pairs", "all_stay", "all_leave"] + [f"rand{i}" for i in range(5)])).split(",")
# A forked compile that runs past this is recorded as failed: NEAT's router can
# loop without end on a foreign layout (seen: zac_placer, dnn_n33 stage 59).
FORK_LIMIT = int(os.environ.get("SC_FORK_LIMIT", "60"))


def _timeout(*_):
    raise TimeoutError(f"forked compile exceeded {FORK_LIMIT} s")


signal.signal(signal.SIGALRM, _timeout)
FOREIGN = ("zap_placer", "zac_placer", "pm_placer")
S = {"stage": -1, "force": None, "pol": None, "rng": None, "costs": None, "n0": 0,
     "pred": None, "sig": None, "decisions": None, "zp": None}


def _active():
    return S["force"] is not None and S["stage"] == S["force"]


def _place_gate(self, q0, q1):
    opts = self._pair_options(q0, q1)
    if not opts:
        return False
    pol = S["pol"] if _active() else "shipped"
    base_batches, base_movers = self._stage_base_plan({q0, q1})
    base_vecs = [v for b in base_batches for v in b[0]]
    start = self._stage_start_mapping
    rows = []
    for moves in opts:
        mm = dict(moves)
        targets = {q0: mm.get(q0, self.current_mapping[q0]),
                   q1: mm.get(q1, self.current_mapping[q1])}
        distance = sum(math.dist(self.current_mapping[q], t) for q, t in moves)
        to_storage = sum(self._site_storage_dist(s) for s in targets.values())
        teqv = self._score_moves(targets, [[list(b[0]), b[1]] for b in base_batches], base_movers)
        new = [(start[q][0], targets[q][0], start[q][1], targets[q][1])
               for q in (q0, q1) if start[q] != targets[q]]
        conf = sum(not compatible_2D(v, u) for v in new for u in base_vecs)
        if len(new) == 2:
            conf += not compatible_2D(new[0], new[1])
        rows.append((teqv, distance, to_storage, conf))
    idx = range(len(opts))
    if pol == "zap_pairs":
        pick = min(idx, key=lambda i: (1000.0 * rows[i][3] + rows[i][1], rows[i][1], rows[i][2]))
    elif pol == "dist_pairs":
        pick = min(idx, key=lambda i: (rows[i][1], rows[i][2]))
    elif pol.startswith("rand") and S["rng"].random() < 0.5:
        pick = S["rng"].randrange(len(opts))
    else:
        pick = min(idx, key=lambda i: rows[i][:3])
    if _active() and len(opts) >= 2:
        S["decisions"] += 1
    for q, t in opts[pick]:
        self._commit_move(q, t)
    return True


def _shortlist(self, q):
    taken = {p for i, p in enumerate(self.current_mapping) if i != q}
    qx, qy = self.current_mapping[q]
    cell = self._stg_grid_cell_size
    gx, gy = int(qx // cell), int(qy // cell)
    radius = self._stg_grid_search_radius
    while True:
        free = [s for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)
                for s in self._stg_grid.get((gx + dx, gy + dy), ()) if s not in taken]
        if free or radius >= 12:
            break
        radius += 1
    batches = self._planned_batches
    free.sort(key=lambda s: self._move_rank(q, s, batches))
    return free[:4]


def _move_unentangled_qubits(self):
    """The shipped rule (r_q = 1), with this stage's policy applied when forced."""
    self._evacuated_this_stage = []
    num_stages = self.results_code['stages']['num_stage']
    i = self.index_list_gate
    pol = S["pol"] if _active() else "shipped"
    f_idle = -math.log(self.fidelity_2q_idle)
    f_tr = -math.log(self.fidelity_atom_transfer)
    for q in sorted(range(self.n_q), key=lambda q: -self._urgency(q)):
        if not (self.current_mapping[q] in self._ent_set
                and self.qs_status[q][i]['status'] != "2qGate"):
            continue
        nxt = self._next_2q_stage(q, i)
        idle_end = num_stages if nxt is None else nxt
        pre = self._prefix_2q()
        stay = (pre[idle_end] - pre[i]) * f_idle
        cands = _shortlist(self, q)
        if not cands:
            continue
        vec = self._vector_for_move(q, cands[0])
        leave = 2 * f_tr + self.n_q * self._marginal_wave_cost(vec, self._planned_batches) / self.time_coherence
        go, site = stay > leave, cands[0]
        if pol == "all_stay":
            go = False
        elif pol == "all_leave":
            go = True
        elif pol.startswith("rand"):
            if S["rng"].random() < 0.3:
                go = not go
            if S["rng"].random() < 0.3:
                site = S["rng"].choice(cands)
        if _active():
            S["decisions"] += 1
        if go:
            self._commit_move(q, site)
            self._evacuated_this_stage.append(q)


def _zap_place(self, i, start):
    """Baseline ZAP's Placer on stage i from ``start`` (home traps = NEAT's)."""
    zp = S["zp"]
    if zp is None:
        zp = ZapPlacer([list(self.stg_slm_sites), list(self.ent_slm_sites)], self.results_code,
                       self.list_full_gates, list(self.initial_mapping), "baseline", S["arch"])
        S["zp"] = zp
    zp.current_mapping = list(start)
    return list(zp.placing(i))


def _zac_place(self, i, start):
    """ZAC's qubit then gate matchings (see the module docstring)."""
    out = list(start)
    gates = [tuple(g) for g in self.list_full_gates[i] if g[0] != g[1]]
    gq = {q for g in gates for q in g}
    idle = [q for q in range(self.n_q) if start[q] in self._ent_set and q not in gq]
    if idle:
        taken = set(start)
        free = [s for s in self.stg_slm_sites if s not in taken]
        cost = np.array([[math.sqrt(math.dist(start[q], s)) for s in free] for q in idle])
        for a, b in zip(*linear_sum_assignment(cost)):
            out[idle[a]] = free[b]
    if not gates:
        return out
    occ = {out[q] for q in range(self.n_q) if q not in gq}
    pairs = [(p0, p1) for p0, p1 in self.all_pairs if p0 not in occ and p1 not in occ]
    # reuse: gate -> (staying qubit, its pair); a trap is claimed by one gate only
    pinned, claimed = {}, set()
    for g, (q0, q1) in enumerate(gates):
        for keep, other in ((q0, q1), (q1, q0)):
            t = out[keep]
            if t not in self._ent_set or t in claimed:
                continue
            twin = next((u for u in self._twin_map.get(t, ())
                         if u not in occ and u not in claimed
                         and u not in {out[x] for x in gq if x not in (q0, q1)}), None)
            if twin is not None:
                pinned[g] = (keep, (t, twin))
                claimed |= {t, twin}
                break
    pairs = [p for p in pairs if p[0] not in claimed and p[1] not in claimed]
    pairs += [pr for _, pr in pinned.values()]
    big = 1e9
    held = {out[q] for q in gq}
    cost = np.empty((len(gates), len(pairs)))
    blocked = np.zeros((len(gates), len(pairs)), dtype=bool)
    for g, (q0, q1) in enumerate(gates):
        a, b = out[q0], out[q1]
        if b < a:
            a, b = b, a                    # a is the qubit further left
        own = {a, b}
        for k, (p0, p1) in enumerate(pairs):
            if g in pinned or (p0 in claimed and p1 in claimed):
                cost[g, k] = 0.0 if g in pinned and pinned[g][1] == (p0, p1) else big
                continue
            blocked[g, k] = (p0 in held and p0 not in own) or (p1 in held and p1 not in own)
            l, r = (p0, p1) if p0 <= p1 else (p1, p0)
            d0, d1 = math.dist(a, l), math.dist(b, r)
            cost[g, k] = math.sqrt(max(d0, d1)) if a[1] == b[1] else math.sqrt(d0) + math.sqrt(d1)
    strict = np.where(blocked, big, cost)
    rows, cols = linear_sum_assignment(strict)
    if strict[rows, cols].max() < big:
        cost = strict
        S["zac_strict"] = S.get("zac_strict", 0) + 1
    else:
        rows, cols = linear_sum_assignment(cost)
        S["zac_loose"] = S.get("zac_loose", 0) + 1
    used = set()
    for g, k in sorted(zip(rows, cols), key=lambda x: cost[x]):
        k_ok = k
        if pairs[k][0] in used or pairs[k][1] in used:
            # a trap shared by two pairs: take this gate's cheapest pair still free
            k_ok = min((kk for kk in range(len(pairs))
                        if pairs[kk][0] not in used and pairs[kk][1] not in used),
                       key=lambda kk: cost[g, kk])
        p0, p1 = pairs[k_ok]
        q0, q1 = gates[g]
        if g in pinned:
            keep, (t, twin) = pinned[g]
            out[q1 if keep == q0 else q0] = twin
        else:
            l, r = (p0, p1) if p0 <= p1 else (p1, p0)
            qa, qb = (q0, q1) if out[q0] <= out[q1] else (q1, q0)
            out[qa], out[qb] = l, r
        used |= {p0, p1}
    return out


def _pm_place(self, i, start):
    """PowerMove's anchor-and-join rule (see the module docstring)."""
    out = list(start)
    gates = [tuple(g) for g in self.list_full_gates[i] if g[0] != g[1]]
    gq = {q for g in gates for q in g}
    taken = set(start)
    for q in sorted((q for q in range(self.n_q) if start[q] in self._ent_set and q not in gq),
                    key=lambda q: -start[q][1]):
        x, y = start[q]
        site = min((s for s in self.stg_slm_sites if s not in taken),
                   key=lambda s: (abs(s[0] - x), abs(s[1] - y)))
        taken.discard(out[q])
        taken.add(site)
        out[q] = site
    for q0, q1 in gates:
        twin = None
        for anchor, mover in ((q1, q0), (q0, q1)):
            if out[anchor] in self._ent_set:
                twin = next((t for t in self._twin_map.get(out[anchor], ()) if t not in taken), None)
                if twin is not None:
                    break
        if twin is None:
            pair = min(((p0, p1) for p0, p1 in self.all_pairs if p0 not in taken and p1 not in taken),
                       key=lambda p: min(math.dist(out[q0], p[0]) + math.dist(out[q1], p[1]),
                                         math.dist(out[q0], p[1]) + math.dist(out[q1], p[0])))
            if math.dist(out[q0], pair[0]) + math.dist(out[q1], pair[1]) > \
                    math.dist(out[q0], pair[1]) + math.dist(out[q1], pair[0]):
                pair = pair[::-1]
            for q, t in ((q0, pair[0]), (q1, pair[1])):
                taken.discard(out[q])
                taken.add(t)
                out[q] = t
            continue
        taken.discard(out[mover])
        taken.add(twin)
        out[mover] = twin
    return out


def _canon_waves(self, vecs):
    """Waves for ``vecs`` alone: longest first, least-increase (the placer's insert)."""
    batches = []
    for v in sorted(vecs, key=lambda v: -math.dist((v[0], v[2]), (v[1], v[3]))):
        self._absorb_vector(v, batches)
    return sum(b[1] + 2 * self.time_atom_transfer for b in batches), len(batches)


PLACE_FNS = {"zap_placer": _zap_place, "zac_placer": _zac_place, "pm_placer": _pm_place}
Placer._place_gate = _place_gate
Placer._move_unentangled_qubits = _move_unentangled_qubits

_placing = Placer.placing


def _placing_rec(self, i):
    S["stage"] = i
    if _active() and S["pol"] in FOREIGN:
        start = list(self.current_mapping)
        out = PLACE_FNS[S["pol"]](self, i, start)
        assert len(set(out)) == self.n_q, "foreign placer produced a double occupancy"
        # NEAT's stage state, as placing() leaves it, for the foreign layout
        self.index_list_gate = i
        self.init_params()
        self._stage_start_mapping = list(start)
        self._occupied = set(start)
        moved = sorted((q for q in range(self.n_q) if out[q] != start[q]),
                       key=lambda q: -math.dist(start[q], out[q]))
        for q in moved:
            self._commit_move(q, out[q])
        out = self.current_mapping
    else:
        out = _placing(self, i)
    if _active():
        start = self._stage_start_mapping
        vecs = [(start[q][0], out[q][0], start[q][1], out[q][1])
                for q in range(self.n_q) if start[q] != out[q]]
        dist = sum(math.dist((v[0], v[2]), (v[1], v[3])) for v in vecs)
        conf = sum(not compatible_2D(vecs[a], vecs[b])
                   for a in range(len(vecs)) for b in range(a + 1, len(vecs)))
        tau_tr = -math.log(self.fidelity_atom_transfer) * self.time_coherence / self.n_q
        tau_idle = -math.log(self.fidelity_2q_idle) * self.time_coherence / self.n_q
        gating = {q for q in range(self.n_q) if self.qs_status[q][i]['status'] == "2qGate"}
        pre = self._prefix_2q()
        num_stages = self.results_code['stages']['num_stage']
        idle, beta = 0.0, 0
        for q in range(self.n_q):
            if out[q] in self._ent_set and q not in gating:
                nxt = self._next_2q_stage(q, i)
                idle += (pre[num_stages if nxt is None else nxt] - pre[i]) * tau_idle
                beta += 1
        waves = sum(b[1] + 2 * self.time_atom_transfer for b in self._planned_batches)
        wc, nwc = _canon_waves(self, vecs)
        S["pred_move_us"] = wc
        S["pred"] = {"dist": dist, "dconf": dist + 1000.0 * conf,
                     "teqv": waves + 2 * len(vecs) * tau_tr + idle,
                     "teqv_stage": waves + 2 * len(vecs) * tau_tr + beta * tau_idle,
                     "teqv_c": wc + 2 * len(vecs) * tau_tr + idle,
                     "teqv_stage_c": wc + 2 * len(vecs) * tau_tr + beta * tau_idle,
                     "waves_c": nwc, "move_us_c": wc,
                     "teqv_move": waves + 2 * len(vecs) * tau_tr, "beta": beta,
                     "n_moves": len(vecs), "conf": conf, "waves": len(self._planned_batches)}
        S["sig"] = tuple(map(tuple, out))
    return out


Placer.placing = _placing_rec

_rsm = R.Router._route_stage_moves


def _route_stage_moves(self):
    S["n0"] = len(self.results_code["instructions"])
    return _rsm(self)


R.Router._route_stage_moves = _route_stage_moves
_pg = R.Router.process_gate


def _process_gate(self):
    _pg(self)
    ins = self.results_code["instructions"][S["n0"]:]
    a = self.architecture.get("operation_fidelity", {})
    c = self._stage_cost(ins)
    for x in ins:
        if x["type"] == "Crosstalk":
            # default as in the Simulator: f_idle = 1 - (1 - f_2q) / 2
            c -= len(x["qs"]) * math.log(a.get("two_qubit_gate_for_idle",
                                               1 - (1 - a.get("two_qubit_gate", 0.999)) / 2))
        elif x["type"] == "2qGate":
            c -= len(x["gates"]) * math.log(a.get("two_qubit_gate", 1.0))
        elif x["type"] == "1qGate":
            c -= len(x["gates"]) * math.log(a.get("single_qubit_gate", 1.0))
    S["costs"].append(c)
    S["detail"].append({"move_us": sum(max(x.get("duration") or [0]) for x in ins
                                       if x["type"] not in ("1qGate", "2qGate", "Crosstalk")),
                        "activates": sum(x["type"] == "Activate" for x in ins)})


R.Router.process_gate = _process_gate


def compile_once(b, a, force=None, pol=None, seed=0):
    S.update(stage=-1, force=force, pol=pol, rng=random.Random(seed), costs=[],
             pred=None, sig=None, decisions=0, zp=None, arch=a, detail=[])
    z = Zap(benchmark=b, architecture=a, initial_mapping=[], output_dir="")
    rc, _ = z._compile_once("alap_separate", a)
    # legality of the forked trace (the built-in verifier, as Zap.solve runs it)
    ds = float((a.get('routing') or {}).get('min_tone_sep', 2.0) or 0.0)
    v = Verifier(results_code=rc, traps=list(z.stg_slm_sites) + list(z.ent_slm_sites), d_s=ds)
    rc["verification"] = {"aod_legal": v.legal}
    return list(S["costs"]), rc


def main():
    suite, circ = sys.argv[1], sys.argv[2]
    n_stages = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    out_dir = os.path.join(OUT, suite.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{circ}.json")
    if os.path.exists(out):
        return
    b = f"{suite}/{circ}.qasm"
    a = build_architecture(_qasm_n_qubits(b))
    a.setdefault("routing", {})["initial_mapping_parallel_lookahead"] = run_zap.CONFIG[suite][1]   # read only by ZAP's placer
    if T_TRANS:
        a.setdefault("operation_duration", {})["atom_transfer"] = float(T_TRANS)
    base, rc = compile_once(b, a)
    stages = rc["stages"]["stage"]
    n_q = rc["n_q"]
    t2 = a.get("qubit_spec", {}).get("T2", 1.5e6)
    # Stages with at least one two-qubit gate, spread evenly.
    two_q = [s for s in range(len(base)) if stages[s]["type"] != "1qGate"]
    step = max(1, len(two_q) / n_stages)
    picked = [two_q[int(j * step)] for j in range(min(n_stages, len(two_q)))]
    recs, failed = [], []
    for s in picked:
        cands, seen = [], {}
        for k, pol in enumerate(POLICIES):
            signal.alarm(FORK_LIMIT)
            try:
                costs, crc = compile_once(b, a, force=s, pol=pol, seed=1000 * s + k)
            except Exception as e:             # illegal trace (strict mode) or router stuck
                signal.alarm(0)
                failed.append({"stage": s, "policy": pol, "error": repr(e)[:200]})
                print(json.dumps({"circuit": circ, **failed[-1]}), flush=True)
                continue
            signal.alarm(0)
            if S["pred"] is None:
                continue
            sig = S["sig"]
            if sig in seen:
                cands[seen[sig]]["policies"].append(pol)
                continue
            seen[sig] = len(cands)
            cands.append({"policies": [pol], "pred": S["pred"],
                          "stage_us": costs[s] * t2 / n_q, "total_us": sum(costs) * t2 / n_q,
                          "decisions": S["decisions"],
                          "actual_move_us": S["detail"][s]["move_us"],
                          "actual_activates": S["detail"][s]["activates"],
                          "aod_legal": (crc.get("verification") or {}).get("aod_legal")})
        recs.append({"stage": s, "candidates": cands})
    rec = {"suite": suite, "circuit": circ, "n_q": n_q, "t_trans": float(T_TRANS or 15),
           "stages": recs, "failed": failed}
    with open(out + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(out + ".tmp", out)
    print(json.dumps({"circuit": circ, "stages": len(recs),
                      "mean_distinct": round(sum(len(r["candidates"]) for r in recs) / max(1, len(recs)), 2)}),
          flush=True)


if __name__ == "__main__":
    main()
