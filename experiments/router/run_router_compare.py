"""Routing with placement and array fixed: NEAT vs QMAP strict vs QMAP relaxed (2026-10-30).

    python3 experiments/router/run_router_compare.py <suite> <circuit> [N_SAMPLE]

Measurement only. NEAT compiles the circuit as shipped (placer, router, peephole
pass). Every stage's (start, target) mapping from NEAT's placer is then routed
three ways on NEAT's own generated array:
  neat     NEAT's router: the instructions it emitted for that stage
  strict   MQT QMAP IndependentSetRouter, method STRICT, then QMAP's
           CodeGenerator (load row/column-wise with offset moves, one main
           move, store row/column-wise)
  relaxed  same, method RELAXED (strict groups merged when they cross, cost
           in cube roots, preferSplit = 1, QMAP's default)
Ported from mqt-qmap 3.8.0 (github.com/munich-quantum-toolkit/qmap),
src/na/zoned/{layout_synthesizer/router/
IndependentSetRouter.cpp, code_generator/CodeGenerator.cpp}. Two additions,
both needed because NEAT's placements are not QMAP's:
  - an atom joins a group only if every atom of that group starts in its zone
    and ends in its zone (QMAP's generator assumes one source and one target
    zone per group; QMAP's own phases always satisfy this);
  - an atom whose target trap is the start trap of another moving atom goes in
    that atom's group or a later one (NEAT's placements chain, QMAP's do not).
The generator's zone constants on NEAT's array: storage pitch 3 x 3 um,
entanglement pitch 6 x 6 um with pair separation 3 um (the interleaved SLMs
seen as one grid). QMAP's offsets are not checked for AOD legality here; only
NEAT's trace is (built-in verifier).

Placement is identical across the three, so gate and idle-crosstalk errors are
identical too: the routing difference in -log10 F is exactly
(n_q T_transport / T2 + n_transfers * (-ln f_transfer)) / ln 10.
Per stage and router: transport time (pick-ups, drop-offs, moves; each
instruction lasts as long as its slowest atom, t(d) = 200 sqrt(d / 110) us,
15 us per pick-up or drop-off), pick-up and drop-off rounds, movement
segments, atoms moved per segment, atom transfers, groups, and total atom
travel in um (sum of Euclidean segment lengths over atoms, including offsets).
Writes experiments/router/data/{waypoints,no_waypoints}/<suite>/<circuit>.json.
"""
import json
import math
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "baselines"))   # for `import zap`
os.chdir(ROOT)

from NEAT.architecture import build_architecture            # noqa: E402
from benchmark.suites import n_qubits as _qasm_n_qubits             # noqa: E402
import run_zap                                                     # noqa: E402
from NEAT.neat import Neat as Zap                                  # noqa: E402
from NEAT.verifier import Verifier                        # noqa: E402
from NEAT import router as R                               # noqa: E402

# RC_PLACER=qmap_placer (or zap_placer, zac_placer, pm_placer): every stage is
# placed by that placer port of run_five_placers.py instead of NEAT's, from
# NEAT's starting layout; NEAT's router still emits the "neat" trace.
PLACER = os.environ.get("RC_PLACER")
if PLACER:
    sys.path.insert(0, os.path.join(ROOT, "experiments", "placer"))
    import run_five_placers as F                                  # noqa: E402

# RC_ARCH=zac (2026-10-31): ZAC's published array (Lin et al., HPCA 2025:
# 100 x 100 storage at 3 um, 7 x 20 Rydberg pairs at 12 x 10 um, 2 um within a
# pair, 10 um zone gap) instead of NEAT's generated one; 140 pairs hold every
# stage up to n = 200.
ARCH = os.environ.get("RC_ARCH")
ZAC_ARCH = {"name": "zac_published",
            "operation_duration": {"2qGate": 0.27, "1qGate": 52, "atom_transfer": 15},
            "operation_fidelity": {"two_qubit_gate": 0.995, "single_qubit_gate": 0.9997, "atom_transfer": 0.999},
            "qubit_spec": {"T2": 1.5e6}, "routing": {"parking_dist": 1},
            "storage_zones": [{"zone_id": 0, "slms": [
                {"id": 0, "site_seperation": [3, 3], "r": 100, "c": 100, "location": [0, 0]}]}],
            "entanglement_zones": [{"zone_id": 0, "slms": [
                {"id": 1, "site_seperation": [12, 10], "r": 7, "c": 20, "location": [35, 307]},
                {"id": 2, "site_seperation": [12, 10], "r": 7, "c": 20, "location": [37, 307]}]}]}
OUT = os.path.join(ROOT, "experiments", "router",
                   os.environ.get("RC_OUT", "data/" + ("no_waypoints" if os.environ.get("RC_ABLATE_WAYPOINTS") == "1"
                                                        else "waypoints")
                                  + (f"_{PLACER}" if PLACER else "") + (f"_{ARCH}arch" if ARCH else "")))
MOVE = ("Activate", "Deactivate", "BigMove", "Park")
GATE = ("2qGate", "1qGate", "Crosstalk")
T_TRANS = 15.0
REC = {"stages": []}
ZAP_LIMIT = int(os.environ.get("RC_ZAP_LIMIT", 300))
TRACE = []          # instructions the ported routers emit (verified with NEAT's AOD verifier)


def _act(qs):
    TRACE.append({"type": "Activate", "locs": [{"id": q} for q in qs]})


def _deact(qs):
    TRACE.append({"type": "Deactivate", "locs": [{"id": q} for q in qs]})


def _move(pairs):
    """pairs: {q: ((x0, y0), (x1, y1))} for atoms that move."""
    TRACE.append({"type": "BigMove", "locs": [
        {"id": q, "x_begin": a[0], "y_begin": a[1], "x_end": b[0], "y_end": b[1]} for q, (a, b) in pairs.items()]})
# routers compared against NEAT's emitted trace ("neat"); QMAP strict is not used
ROUTERS = (("mqt", lambda arch, st, tg: qmap_stage(arch, st, tg, True)),
           ("zac", lambda arch, st, tg: zac_stage(st, tg)),
           ("pm", lambda arch, st, tg: pm_stage(st, tg)),
           ("zap", None))
# RC_ROUTERS=zac (2026-10-31): compare against these routers only (comma list).
if os.environ.get("RC_ROUTERS"):
    ROUTERS = tuple(r for r in ROUTERS if r[0] in os.environ["RC_ROUTERS"].split(","))


def t_move(d):
    return 200.0 * math.sqrt(d / 110.0) if d > 0 else 0.0


# Offset moves (2026-11-04): instructions that only clear a pick-up, told apart
# by what emits them, not by distance. Pick-up slides and tone shifts, and the
# pre-park before a wave's main move; chain legs and set-downs carry atoms.
OFFSET_FNS = {"_safe_lift", "_shift_offending_tones", "_pick_up_planned", "_pick_up_rounds",
              "_pick_up_rounds_old"}
_write_move = R.Router.write_move_instruction


def _write_move_tagged(self, qs, type, end_locs):
    ins = self.results_code["instructions"]
    n = len(ins)
    _write_move(self, qs, type, end_locs)
    if len(ins) > n:
        fns, f = set(), sys._getframe(1)
        while f:
            fns.add(f.f_code.co_name)
            f = f.f_back
        ins[-1]["offset"] = bool(fns & OFFSET_FNS) or (
            type == "Park" and "process_rearrangement" in fns
            and not fns & {"_run_chain_legs", "_set_down_elsewhere"})


R.Router.write_move_instruction = _write_move_tagged


def _main(m, ops):
    """Main movement segments (kind M: every move that is not an offset, kind O)
    and the atom-moves they carry, from (kind, duration, n_atoms) ops."""
    if m is not None:
        m["main_segments"] = sum(1 for k, _, _ in ops if k == "M")
        m["main_atom_moves"] = sum(n for k, _, n in ops if k == "M")


def tdiv(a, b):
    """C++ integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


# ---------------------------------------------------------------- NEAT side
_rsm = R.Router._route_stage_moves
def _route_stage_moves(self):
    rc = self.results_code
    REC["gates"] = self.list_full_gates
    REC["stages"].append({
        "stage": self.index_list_gate,
        "start": [tuple(int(c) for c in p) for p in self.current_mapping],
        "target": [tuple(int(c) for c in p) for p in self.final_mapping],
        "g0": sum(1 for x in rc["instructions"] if x["type"] in GATE)})
    return _rsm(self)
R.Router._route_stage_moves = _route_stage_moves


def atom_distance_um(ins):
    """Total atom travel, including offset moves; stationary atoms contribute zero."""
    return math.fsum(math.hypot(l["x_end"] - l["x_begin"], l["y_end"] - l["y_begin"])
                     for x in ins if x["type"] in ("BigMove", "Park") for l in x["locs"])


def neat_metrics(ins):
    T = rounds = segs = moved = transfers = 0
    ops = []
    for x in ins:
        d = max(x.get("duration") or [0])
        if x["type"] in ("Activate", "Deactivate"):
            rounds += 1
            transfers += len(x["qs"])
            ops.append(("P" if x["type"] == "Activate" else "D", d, len(x["qs"])))
        else:
            k = sum(1 for l in x["locs"]
                    if (l["x_begin"], l["y_begin"]) != (l["x_end"], l["y_end"]))
            if k == 0:
                continue
            segs += 1
            moved += k
            ops.append(("O" if x.get("offset") else "M", d, k))
        T += d
    return {"t_us": T, "rounds": rounds, "segments": segs, "atom_moves": moved,
            "transfers": transfers, "atom_distance_um": atom_distance_um(ins)}, ops


# ---------------------------------------------------------------- QMAP side
class Arch:
    def __init__(self, a):
        self.kind = {}
        stg = [s for z in a["storage_zones"] for s in z["slms"]]
        for s in stg:
            for r in range(s["r"]):
                for c in range(s["c"]):
                    self.kind[(s["location"][0] + c * s["site_seperation"][0],
                               s["location"][1] + r * s["site_seperation"][1])] = "S"
        for z in a["entanglement_zones"]:
            sl = z["slms"]
            for i in range(0, len(sl), 2):
                pair = sorted(sl[i:i + 2], key=lambda s: s["location"][0])
                for s, k in zip(pair, ("L", "R")):
                    for r in range(s["r"]):
                        for c in range(s["c"]):
                            self.kind[(s["location"][0] + c * s["site_seperation"][0],
                                       s["location"][1] + r * s["site_seperation"][1])] = k
        sx = [p[0] for p, k in self.kind.items() if k == "S"]
        sy = [p[1] for p, k in self.kind.items() if k == "S"]
        ex = [p[0] for p, k in self.kind.items() if k != "S"]
        ey = [p[1] for p, k in self.kind.items() if k != "S"]
        # (dx, dy, minX, maxX, minY, maxY), maxima one pitch past the last site as in QMAP
        if ARCH == "zac":                # one regular grid per zone, as QMAP's own format has
            l, r = sorted(a["entanglement_zones"][0]["slms"], key=lambda s: s["location"][0])
            (ex_, ey_), sp = l["site_seperation"], r["location"][0] - l["location"][0]
            ss = a["storage_zones"][0]["slms"][0]["site_seperation"]
            self.zone = {"S": (ss[0], ss[1], min(sx), max(sx) + ss[0], min(sy), max(sy) + ss[1]),
                         "E": (ex_, ey_, min(ex), max(ex) + ex_, min(ey), max(ey) + ey_)}
            self.pair_sep = sp
            return
        self.zone = {"S": (3, 3, min(sx), max(sx) + 3, min(sy), max(sy) + 3),
                     "E": (6, 6, min(ex), max(ex) + 6, min(ey), max(ey) + 6)}
        self.pair_sep = 3

    def zk(self, p):
        return "S" if self.kind[p] == "S" else "E"


def strict_compat(v, w):
    v0, v1, v2, v3 = v
    w0, w1, w2, w3 = w
    return ((v0 == w0) == (v2 == w2) and (v0 < w0) == (v2 < w2)
            and (v1 == w1) == (v3 == w3) and (v1 < w1) == (v3 < w3))


def _cubed_sum(a, b):
    x, y = math.copysign(abs(a) ** (1 / 3), a), math.copysign(abs(b) ** (1 / 3), b)
    return a + b + 3.0 * x * y * (x + y)


def _cubed_sub(a, b):
    x, y = math.copysign(abs(a) ** (1 / 3), a), math.copysign(abs(b) ** (1 / 3), b)
    return a - b + 3.0 * x * y * (y - x)


def relaxed_compat(v, w):
    """None = incompatible, 'strict' = strictly compatible, float = merge cost."""
    v0, v1, v2, v3 = v
    w0, w1, w2, w3 = w
    if (v0 == w0) != (v2 == w2) or (v1 == w1) != (v3 == w3):
        return None
    cx, cy = (v0 < w0) != (v2 < w2), (v1 < w1) != (v3 < w3)
    if cx and cy:
        return _cubed_sum(abs(v0 - w0) + abs(v2 - w2), abs(v1 - w1) + abs(v3 - w3))
    if cx:
        return float(abs(v0 - w0) + abs(v2 - w2))
    if cy:
        return float(abs(v1 - w1) + abs(v3 - w3))
    return "strict"


def qmap_route(atoms, vec, zcls, blocker, relaxed):
    """IndependentSetRouter on one stage. ``atoms`` in QMAP's order (distance
    descending, ties by index descending). Returns a list of groups."""
    dist = {q: math.hypot(vec[q][2] - vec[q][0], vec[q][3] - vec[q][1]) for q in atoms}
    sconf = {q: set() for q in atoms}
    rconf = {q: {} for q in atoms}
    for i, a in enumerate(atoms):
        for b in atoms[i + 1:]:
            c = relaxed_compat(vec[a], vec[b]) if zcls[a] == zcls[b] else None
            if c != "strict":
                sconf[a].add(b)
                sconf[b].add(a)
                rconf[a][b] = c
                rconf[b][a] = c
    # strict groups (makeStrictRoutingForRelaxedRouting is the same loop)
    groups = []                      # [members, maxDistance, relaxedConflictingAtoms]
    where = {}
    todo = list(atoms)
    while todo:
        members, conflicting, rest = [], set(), []
        mset = set()
        for q in todo:
            b = blocker.get(q)
            if q not in conflicting and (b is None or b in where or b in mset):
                members.append(q)
                mset.add(q)
                conflicting |= sconf[q]
            else:
                rest.append(q)
        if not members:
            raise RuntimeError("dependency cycle")
        g = [members, max(dist[q] for q in members), {}]
        for q in members:
            where[q] = len(groups)
            for nb, c in rconf[q].items():
                if nb in g[2]:
                    if g[2][nb] is not None:
                        g[2][nb] = None if c is None else max(g[2][nb], c)
                else:
                    g[2][nb] = c
        groups.append(g)
        todo = rest
    if not relaxed:
        return [g[0] for g in groups]
    dependents = {}
    for q, b in blocker.items():
        dependents.setdefault(b, []).append(q)
    gi = len(groups) - 1
    while gi >= 0:
        G = groups[gi]
        inG = set(G[0])
        new, total, total3, ok = {}, 0.0, 0.0, True
        for q in G[0]:
            lo, hi = 0, len(groups) - 1          # dependency window from atoms outside G
            b = blocker.get(q)
            if b is not None and b not in inG:
                lo = where[b]
            for d in dependents.get(q, ()):
                if d not in inG:
                    hi = min(hi, where[d])
            found, cost, cost3 = None, math.inf, math.inf
            for k in range(len(groups) - 1, -1, -1):
                H = groups[k]
                if k == gi or not (lo <= k <= hi):
                    continue
                if q not in H[2]:
                    if H[1] >= dist[q]:
                        found, cost, cost3 = k, 0.0, 0.0
                        break
                    diff = _cubed_sub(dist[q], H[1])
                    if cost3 > diff:
                        found, cost3, cost = k, diff, math.copysign(abs(diff) ** (1 / 3), diff)
                elif H[2][q] is not None:
                    c = H[2][q] if H[1] >= dist[q] else _cubed_sum(_cubed_sub(dist[q], H[1]), H[2][q])
                    if cost3 > c:
                        found, cost3, cost = k, c, math.copysign(abs(c) ** (1 / 3), c)
            if found is None:
                ok = False
                break
            new[q] = found
            total3 = cost3 + total3 + 3 * cost * total * (cost + total)
            total += cost
        if ok:                                   # dependencies inside G under the new assignment
            for q in G[0]:
                b = blocker.get(q)
                if b in inG and new[b] > new[q]:
                    ok = False
                    break
        if ok and G[1] > 1.0 * total3:
            for q, k in new.items():
                H = groups[k]
                H[0].append(q)
                H[1] = max(H[1], dist[q])
                for nb, c in rconf[q].items():
                    if nb in H[2]:
                        if H[2][nb] is not None:
                            H[2][nb] = None if c is None else max(H[2][nb], c)
                    else:
                        H[2][nb] = c
            del groups[gi]
            where = {q: k for k, g in enumerate(groups) for q in g[0]}
        gi -= 1
    return [g[0] for g in groups]


class Gen:
    """QMAP's RearrangementGenerator for one group, emitting (kind, duration, n_atoms)."""

    def __init__(self, arch, qs, vec):
        self.ops = []
        self.mv = {}
        for q in qs:
            sx, sy, tx, ty = vec[q]
            self.mv[q] = (arch.kind[(sx, sy)], sx, sy, arch.kind[(tx, ty)], tx, ty)
        q0 = self.mv[qs[0]]
        (self.sdx, self.sdy, self.sminx, self.smaxx, self.sminy, self.smaxy) = arch.zone["S" if q0[0] == "S" else "E"]
        (self.tdx, self.tdy, self.tminx, self.tmaxx, self.tminy, self.tmaxy) = arch.zone["S" if q0[3] == "S" else "E"]
        self.pair = arch.pair_sep
        self.V, self.H = {}, {}
        for m in self.mv.values():
            self.V.setdefault(m[2], m[5])
            self.H.setdefault(m[1], m[4])
        self.V = dict(sorted(self.V.items()))
        self.H = dict(sorted(self.H.items()))
        tv, th = list(self.V.values()), list(self.H.values())
        self.same_rows = tv == sorted(tv)
        self.same_cols = th == sorted(th)
        k = next(iter(self.V))
        self.up = k < self.V[k]
        self.rows, self.cols, self.loc = {}, {}, {}
        self.miss = 0                # atoms stored off their target (must stay 0)

    # ordered-map helpers on dict[int -> int] keyed by AOD index
    @staticmethod
    def _below(m, k):
        return sorted((j for j in m if j < k), reverse=True)

    @staticmethod
    def _next(m, k):
        c = [j for j in m if j > k]
        return min(c) if c else None

    @staticmethod
    def _prev(m, k):
        c = [j for j in m if (k is None or j < k)]
        return max(c) if c else None

    def _emit_move(self, xcol, yrow, src):
        far, n, pairs = 0.0, 0, {}
        for q, loc in self.loc.items():
            m = self.mv[q]
            new = (self.cols[xcol[m[1] if src else m[4]]], self.rows[yrow[m[2] if src else m[5]]])
            if new != loc:
                far = max(far, math.dist(new, loc))
                n += 1
                pairs[q] = (loc, new)
                self.loc[q] = new
        if n:
            self.ops.append(("M", t_move(far), n))
            _move(pairs)

    def _same(self, a, b, i):
        return {self.mv[q][i] for q in a} == {self.mv[q][i] for q in b}

    def load_rows(self):
        by_y = {}
        for q, m in sorted(self.mv.items()):
            by_y.setdefault(m[2], set()).add(q)
        by_y = dict(sorted(by_y.items()))
        rev = dict(sorted((v, k) for k, v in self.V.items()))
        y2r = {sy: i for i, (ty, sy) in enumerate(rev.items())}
        x2c = {x: i for i, x in enumerate(self.H)}
        sign = 1 if self.up else -1
        dy = self.sdy
        for sy, qs in by_y.items():
            nr = y2r[sy]
            if nr in self.rows:
                continue
            allq = set(qs)
            y = sy + sign * tdiv(dy, 2)
            self.rows[nr] = y
            ny = y - dy
            for j in self._below(self.rows, nr):
                if self.rows[j] <= ny:
                    break
                self.rows[j] = ny
                ny -= dy if ny > self.sminy else tdiv(dy, 2)
            ny = y + dy
            up = self._next(self.rows, nr)
            while True:
                for ay in [k for k in by_y if k >= ny - sign * tdiv(dy, 2)]:
                    ar = y2r[ay]
                    if self._prev(self.rows, up) < ar and (up is None or ar < up):
                        if self._same(qs, by_y[ay], 1):
                            self.rows.setdefault(ar, ay + sign * tdiv(dy, 2))
                            allq |= by_y[ay]
                            ny = ay + dy + sign * tdiv(dy, 2)
                if up is None:
                    break
                if self.rows[up] < ny:
                    self.rows[up] = ny
                    ny += dy if ny < self.smaxy else tdiv(dy, 2)
                up = self._next(self.rows, up)
            self._emit_move(x2c, y2r, True)
            for q in qs:
                self.cols[x2c[self.mv[q][1]]] = self.mv[q][1]
            self._emit_move(x2c, y2r, True)
            for q in sorted(allq):
                m = self.mv[q]
                self.loc[q] = (m[1], m[2])
                c = x2c[m[1]]
                self.cols[c] = m[1] + (tdiv(self.sdx, 2) if m[0] == "S" else
                                       -tdiv(self.sdx, 4) if m[0] == "L" else tdiv(self.sdx, 4))
            self.ops.append(("P", T_TRANS, len(allq)))
            _act(sorted(allq))
            self._emit_move(x2c, y2r, True)

    def load_cols(self):
        by_x = {}
        for q, m in sorted(self.mv.items()):
            by_x.setdefault(m[1], set()).add(q)
        by_x = dict(sorted(by_x.items()))
        x2c = {x: i for i, x in enumerate(self.H)}
        y2r = {y: i for i, y in enumerate(self.V)}
        sign = 1 if self.up else -1
        dx, pair = self.sdx, self.pair
        for sx, qs in by_x.items():
            nc = x2c[sx]
            if nc in self.cols:
                continue
            allq = set(qs)
            kind = self.mv[min(qs)][0]
            if kind == "S":
                self.cols[nc] = sx + tdiv(dx, 2)
                nx = sx - tdiv(dx, 2)
                for j in self._below(self.cols, nc):
                    if self.cols[j] <= nx:
                        break
                    self.cols[j] = nx
                    nx -= dx if nx > self.sminx else tdiv(dx, 2)
                nx = sx + dx + tdiv(dx, 2)
                up = self._next(self.cols, nc)
                while True:
                    for ax in [k for k in by_x if k >= nx - tdiv(dx, 2)]:
                        ac = x2c[ax]
                        if self._prev(self.cols, up) < ac and (up is None or ac < up):
                            if self._same(qs, by_x[ax], 2):
                                self.cols.setdefault(ac, ax + tdiv(dx, 2))
                                allq |= by_x[ax]
                                nx = ax + dx + tdiv(dx, 2)
                    if up is None:
                        break
                    if self.cols[up] < nx:
                        self.cols[up] = nx
                        nx += dx if nx < self.smaxx else tdiv(dx, 2)
                    up = self._next(self.cols, up)
            else:
                left = kind == "L"
                off = tdiv(dx - pair, 3)
                self.cols[nc] = sx + (-off if left else off)
                nx = sx - ((dx - pair - off) if left else (pair + off))
                for j in self._below(self.cols, nc):
                    if self.cols[j] <= nx:
                        break
                    self.cols[j] = nx
                    nx -= (dx - off) if (nx > self.sminx and left) else off
                    left = not left
                left = kind == "L"
                nx = sx + ((pair + off) if left else (dx - pair - off))
                up = self._next(self.cols, nc)
                while True:
                    for ax in [k for k in by_x if k >= nx - ((pair + off) if left else (dx - pair - off))]:
                        ac = x2c[ax]
                        if self._prev(self.cols, up) < ac and (up is None or ac < up):
                            if self._same(qs, by_x[ax], 2):
                                left = self.mv[min(by_x[ax])][0] == "L"
                                self.cols.setdefault(ac, ax + (-off if left else off))
                                allq |= by_x[ax]
                                nx = ax + ((pair + off) if left else (dx - pair - off))
                    if up is None:
                        break
                    if self.cols[up] < nx:
                        self.cols[up] = nx
                        nx += off if (nx >= self.smaxx or left) else (dx - off)
                        left = not left
                    up = self._next(self.cols, up)
            self._emit_move(x2c, y2r, True)
            for q in qs:
                self.rows[y2r[self.mv[q][2]]] = self.mv[q][2]
            self._emit_move(x2c, y2r, True)
            for q in sorted(allq):
                m = self.mv[q]
                self.loc[q] = (m[1], m[2])
                self.rows[y2r[m[2]]] = m[2] + sign * tdiv(self.sdy, 2)
            self.ops.append(("P", T_TRANS, len(allq)))
            _act(sorted(allq))
            self._emit_move(x2c, y2r, True)

    def store_rows(self):
        sign = 1 if not self.up else -1
        dy, dx = self.tdy, self.tdx
        rev = dict(sorted((v, k) for k, v in self.V.items()))
        ty2r, r2ty = {}, {}
        for i, ty in enumerate(rev):
            ty2r[ty] = i
            r2ty[i] = ty
            self.rows[i] = ty + sign * tdiv(dy, 2)
        txs = sorted(set(self.H.values()))
        tx2c = {x: i for i, x in enumerate(txs)}
        by_y = {}
        for q, m in sorted(self.mv.items()):
            by_y.setdefault(m[5], set()).add(q)
            c = tx2c[m[4]]
            self.cols[c] = m[4] + (tdiv(dx, 2) if m[3] == "S" else
                                   -tdiv(dx, 4) if m[3] == "L" else tdiv(dx, 4))
        by_y = dict(sorted(by_y.items()))
        first = ty2r[next(iter(by_y))]
        y = self.rows[first]
        ny = y
        for j in self._below(self.rows, first):
            ny -= dy if ny > self.tminy else tdiv(dy, 2)
            if self.rows[j] > ny:
                self.rows[j] = ny
            else:
                ny = self.rows[j]
        ny = y
        for j in sorted(k for k in self.rows if k > first):
            ny += dy if ny < self.tmaxy else tdiv(dy, 2)
            if self.rows[j] < ny:
                self.rows[j] = ny
            else:
                ny = self.rows[j]
        for ty, qs in by_y.items():
            orow = ty2r[ty]
            if orow not in self.rows:
                continue
            allq = set(qs)
            store = [(orow, ty)]
            y = ty + sign * tdiv(dy, 2)
            self.rows[orow] = y
            ny = y - dy
            for j in self._below(self.rows, orow):
                if self.rows[j] <= ny:
                    break
                self.rows[j] = ny
                ny -= dy if ny > self.tminy else tdiv(dy, 2)
            ny = y + dy
            free = 0
            up = self._next(self.rows, orow)
            while up is not None:
                rty = r2ty[up]
                rq = by_y.get(rty, set())
                if rty >= ny - sign * tdiv(dy, 2) + free * dy and self._same(qs, rq, 4):
                    store.append((up, rty))
                    self.rows[up] = rty + sign * tdiv(dy, 2)
                    allq |= rq
                    ny = rty - sign * tdiv(dy, 2)
                    for j in self._below(self.rows, up):
                        if self.rows[j] <= ny:
                            break
                        self.rows[j] = ny
                        ny -= dy if ny > self.tminy else tdiv(dy, 2)
                    ny = rty + dy + sign * tdiv(dy, 2)
                    free = 0
                elif self.rows[up] < ny:
                    self.rows[up] = ny
                    ny += dy if ny < self.tmaxy else tdiv(dy, 2)
                else:
                    free += 1
                up = self._next(self.rows, up)
            self._emit_move(tx2c, ty2r, False)
            for q in qs:
                self.cols[tx2c[self.mv[q][4]]] = self.mv[q][4]
            for r, rty in store:
                self.rows[r] = rty
            self._emit_move(tx2c, ty2r, False)
            for q in sorted(allq):
                m = self.mv[q]
                self.miss += self.loc[q] != (m[4], m[5])
                del self.loc[q]
                self.cols[tx2c[m[4]]] = m[4] + (tdiv(dx, 2) if m[3] == "S" else
                                                -tdiv(dx, 4) if m[3] == "L" else tdiv(dx, 4))
            self.ops.append(("D", T_TRANS, len(allq)))
            _deact(sorted(allq))
            for r, _ in store:
                del self.rows[r]
            self._emit_move(tx2c, ty2r, False)

    def store_cols(self):
        by_x = {}
        for q, m in sorted(self.mv.items()):
            by_x.setdefault(m[4], set()).add(q)
        by_x = dict(sorted(by_x.items()))
        tx2c, c2tx = {}, {}
        for i, (sx, tx) in enumerate(self.H.items()):
            tx2c[tx] = i
            c2tx[i] = tx
        sign = 1 if not self.up else -1
        dx, dy, pair = self.tdx, self.tdy, self.pair
        off = tdiv(dx - pair, 3)
        ty2r = {}
        for i, ty in enumerate(sorted(set(self.V.values()))):
            ty2r[ty] = i
            self.rows[i] = ty + sign * tdiv(dy, 2)
        for tx, qs in by_x.items():
            c = tx2c[tx]
            k = self.mv[min(qs)][3]
            self.cols[c] = tx + (tdiv(dx, 2) if k == "S" else -off if k == "L" else off)
        tx0, q0 = next(iter(by_x.items()))
        first = tx2c[tx0]
        x = self.cols[first]

        def flip(k):
            return "R" if k == "L" else "L"
        k = self.mv[min(q0)][3]
        nx = x
        for j in self._below(self.cols, first):
            if k == "S":
                nx -= dx if nx > self.tminx else tdiv(dx, 2)
            else:
                nx -= (dx - off) if (nx > self.tminx and k == "R") else off
                k = flip(k)
            if self.cols[j] > nx:
                self.cols[j] = nx
            else:
                if k != "S" and (nx - self.cols[j]) % dx != 0:
                    k = flip(k)
                nx = self.cols[j]
        k = self.mv[min(q0)][3]
        nx = x
        for j in sorted(c for c in self.cols if c > first):
            if k == "S":
                nx += dx if nx > self.tminx else tdiv(dx, 2)
            else:
                nx += (dx - off) if (nx > self.tminx and k == "L") else off
                k = flip(k)
            if self.cols[j] < nx:
                self.cols[j] = nx
            else:
                if k != "S" and (self.cols[j] - nx) % dx != 0:
                    k = flip(k)
                nx = self.cols[j]
        for tx, qs in by_x.items():
            oc = tx2c[tx]
            if oc not in self.cols:
                continue
            allq = set(qs)
            store = [(oc, tx)]
            kind = self.mv[min(qs)][3]
            if kind == "S":
                self.cols[oc] = tx + tdiv(dx, 2)
                nx = tx - tdiv(dx, 2)
                for j in self._below(self.cols, oc):
                    if self.cols[j] <= nx:
                        break
                    self.cols[j] = nx
                    nx -= dx if nx > self.tminx else tdiv(dx, 2)
                nx = tx + dx + tdiv(dx, 2)
                free = 0
                up = self._next(self.cols, oc)
                while up is not None:
                    ctx = c2tx[up]
                    cq = by_x.get(ctx, set())
                    if ctx >= nx + free * dx - tdiv(dx, 2) and self._same(qs, cq, 5):
                        store.append((up, ctx))
                        self.cols[up] = ctx + tdiv(dx, 2)
                        allq |= cq
                        nx = ctx - tdiv(dx, 2)
                        for j in self._below(self.cols, up):
                            if self.cols[j] <= nx:
                                break
                            self.cols[j] = nx
                            nx -= dx if nx > self.tminx else tdiv(dx, 2)
                        nx = ctx + dx + tdiv(dx, 2)
                        free = 0
                    elif self.cols[up] < nx:
                        self.cols[up] = nx
                        nx += dx if nx < self.tmaxx else tdiv(dx, 2)
                        free = 0
                    else:
                        free += 1
                    up = self._next(self.cols, up)
            else:
                left = kind == "L"
                self.cols[oc] = tx + (-off if left else off)
                nx = tx - ((dx - pair - off) if left else (pair + off))
                for j in self._below(self.cols, oc):
                    if self.cols[j] <= nx:
                        break
                    self.cols[j] = nx
                    nx -= (dx - off) if (nx > self.tminx and left) else off
                    left = not left
                left = kind == "L"
                nx = tx + ((pair + off) if left else (dx - pair - off))
                free = 0
                up = self._next(self.cols, oc)
                while up is not None:
                    ctx = c2tx[up]
                    cq = by_x.get(ctx, set())
                    need = (nx - (off if left else -off) + (free // 2) * dx
                            + ((off if left else dx - off) if free % 2 == 1 else 0))
                    if ctx >= need and self._same(qs, cq, 5):
                        left = self.mv[min(cq)][3] == "L"
                        store.append((up, ctx))
                        self.cols[up] = ctx + (-off if left else off)
                        allq |= cq
                        nx = ctx - ((dx - pair - off) if left else (pair + off))
                        for j in self._below(self.cols, up):
                            if self.cols[j] <= nx:
                                break
                            self.cols[j] = nx
                            nx -= (dx - off) if (nx > self.tminx and left) else off
                            left = not left
                        left = self.mv[min(cq)][3] == "L"
                        nx = ctx + ((pair + off) if left else (dx - pair - off))
                        free = 0
                    elif self.cols[up] < nx:
                        self.cols[up] = nx
                        nx += off if (nx >= self.tmaxx or left) else (dx - off)
                        left = not left
                        free = 0
                    else:
                        free += 1
                    up = self._next(self.cols, up)
            self._emit_move(tx2c, ty2r, False)
            for q in qs:
                self.rows[ty2r[self.mv[q][5]]] = self.mv[q][5]
            for cc, ctx in store:
                self.cols[cc] = ctx
            self._emit_move(tx2c, ty2r, False)
            for q in sorted(allq):
                m = self.mv[q]
                self.miss += self.loc[q] != (m[4], m[5])
                del self.loc[q]
                self.rows[ty2r[m[5]]] = m[5] + sign * tdiv(dy, 2)
            self.ops.append(("D", T_TRANS, len(allq)))
            _deact(sorted(allq))
            for cc, _ in store:
                del self.cols[cc]
            self._emit_move(tx2c, ty2r, False)

    def generate(self):
        nv = len(self.V)
        nh = len(self.H)
        if self.same_rows and nh < nv:
            self.load_cols()
            self.store_cols()
        elif self.same_cols and nv < nh:
            self.load_rows()
            self.store_rows()
        else:
            self.load_rows()
            self.store_cols()
        return self.ops


def qmap_stage(arch, start, target, relaxed):
    moving = [q for q in range(len(start)) if start[q] != target[q]]
    if not moving:
        return None, []
    vec = {q: (start[q][0], start[q][1], target[q][0], target[q][1]) for q in moving}
    at = {start[q]: q for q in moving}
    blocker = {q: at[target[q]] for q in moving if target[q] in at}
    zcls = {q: (arch.zk(start[q]), arch.zk(target[q])) for q in moving}
    order = sorted(moving, key=lambda q: (math.hypot(vec[q][2] - vec[q][0], vec[q][3] - vec[q][1]), q),
                   reverse=True)
    groups = qmap_route(order, vec, zcls, blocker, relaxed)
    ops = []
    for g in groups:
        gen = Gen(arch, g, vec)
        ops += gen.generate()
        if gen.miss:
            raise RuntimeError(f"{gen.miss} atoms stored off target")
    T = sum(d for _, d, _ in ops)
    rounds = sum(1 for k, _, _ in ops if k != "M")
    segs = sum(1 for k, _, _ in ops if k == "M")
    return {"t_us": T, "rounds": rounds, "segments": segs,
            "atom_moves": sum(n for k, _, n in ops if k == "M"),
            "transfers": sum(n for k, _, n in ops if k != "M"),
            "groups": len(groups)}, ops


def _stage_moves(start, target):
    moving = [q for q in range(len(start)) if start[q] != target[q]]
    vec = {q: (start[q][0], start[q][1], target[q][0], target[q][1]) for q in moving}
    at = {start[q]: q for q in moving}
    blocker = {q: at[target[q]] for q in moving if target[q] in at}
    return moving, vec, blocker


def _metrics(ops, groups):
    return {"t_us": sum(d for _, d, _ in ops), "rounds": sum(1 for k, _, _ in ops if k in "PD"),
            "segments": sum(1 for k, _, _ in ops if k in "MO"),
            "atom_moves": sum(n for k, _, n in ops if k in "MO"),
            "transfers": sum(n for k, _, n in ops if k in "PD"), "groups": groups}


def zac_stage(start, target, park=1):
    """ZAC's router (baselines/zac/zac/router/router.py, maximalis_sort, window
    1000 > every stage here): greedy maximal independent set of compatible_2D
    moves, longest first; each job is expanded as ZAC's expand_arrangement does
    -- per source row (in pick-up order): shift parked columns back, activate
    the row, park it by PARKING_DIST (1 um) unless it is the last row; then one
    big move (as long as the longest row x column displacement) and one
    deactivate."""
    moving, vec, blocker = _stage_moves(start, target)
    if not moving:
        return None, []
    order = sorted(moving, key=lambda q: (-math.hypot(vec[q][2] - vec[q][0], vec[q][3] - vec[q][1]), q))
    done, groups, todo = set(), [], order
    while todo:
        g, gs, rest = [], set(), []
        for q in todo:
            b = blocker.get(q)
            if (b is None or b in done or b in gs) and all(strict_compat(vec[q], vec[o]) for o in g):
                g.append(q)
                gs.add(q)
            else:
                rest.append(q)
        if not g:
            raise RuntimeError("dependency cycle")
        groups.append(g)
        done |= gs
        todo = rest
    ops = []
    for g in groups:
        rows = {}
        for q in sorted(g):
            rows.setdefault(vec[q][1], []).append(q)
        pos = {q: (vec[q][0], vec[q][1]) for q in g}
        held, colx = [], set()          # held atoms; source x of the active AOD columns
        rl = list(rows.values())
        for i, row in enumerate(rl):
            # shift the parked columns this row reuses back onto the trap column
            back = {vec[q][0] for q in row} & colx
            pairs = {h: (pos[h], (pos[h][0] - park, pos[h][1])) for h in held if vec[h][0] in back}
            if pairs:
                ops.append(("O", t_move(park), len(pairs)))
                _move(pairs)
                for h, (_, b) in pairs.items():
                    pos[h] = b
            ops.append(("P", T_TRANS, len(row)))
            _act(row)
            held += row
            colx |= {vec[q][0] for q in row}
            if i < len(rl) - 1:
                # park: this row +park in y, its columns +park in x (with every held atom in them)
                cx = {vec[q][0] for q in row}
                pairs = {}
                for h in held:
                    dx = park if vec[h][0] in cx else 0
                    dy = park if h in row else 0
                    if dx or dy:
                        pairs[h] = (pos[h], (pos[h][0] + dx, pos[h][1] + dy))
                if pairs:
                    ops.append(("O", t_move(max(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in pairs.values())),
                                len(pairs)))
                    _move(pairs)
                    for h, (_, b) in pairs.items():
                        pos[h] = b
        dx = max(abs(vec[q][2] - pos[q][0]) for q in g)
        dy = max(abs(vec[q][3] - pos[q][1]) for q in g)
        ops.append(("M", t_move(math.hypot(dx, dy)), len(g)))
        _move({q: (pos[q], (vec[q][2], vec[q][3])) for q in g})
        ops.append(("D", T_TRANS, len(g)))
        _deact(g)
    return _metrics(ops, len(groups)), ops


def pm_stage(start, target):
    """PowerMove's router (baselines/powermove/mvqc.py): moves sorted by
    ascending Manhattan distance, each put in the first group with which it
    keeps the order on both axes (check_conflict); each group is one pick-up of
    all its atoms, one move (as long as its longest atom) and one drop-off."""
    moving, vec, blocker = _stage_moves(start, target)
    if not moving:
        return None, []
    order = sorted(moving, key=lambda q: (abs(vec[q][2] - vec[q][0]) + abs(vec[q][3] - vec[q][1]), q))
    groups, where, todo = [], {}, order
    while todo:
        rest = []
        for q in todo:
            b = blocker.get(q)
            if b is not None and b not in where:
                rest.append(q)
                continue
            lo = where[b] if b is not None else 0
            for i in range(lo, len(groups)):
                if all(strict_compat(vec[q], vec[o]) for o in groups[i]):
                    groups[i].append(q)
                    where[q] = i
                    break
            else:
                groups.append([q])
                where[q] = len(groups) - 1
        if len(rest) == len(todo):
            raise RuntimeError("dependency cycle")
        todo = rest
    ops = []
    for g in groups:
        ops.append(("P", T_TRANS, len(g)))
        _act(g)
        ops.append(("M", t_move(max(math.hypot(vec[q][2] - vec[q][0], vec[q][3] - vec[q][1]) for q in g)), len(g)))
        _move({q: ((vec[q][0], vec[q][1]), (vec[q][2], vec[q][3])) for q in g})
        ops.append(("D", T_TRANS, len(g)))
        _deact(g)
    return _metrics(ops, len(groups)), ops


def zap_chunks(z, rc, a, gates, stages):
    """Baseline ZAP's router (baselines/zap/router.py, unmodified) driven through
    NEAT's per-stage targets: its Placer is replaced by a stub returning them.
    Returns {stage: [movement instructions]}."""
    import zap.router as ZR
    init = [tuple(p) for p in stages[0]["start"]]
    tg = {s["stage"]: [tuple(p) for p in s["target"]] for s in stages}

    class Stub:
        def __init__(self, *args, **kw):
            self.current_mapping = list(init)

        def placing(self, i):
            return list(tg[i])
    marks = []
    pg = ZR.Router.process_gate

    def _pg(self):
        marks.append((self.index_list_gate, len(self.results_code["instructions"])))
        pg(self)
        marks.append((None, len(self.results_code["instructions"])))
    ZR.Placer, ZR.Router.process_gate = Stub, _pg
    import signal

    def _late(*_):
        raise TimeoutError(f"zap router > {ZAP_LIMIT} s (its fan-out loop does not finish on this array)")
    signal.signal(signal.SIGALRM, _late)
    signal.alarm(ZAP_LIMIT)
    try:
        rc2 = {k: v for k, v in rc.items() if k not in ("instructions",)}
        rc2["instructions"] = []
        r = ZR.Router(slm_sites=[list(z.stg_slm_sites), list(z.ent_slm_sites)], results_code=rc2,
                      list_full_gates=gates, qubit_mapping=init, architecture=a)
    finally:
        signal.alarm(0)
        ZR.Router.process_gate = pg
    ins = r.results_code["instructions"]
    ds = float((a.get('routing') or {}).get('min_tone_sep', 2.0) or 0.0)
    v = Verifier(results_code=r.results_code, traps=list(z.stg_slm_sites) + list(z.ent_slm_sites), d_s=ds)
    out, prev = {"_verify": {"legal": v.legal, "violations": sum(v.counts.values())}}, 0
    for st, n in marks:
        if st is not None:
            out[st] = [x for x in ins[prev:n] if x["type"] in MOVE]
        prev = n
    return out



def main():
    suite, circ = sys.argv[1], sys.argv[2]
    n_sample = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    out_dir = os.path.join(OUT, suite.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{circ}.json")
    if os.path.exists(out):
        return
    b = f"{suite}/{circ}.qasm"
    a = json.loads(json.dumps(ZAC_ARCH)) if ARCH == "zac" else build_architecture(_qasm_n_qubits(b))
    a.setdefault("routing", {})["initial_mapping_parallel_lookahead"] = run_zap.CONFIG[suite][1]   # read only by ZAP's placer
    # RC_ABLATE_WAYPOINTS=1 (2026-11-07): NEAT's router with waypoints off (the
    # router's `ablate_waypoints` gate), the "NEAT w/o waypoints" arm of the routing table.
    if os.environ.get("RC_ABLATE_WAYPOINTS") == "1":
        a["routing"]["ablate_waypoints"] = True
    REC["stages"] = []
    if PLACER:
        F.M.S.update(stage=-1, force="all", pol=PLACER, rng=F.M.random.Random(0), costs=[], pred=None,
                     sig=None, decisions=0, zp=None, arch=a, detail=[])
    z = Zap(benchmark=b, architecture=a, initial_mapping=[], output_dir="")
    rc, _ = z._compile_once("alap_separate", a)
    z._merge_transfers(rc)
    ds = float((a.get('routing') or {}).get('min_tone_sep', 2.0) or 0.0)
    legal = Verifier(results_code=rc, traps=list(z.stg_slm_sites) + list(z.ent_slm_sites), d_s=ds).legal
    ins = rc["instructions"]
    # movement instructions between the g0-th and the next gate instruction belong to that stage
    chunks, cur, g = {}, [], 0
    for x in ins:
        if x["type"] in GATE:
            chunks[g] = cur
            cur, g = [], g + 1
        elif x["type"] in MOVE:
            cur.append(x)
    chunks[g] = cur
    arch = Arch(a)
    n_q = rc["n_q"]
    f_tr = a["operation_fidelity"]["atom_transfer"]
    t2 = a["qubit_spec"]["T2"]
    with_moves = [s for s in REC["stages"] if s["start"] != s["target"]]
    k = min(n_sample, len(with_moves))
    sample = {with_moves[round(j * (len(with_moves) - 1) / max(1, k - 1))]["stage"] for j in range(k)}
    stages, used, fails = [], set(), []
    t0 = time.time()
    zch = None
    if any(n == "zap" for n, _ in ROUTERS):
        try:
            zch = zap_chunks(z, rc, a, REC["gates"], REC["stages"])
        except Exception as e:                       # noqa: BLE001
            fails.append({"stage": -1, "router": "zap", "error": repr(e)})
    traces = {n: [{"type": "Init", "locs": [{"id": q, "x": p[0], "y": p[1]}
                                         for q, p in enumerate(REC["stages"][0]["start"])]}]
              for n, fn in ROUTERS if fn is not None}
    for s in REC["stages"]:
        if s["start"] == s["target"]:
            continue
        used.add(s["g0"])
        rec = {"stage": s["stage"], "n_moving": sum(p != q for p, q in zip(s["start"], s["target"]))}
        rec["neat"], nops = neat_metrics(chunks.get(s["g0"], []))
        _main(rec["neat"], nops)
        ops = {"neat": nops}
        for name, fn in ROUTERS:
            try:
                if name == "zap":
                    if zch is None:
                        raise RuntimeError("zap router failed")
                    rec[name], ops[name] = neat_metrics(zch.get(s["stage"], []))
                else:
                    TRACE.clear()
                    rec[name], ops[name] = fn(arch, s["start"], s["target"])
                    rec[name]["atom_distance_um"] = atom_distance_um(TRACE)
                    traces[name] += TRACE
                _main(rec[name], ops[name])
            except Exception as e:                   # noqa: BLE001
                fails.append({"stage": s["stage"], "router": name, "error": repr(e)})
                rec[name] = None
        if s["stage"] in sample:
            rec["ops"] = ops
            rec["moves"] = [[*s["start"][q], *s["target"][q]] for q in range(n_q)
                            if s["start"][q] != s["target"][q]]
        stages.append(rec)
    stray = sum(len(v) for g0, v in chunks.items() if g0 not in used)
    traps = list(z.stg_slm_sites) + list(z.ent_slm_sites)
    verify = {"neat": {"legal": legal}, "zap": zch.get("_verify") if zch else None}
    for n, tr in traces.items():
        v = Verifier(results_code={"instructions": tr}, traps=traps, d_s=ds)
        verify[n] = {"legal": v.legal, "violations": sum(v.counts.values()),
                     "counts": {k: c for k, c in v.counts.items() if c}}

    def loss(m):
        return (n_q * m["t_us"] / t2 - m["transfers"] * math.log(f_tr)) / math.log(10)
    summ = {}
    for name in ["neat"] + [n for n, _ in ROUTERS]:
        ms = [st[name] for st in stages if st[name] is not None]
        tot = {key: sum(m.get(key, 0) for m in ms) for key in ("t_us", "rounds", "segments", "atom_moves", "transfers",
                                                                  "main_segments", "main_atom_moves", "atom_distance_um")}
        tot["neglog10F_routing"] = sum(loss(m) for m in ms)
        tot["parallelism"] = tot["atom_moves"] / tot["segments"] if tot["segments"] else 0.0
        summ[name] = tot
    rec = {"suite": suite, "circuit": circ, "placer": PLACER or "neat", "n_q": n_q, "neat_aod_legal": legal,
           "zap_verify": zch.get("_verify") if zch else None, "verify": verify,
           "n_stages_moved": len(stages), "stray_move_instructions": stray,
           "failed": fails, "summary": summ, "stages": stages, "qmap_s": time.time() - t0}
    with open(out + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(out + ".tmp", out)
    print(json.dumps({"circuit": circ, "stages": len(stages), "stray": stray, "fails": len(fails),
                      **{n: round(summ[n]["t_us"]) for n in summ}}), flush=True)


if __name__ == "__main__":
    main()
