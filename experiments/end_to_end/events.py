"""One event-trace format for every compiler (2026-10-06 regeneration run).

Each compiler's native output (ZAP/NEAT instruction JSON, ZAC's ZAIR, QMAP's
.naviz, PowerMove's mvqc trace) is converted into the same flat list of events,
so fidelity can be recomputed for all of them by one script
(``experiments/end_to_end/rescore.py``) and checked for AOD legality by one checker
(``experiments/end_to_end/legality.py``) -- no recompilation when a formula changes.

Event types (positions in um, atom ids 0..n_q-1):

  {"t": "1q",    "dur": us, "busy": [[q, us], ...], "n": gates}
  {"t": "cz",    "dur": us, "busy": [[q, us], ...], "pairs": [[a, b], ...]}
  {"t": "xtalk",            "busy": [[q, us], ...], "qs": [idle atoms in zone]}
  {"t": "pick",  "dur": us, "busy": [[q, us], ...], "qs": [...]}
  {"t": "drop",  "dur": us, "busy": [[q, us], ...], "qs": [...]}
  {"t": "move",  "dur": us, "busy": [[q, us], ...],
                 "moves": [[q, x0, y0, x1, y1], ...]}       # one parallel AOD step
  {"t": "job",   "dur": us, "busy": [[q, us], ...], "transfers": k}
                 # ZAC only: a whole rearrangeJob timed by ZAC itself; its
                 # pick/move/drop sub-steps follow as dur-0 events marked
                 # "sub": true, so the legality checker sees the geometry and
                 # the rescorer does not count them twice.

"dur" is what the source compiler's scorer added to the makespan for that
step (events are sequential); "busy" is what it credited each atom. Rescoring
with the recorded durations reproduces the scorer each compiler was scored
with before; ``rescore.py --recompute`` instead derives durations from the
geometry and one shared timing model.

Every events file also carries "traps": the trap coordinates of the hardware
the compiler ran on (for the off-trap release and occupancy rules).
"""
import math

T_1Q, T_2Q, T_TR = 52.0, 0.27, 15.0   # 2026-10-07: paper Table tab:params (2q was 0.36 us)


def movement_duration(d):
    """ZAP/ZAC's shared transport model: t = 200*sqrt(d/110) us."""
    return 200.0 * math.sqrt(d / 110.0) if d > 0 else 0.0


def _busy(qs, durs):
    return [[int(q), float(d)] for q, d in zip(qs, durs)]


# ---------------------------------------------------------------- ZAP / NEAT
def from_zap(results_code, traps):
    """zap / NEAT ``results_code['instructions']`` -> events.

    Mirrors Simulator.simulate exactly: every instruction adds max(duration)
    to the makespan and credits each listed atom its own duration entry;
    Crosstalk adds no makespan.
    """
    ev = []
    init = None
    for ins in results_code['instructions']:
        t = ins['type']
        if t == 'Init':
            init = {int(l['id']): (l['x'], l['y']) for l in ins['locs']}
            continue
        dur = float(max(ins['duration'])) if ins.get('duration') else 0.0
        busy = _busy(ins['qs'], ins['duration'])
        if t == 'Activate':
            ev.append({"t": "pick", "dur": dur, "busy": busy, "qs": list(ins['qs'])})
        elif t == 'Deactivate':
            ev.append({"t": "drop", "dur": dur, "busy": busy, "qs": list(ins['qs'])})
        elif t in ('BigMove', 'Park'):
            ev.append({"t": "move", "dur": dur, "busy": busy,
                       "moves": [[int(l['id']), l['x_begin'], l['y_begin'], l['x_end'], l['y_end']]
                                 for l in ins['locs']]})
        elif t == '1qGate':
            ev.append({"t": "1q", "dur": dur, "busy": busy, "n": len(ins['gates'])})
        elif t == '2qGate':
            ev.append({"t": "cz", "dur": dur, "busy": busy,
                       "pairs": [list(map(int, g)) for g in ins['gates']]})
        elif t == 'Crosstalk':
            ev.append({"t": "xtalk", "busy": busy, "qs": list(ins['qs'])})
        else:
            raise ValueError(f"unknown ZAP instruction {t!r}")
    return {"init": init, "events": ev, "traps": [list(p) for p in traps]}


# ---------------------------------------------------------------- ZAC (ZAIR)
def zac_sites(spec):
    """{slm_id: function (r, c) -> (x, y)} and the full trap list of a ZAC spec."""
    slms = {}
    for z in spec['storage_zones'] + spec['entanglement_zones']:
        for s in z['slms']:
            slms[s['id']] = s
    def xy(a, r, c):
        s = slms[a]
        return (s['location'][0] + c * s['site_seperation'][0],
                s['location'][1] + r * s['site_seperation'][1])
    traps = [xy(a, r, c) for a, s in slms.items() for r in range(s['r']) for c in range(s['c'])]
    return xy, traps


def from_zair(code, spec, ent_ids):
    """ZAC ZAIR -> events. Mirrors baselines/run_zac.py score(): 1qGate layers
    charged t_1q x (max gates on one atom), every other instruction its own
    ZAC-reported span; rearrangeJob atoms busy for the whole span."""
    xy, traps = zac_sites(spec)
    insts = code['instructions']
    assert insts[0]['type'] == 'init'
    init = {int(q): xy(a, r, c) for q, a, r, c in insts[0]['init_locs']}
    in_zone = set()
    ev = []
    for ins in insts[1:]:
        t = ins['type']
        span = float(ins['end_time'] - ins['begin_time']) if 'begin_time' in ins else 0.0
        if t == '1qGate':
            per = {}
            for g in ins['gates']:
                per[g['q']] = per.get(g['q'], 0) + 1
            ev.append({"t": "1q", "dur": T_1Q * max(per.values(), default=0),
                       "busy": [[int(g['q']), T_1Q] for g in ins['gates']], "n": len(ins['gates'])})
        elif t == 'rydberg':
            pairs = [[int(g['q0']), int(g['q1'])] for g in ins['gates']]
            gated = {q for p in pairs for q in p}
            ev.append({"t": "cz", "dur": span, "busy": [[q, T_2Q] for q in sorted(in_zone)],
                       "pairs": pairs})
            idle = sorted(in_zone - gated)
            if idle:
                ev.append({"t": "xtalk", "busy": [], "qs": idle})
        elif t == 'rearrangeJob':
            aod = [int(q) for q in ins['aod_qubits']]
            ev.append({"t": "job", "dur": span, "busy": [[q, span] for q in aod],
                       "transfers": 2 * len(aod)})
            # Replayed at the level of AOD lines, by ZAC's own row/col ids
            # (2026-10-23). ZAC's per-atom begin_coord/end_coord go stale: a
            # shift_back updates only the new row's atoms (zac/router/router.py
            # ~816), and move:big then reads col_x_begin from that stale coord.
            # Reading them, and treating only an activate's NEW columns as lit,
            # reported unheld moves and crossings that ZAC never makes. Here a
            # line keeps its tracked position, an activate lights every active
            # line, and a held atom sits wherever its row and column are.
            begin = {int(q): xy(a, r, c) for q, a, r, c in ins['begin_locs']}
            rows, cols, on = {}, {}, {}     # line id -> coord; atom -> (row id, col id)
            for s in ins.get('insts', []):
                st = s['type']
                if st == 'activate':
                    rows.update(zip(s['row_id'], s['row_y']))
                    cols.update(zip(s['col_id'], s['col_x']))
                    at = {(x, y): (ri, ci) for ri, y in rows.items() for ci, x in cols.items()}
                    new = [q for q in aod if q not in on and begin[q] in at]
                    for q in new:
                        on[q] = at[begin[q]]
                    ev.append({"t": "pick", "dur": 0.0, "busy": [], "qs": new, "sub": True,
                               "lines": [sorted(cols.values()), sorted(rows.values())]})
                elif st.startswith('move'):
                    before = {q: (cols[ci], rows[ri]) for q, (ri, ci) in on.items()}
                    rows.update(zip(s['row_id'], s['row_y_end']))
                    cols.update(zip(s['col_id'], s['col_x_end']))
                    mv = [[q, *before[q], cols[ci], rows[ri]] for q, (ri, ci) in on.items()]
                    ev.append({"t": "move", "dur": 0.0, "busy": [], "moves": mv, "sub": True})
                elif st == 'deactivate':
                    ev.append({"t": "drop", "dur": 0.0, "busy": [], "qs": list(on), "sub": True})
                    rows, cols, on = {}, {}, {}
            for q, a, r, c in ins['end_locs']:
                (in_zone.add if a in ent_ids else in_zone.discard)(int(q))
        else:
            raise ValueError(f"unknown ZAIR instruction {t!r}")
    return {"init": init, "events": ev, "traps": [list(p) for p in traps]}


# ---------------------------------------------------------------- QMAP (.naviz)
def from_naviz(code, n_q, traps, in_zone_fn):
    """QMAP .naviz -> events. Mirrors baselines/run_mqt.py parse_naviz:
    1q gates charged one t_1q per layer (a run of 1q lines; a repeated atom
    opens a new layer), load/store one t_transfer per instruction, a move block
    the slowest atom's movement_duration (airborne atoms busy for it)."""
    import re
    coord = re.compile(r"\(\s*([\-0-9.]+)\s*,\s*([\-0-9.]+)\s*\)\s*(\w+)")
    lines = code.split("\n")
    name_to_idx, pos, airborne, in_zone = {}, {}, set(), set()
    init = {}
    ev = []
    layer = None   # open 1q event
    def block(i):
        items, j = [], i + 1
        while j < len(lines) and lines[j].strip() != "]":
            if lines[j].strip():
                items.append(lines[j].strip())
            j += 1
        return items, j + 1
    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        if not raw:
            i += 1
            continue
        if raw.startswith("atom ("):
            m = coord.match(raw[len("atom "):])
            x, y, nm = float(m.group(1)), float(m.group(2)), m.group(3)
            q = len(name_to_idx)
            name_to_idx[nm] = q
            pos[q] = (x, y)
            init[q] = (x, y)
            if in_zone_fn((x, y)):
                in_zone.add(q)
            i += 1
            continue
        line = raw[2:].strip() if raw.startswith("@+") else raw
        parts = line.split(None, 1)
        kw = parts[0] if parts else ""
        rest = parts[1] if len(parts) > 1 else ""
        if kw in ("u", "rz"):
            q = name_to_idx[rest.split()[-1]]
            if layer is None or q in layer['_qs']:
                layer = {"t": "1q", "dur": T_1Q, "busy": [], "n": 0, "_qs": set()}
                ev.append(layer)
            layer['_qs'].add(q)
            layer['busy'].append([q, T_1Q])
            layer['n'] += 1
            i += 1
            continue
        layer = None
        if kw == "cz":
            qs = sorted(in_zone)
            ev.append({"t": "cz", "dur": T_2Q, "busy": [[q, T_2Q] for q in qs],
                       "pairs": _pair_up(qs, pos)})
            idle = [q for q in qs if q not in {a for p in ev[-1]['pairs'] for a in p}]
            if idle:
                ev.append({"t": "xtalk", "busy": [], "qs": idle})
            i += 1
        elif kw in ("load", "store"):
            if rest.strip() in ("[", ""):
                names, i = block(i)
            else:
                names, i = [rest.strip()], i + 1
            qs = [name_to_idx[n] for n in names]
            if kw == "load":
                airborne.update(qs)
                ev.append({"t": "pick", "dur": T_TR, "busy": [[q, T_TR] for q in qs], "qs": qs})
            else:
                for q in qs:
                    airborne.discard(q)
                    (in_zone.add if in_zone_fn(pos[q]) else in_zone.discard)(q)
                ev.append({"t": "drop", "dur": T_TR, "busy": [[q, T_TR] for q in qs], "qs": qs})
        elif kw == "move":
            if rest.strip() in ("[", ""):
                items, i = block(i)
            else:
                items, i = [rest.strip()], i + 1
            mv, busy, dmax = [], [], 0.0
            for it in items:
                m = coord.match(it)
                x, y, q = float(m.group(1)), float(m.group(2)), name_to_idx[m.group(3)]
                ox, oy = pos[q]
                d = movement_duration(math.hypot(x - ox, y - oy))
                if q in airborne:
                    busy.append([q, d])
                dmax = max(dmax, d)
                mv.append([q, ox, oy, x, y])
                pos[q] = (x, y)
            ev.append({"t": "move", "dur": dmax, "busy": busy, "moves": mv})
        else:
            i += 1
    for e in ev:
        e.pop('_qs', None)
    return {"init": init, "events": ev, "traps": [list(p) for p in traps]}


def _pair_up(qs, pos, r=6.0):
    """Greedy nearest pairing of atoms in the zone (Rydberg pairs sit 2 um apart
    on ZAC's array; idle atoms have no partner within ``r``)."""
    left = sorted(qs, key=lambda q: pos[q])
    pairs, used = [], set()
    for a in left:
        if a in used:
            continue
        best = None
        for b in left:
            if b == a or b in used:
                continue
            d = math.dist(pos[a], pos[b])
            if d <= r and (best is None or d < best[0]):
                best = (d, b)
        if best:
            used |= {a, best[1]}
            pairs.append([a, best[1]])
    return pairs


# ---------------------------------------------------------------- PowerMove
def powermove_phys(p, x_sep=19, y_sep=15, s_sep=5):
    x, y = p
    return (x * x_sep, y * y_sep if y >= 0 else -2 * y_sep + (y + 2) * s_sep)


def from_powermove(trace, layers_1q, row, t_tr=T_TR):
    """mvqc LAST_TRACE -> events. Mirrors score_powermove: each parallel move
    group = one pickup + one flight + one drop (t_tr each way), flight timed on
    the straight-line distance; each CZ stage T_2Q with 0.36 us busy per gate
    atom; idle = atoms outside storage (grid y >= 0) not in a pair. 1q layers
    come from PowerMove's own DAG layering (``layers_1q``, atom lists): one
    T_1Q of makespan per layer and T_1Q busy per gate -- the busy credit is the
    2026-10-06 fairness fix (ZAP, ZAC and QMAP all credit it; the old
    score_powermove did not)."""
    init = {int(q): powermove_phys(p) for q, p in trace[0]['pos'].items()}
    grid = {int(q): tuple(p) for q, p in trace[0]['pos'].items()}
    ev = []
    for e in trace[1:]:
        if e['kind'] == 'move':
            qs = [int(q) for q, _, _ in e['qs']]
            mv = []
            for q, a, b in e['qs']:
                pa, pb = powermove_phys(a), powermove_phys(b)
                mv.append([int(q), *pa, *pb])
                grid[int(q)] = tuple(b)
            d = e['duration']
            ev.append({"t": "pick", "dur": t_tr, "busy": [[q, t_tr] for q in qs], "qs": qs})
            ev.append({"t": "move", "dur": d, "busy": [[q, d] for q in qs], "moves": mv})
            ev.append({"t": "drop", "dur": t_tr, "busy": [[q, t_tr] for q in qs], "qs": qs})
        elif e['kind'] == 'gate':
            pairs = [list(map(int, p)) for p in e['pairs']]
            gated = {q for p in pairs for q in p}
            ev.append({"t": "cz", "dur": T_2Q, "busy": [[q, T_2Q] for q in sorted(gated)],
                       "pairs": pairs})
            idle = sorted(q for q, p in grid.items() if p[1] >= 0 and q not in gated)
            if idle:
                ev.append({"t": "xtalk", "busy": [], "qs": idle})
    for lay in layers_1q:
        ev.append({"t": "1q", "dur": T_1Q, "busy": [[int(q), T_1Q] for q in lay], "n": len(lay)})
    traps = [list(powermove_phys((x, y))) for x in range(row) for y in range(-8 * row - 2, row)]
    return {"init": init, "events": ev, "traps": traps, "site_capacity": 2}


def powermove_1q_layers(circuit):
    """PowerMove's own 1q layering (run_powermove.count_1q_layers), returning
    the atom list of each layer instead of only their count."""
    from qiskit.converters import circuit_to_dag
    from qiskit.dagcircuit import DAGOutNode
    dag = circuit_to_dag(circuit)
    front = set(dag.front_layer())
    blocks, types = [], []
    def kind(op):
        return 'CZ' if op.op.num_qubits == 2 else '1q'
    while dag.size():
        cur = types[-1] if types else None
        if cur is None:
            cur = kind(next(iter(front)))
            types.append(cur); blocks.append([])
        added = 0
        for op in list(front):
            if kind(op) != cur:
                continue
            args = [circuit.find_bit(q).index for q in op.qargs]   # global index, not per-register
            if op.op.name in ('measure', 'barrier'):
                succ = list(dag.successors(op)); dag.remove_op_node(op); front.discard(op)
                for s in succ:
                    if not isinstance(s, DAGOutNode) and not list(dag.op_predecessors(s)):
                        front.add(s)
                added += 1
                continue
            if cur != 'CZ' and args[0] in blocks[-1]:
                continue
            added += 1
            blocks[-1].append(tuple(args) if len(args) > 1 else args[0])
            succ = list(dag.successors(op))
            dag.remove_op_node(op); front.discard(op)
            for s in succ:
                if not isinstance(s, DAGOutNode) and not list(dag.op_predecessors(s)):
                    front.add(s)
        if not added:
            cur = kind(next(iter(front)))
            types.append(cur); blocks.append([])
    return [b for b, t in zip(blocks, types) if t != 'CZ' and b]
