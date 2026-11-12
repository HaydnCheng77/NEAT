"""AOD-legality verifier for a compiled ``results_code`` instruction trace.

Runs after compilation, like the Simulator and Animator, so it never counts
toward ``compilation_time``. It replays the trace once, tracking every atom's
position and which atoms the AOD holds (Activate .. Deactivate), and checks
every legality rule the repo's stand-alone checkers enforce, plus two they do
not:

  double_occupancy   two atoms on one position after a move      (verify_traces)
  gate_same_site     the two atoms of a 2q gate on one position   (verify_traces)
  crossing           two atoms moved in one instruction whose
                     tones cross or split (compatible_2D, d_s)    (verify_crossings)
  held_crossing      a mover incompatible with an atom the AOD
                     holds but does not move                      (verify_held)
  tone_separation    two distinct mover tones closer than d_s at
                     either end of a move                         (verify_tone_separation)
  aod_a_collective   AOD rule (a): one tone would split -- every
                     held atom, moving or not, counted per tone
                     pair                                         (verify_aod)
  aod_b_crossing     AOD rule (b): two tones cross                (verify_aod)
  aod_c_separation   AOD rule (c): distinct tones < d_s apart, at
                     both ends of a move and after every Activate (verify_aod)
  aod_d_ghost        AOD rule (d): an Activate lights a row x
                     column corner on an atom it does not mean to
                     lift (per lifted atom, per Activate)         (verify_aod)
  aod_e_drag         a move carries a lit corner off an atom the
                     AOD does not hold (per atom, per move)       (new)
  move_unheld        an atom is moved without being held          (new)
  release_off_trap   an atom is released away from every trap     (new)
  release_stuck      an atom is released while its row and column
                     both hold atoms staying held, so neither line
                     can be switched off                          (new)

Counting matches the stand-alone tools, so their totals and these agree.
The result is stored in ``results_code['verification']``:
``{'legal': bool, 'counts': {rule: n}, 'first': {rule: (instruction, msg)},
'd_s': float}``.
"""


def _compatible(a, b, ds):
    """``transport.model.compatible_2D`` with an explicit d_s."""
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    if a0 == b0:
        if a1 != b1:
            return False
    elif a0 < b0:
        if a1 >= b1:
            return False
    elif a1 <= b1:
        return False
    if a1 == b1 and a0 != b0:
        return False
    if a2 == b2:
        if a3 != b3:
            return False
    elif a2 < b2:
        if a3 >= b3:
            return False
    elif a3 <= b3:
        return False
    if a3 == b3 and a2 != b2:
        return False
    if ds:
        if a0 != b0 and abs(a0 - b0) < ds:
            return False
        if a1 != b1 and abs(a1 - b1) < ds:
            return False
        if a2 != b2 and abs(a2 - b2) < ds:
            return False
        if a3 != b3 and abs(a3 - b3) < ds:
            return False
    return True


def _axis(a0, a1, b0, b1, ds):
    """verify_aod's per-axis tone-pair rules: 'a' split, 'c' merge/close, 'b' cross."""
    out = []
    if a0 == b0:
        if a1 != b1:
            out.append("a")
        return out
    if a1 == b1:
        out.append("c")
        return out
    if (a0 < b0) != (a1 < b1):
        out.append("b")
    if abs(a0 - b0) < ds or abs(a1 - b1) < ds:
        out.append("c")
    return out


RULES = ("double_occupancy", "gate_same_site", "crossing", "held_crossing",
         "tone_separation", "aod_a_collective", "aod_b_crossing",
         "aod_c_separation", "aod_d_ghost", "aod_e_drag", "move_unheld",
         "release_off_trap", "release_stuck")
_AOD = {"a": "aod_a_collective", "b": "aod_b_crossing", "c": "aod_c_separation"}


class Verifier:
    """Replay ``results_code`` and count every AOD-legality violation."""

    def __init__(self, results_code: dict, traps, d_s: float = 2.0):
        self.results_code = results_code
        self.traps = {tuple(t) for t in traps}
        self.ds = float(d_s)
        self.counts = {r: 0 for r in RULES}
        self.first = {}
        self.verify()
        self.legal = not any(self.counts.values())
        results_code['verification'] = {
            'legal': self.legal, 'counts': dict(self.counts),
            'first': dict(self.first), 'd_s': self.ds}

    def _hit(self, rule, idx, msg, n=1):
        self.counts[rule] += n
        self.first.setdefault(rule, (idx, msg))

    def verify(self):
        ds = self.ds
        pos = {}          # atom -> (x, y)
        occ = {}          # (x, y) -> set of atoms
        held = set()
        for idx, inst in enumerate(self.results_code['instructions']):
            t = inst['type']
            if t == 'Init':
                for l in inst['locs']:
                    p = (l['x'], l['y'])
                    pos[l['id']] = p
                    occ.setdefault(p, set()).add(l['id'])
                for p, s in occ.items():
                    if len(s) > 1:
                        self._hit('double_occupancy', idx, f"init {sorted(s)} at {p}")
            elif t == 'Activate':
                new = [l['id'] for l in inst['locs']]
                held.update(new)
                cols = {pos[q][0] for q in held}
                rows = {pos[q][1] for q in held}
                lifted = [q for q, (x, y) in pos.items()
                          if q not in held and x in cols and y in rows]
                if lifted:
                    self._hit('aod_d_ghost', idx,
                              f"q{lifted[0]} at {pos[lifted[0]]} lifted by Activate {sorted(new)}",
                              len(lifted))
                for name, vals in (("col", cols), ("row", rows)):
                    v = sorted(vals)
                    for lo, hi in zip(v, v[1:]):
                        if hi - lo < ds:
                            self._hit('aod_c_separation', idx,
                                      f"{name} tones {lo},{hi} after Activate")
            elif t == 'Deactivate':
                stay = held - {l['id'] for l in inst['locs']}
                cols = {pos[q][0] for q in stay}
                rows = {pos[q][1] for q in stay}
                for l in inst['locs']:
                    q = l['id']
                    if tuple(pos[q]) not in self.traps:
                        self._hit('release_off_trap', idx, f"q{q} released at {pos[q]}")
                    if pos[q][0] in cols and pos[q][1] in rows:
                        self._hit('release_stuck', idx,
                                  f"q{q} at {pos[q]}: row and column both stay on")
                    held.discard(q)
            elif t in ('BigMove', 'Park'):
                vec = {l['id']: (l['x_begin'], l['x_end'], l['y_begin'], l['y_end'])
                       for l in inst['locs']}
                movers = {q: v for q, v in vec.items() if (v[0], v[2]) != (v[1], v[3])}
                for q in movers:
                    if q not in held:
                        self._hit('move_unheld', idx, f"q{q} moved without being held")
                # crossing: movers pairwise
                mlist = list(movers.items())
                for i in range(len(mlist)):
                    for j in range(i + 1, len(mlist)):
                        if not _compatible(mlist[i][1], mlist[j][1], ds):
                            self._hit('crossing', idx,
                                      f"q{mlist[i][0]} vs q{mlist[j][0]}")
                # held_crossing: movers vs stationary held atoms
                stationary = [q for q in held if q not in movers]
                for mq, mv in mlist:
                    for h in stationary:
                        hx, hy = pos[h]
                        if not _compatible(mv, (hx, hx, hy, hy), ds):
                            self._hit('held_crossing', idx, f"moved q{mq} while q{h} held")
                # tone separation among movers, both ends
                if len(mlist) > 1:
                    for k0, k1 in ((0, 2), (1, 3)):
                        for k, name in ((k0, 'col'), (k1, 'row')):
                            v = sorted({mv[k] for _q, mv in mlist})
                            for lo, hi in zip(v, v[1:]):
                                if hi - lo < ds:
                                    self._hit('tone_separation', idx,
                                              f"{name} tones {lo},{hi}")
                                    break
                # AOD rules a-c over every held tone (verify_aod)
                for k0, k1, name in ((0, 1, "col"), (2, 3, "row")):
                    tones = set()
                    for q in held:
                        if q in vec:
                            tones.add((vec[q][k0], vec[q][k1]))
                        else:
                            c = pos[q][0 if k0 == 0 else 1]
                            tones.add((c, c))
                    tl = sorted(tones)
                    for i in range(len(tl)):
                        for j in range(i + 1, len(tl)):
                            for r in _axis(tl[i][0], tl[i][1], tl[j][0], tl[j][1], ds):
                                self._hit(_AOD[r], idx, f"{t}: {name} tones {tl[i]} vs {tl[j]}")
                # drag: a lit corner resting on an unheld atom moves off it
                if held:
                    col_end = {}
                    row_end = {}
                    for q in held:
                        x, y = pos[q]
                        v = vec.get(q, (x, x, y, y))
                        col_end.setdefault(x, set()).add(v[1])
                        row_end.setdefault(y, set()).add(v[3])
                    for q, (x, y) in pos.items():
                        if (q not in held and x in col_end and y in row_end
                                and (col_end[x] != {x} or row_end[y] != {y})):
                            self._hit('aod_e_drag', idx,
                                      f"q{q} at {(x, y)} dragged by {t}")
                # apply the move, then check occupancy of the landing sites
                for q, v in vec.items():
                    old = pos[q]
                    occ[old].discard(q)
                    if not occ[old]:
                        del occ[old]
                for q, v in vec.items():
                    p = (v[1], v[3])
                    pos[q] = p
                    occ.setdefault(p, set()).add(q)
                for q in vec:
                    s = occ[pos[q]]
                    if len(s) > 1 and q == min(s):
                        self._hit('double_occupancy', idx, f"{sorted(s)} at {pos[q]}")
            elif t == '2qGate':
                for a, b in inst.get('gates', []):
                    if pos[a] == pos[b]:
                        self._hit('gate_same_site', idx, f"q{a}, q{b} at {pos[a]}")

    def summary(self) -> str:
        if self.legal:
            return "legal (0 violations)"
        bad = {r: n for r, n in self.counts.items() if n}
        return f"{sum(bad.values())} violations {bad}"
