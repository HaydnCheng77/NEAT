"""One AOD-legality checker for every compiler's event trace.

Applies the rules of NEAT/verifier.py to the shared event
format (experiments/end_to_end/events.py), so the violation counts of zap, NEAT, ZAC,
QMAP and PowerMove are comparable:

  double_occupancy   more atoms on a trap than it holds (1; PowerMove gate
                     sites hold a pair: 2)
  ghost              a pickup lights a row x column crossing on an atom not
                     being held (AOD rule d)
  ghost_cosite       ...on an atom sharing a held atom's exact coordinate
                     (PowerMove's grid does not resolve a pair site's two
                     positions; reported separately, not counted as illegal)
  drag               a move carries a lit crossing off an atom not being held
  tone_separation    two lit rows or columns closer than d_s (2 um) (rule c)
  crossing           two atoms of one parallel move cross or merge (rule b)
  held_crossing      a mover crosses a held, stationary atom's lines
  move_unheld        an atom moves without being held
  release_off_trap   an atom is released away from every trap
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from NEAT.transport import compatible_2D  # noqa: E402

RULES = ("double_occupancy", "ghost", "drag", "tone_separation", "crossing",
         "held_crossing", "move_unheld", "release_off_trap")


def check(trace, d_s=2.0):
    traps = {tuple(map(float, t)) for t in trace['traps']}
    cap2 = trace.get('site_capacity', 1) == 2
    def cap(p):
        return 2 if cap2 and p[1] >= 0 else 1
    pos = {int(q): tuple(map(float, p)) for q, p in trace['init'].items()}
    held = set()
    c = {r: 0 for r in RULES}
    c['ghost_cosite'] = 0
    first = {}
    def hit(rule, i, msg, n=1):
        c[rule] += n
        first.setdefault(rule, (i, msg))
    def occupancy(i, qs):
        cnt = {}
        for q, p in pos.items():
            if q not in held:
                cnt[p] = cnt.get(p, 0) + 1
        for q in qs:
            p = pos[q]
            if cnt.get(p, 0) > cap(p):
                hit('double_occupancy', i, f"q{q} at {p}")
    occupancy(-1, list(pos))
    for i, e in enumerate(trace['events']):
        t = e['t']
        if t == 'pick':
            held.update(e['qs'])
            if 'lines' in e:
                cols, rows = map(set, e['lines'])
            else:
                cols = {pos[h][0] for h in held}
                rows = {pos[h][1] for h in held}
            heldpos = {pos[h] for h in held}
            for q, p in pos.items():
                if q in held or p[0] not in cols or p[1] not in rows:
                    continue
                if p in heldpos:
                    c['ghost_cosite'] += 1
                else:
                    hit('ghost', i, f"q{q} at {p} lit by pickup of {sorted(e['qs'])[:6]}")
            for name, v in (("col", sorted(cols)), ("row", sorted(rows))):
                for lo, hi in zip(v, v[1:]):
                    if hi - lo < d_s - 1e-9:
                        hit('tone_separation', i, f"{name} lines {lo},{hi}")
        elif t == 'move':
            vecs = []
            for q, x0, y0, x1, y1 in e['moves']:
                if (x0, y0) != (x1, y1):
                    if q not in held:
                        hit('move_unheld', i, f"q{q}")
                    vecs.append((q, (x0, x1, y0, y1)))
            for a in range(len(vecs)):
                for b in range(a + 1, len(vecs)):
                    if not compatible_2D(vecs[a][1], vecs[b][1]):
                        hit('crossing', i, f"q{vecs[a][0]} and q{vecs[b][0]}")
            movers = {q for q, _ in vecs}
            # Lit lines are read off the held atoms: a converter's 'lines' only
            # describe a pickup, and a line moves with the atoms on it.
            if held:
                ends = {q: v for q, v in vecs}
                col_end, row_end = {}, {}
                for h in held:
                    x, y = pos[h]
                    v = ends.get(h, (x, x, y, y))
                    col_end.setdefault(x, set()).add(float(v[1]))
                    row_end.setdefault(y, set()).add(float(v[3]))
                heldpos = {pos[h] for h in held}
                for q, p in pos.items():
                    if (q not in held and p[0] in col_end and p[1] in row_end
                            and p not in heldpos
                            and (col_end[p[0]] != {p[0]} or row_end[p[1]] != {p[1]})):
                        hit('drag', i, f"q{q} at {p} dragged by a move")
            for h in held - movers:
                st = (pos[h][0], pos[h][0], pos[h][1], pos[h][1])
                for q, v in vecs:
                    if not compatible_2D(v, st):
                        hit('held_crossing', i, f"q{q} crosses held q{h}")
            for q, x0, y0, x1, y1 in e['moves']:
                pos[q] = (float(x1), float(y1))
        elif t == 'drop':
            for q in e['qs']:
                if pos[q] not in traps:
                    hit('release_off_trap', i, f"q{q} at {pos[q]}")
            held.difference_update(e['qs'])
            occupancy(i, e['qs'])
    illegal = sum(c[r] for r in RULES)
    return {"legal": illegal == 0, "violations": illegal, "counts": c,
            "first": {k: list(v) for k, v in first.items()}}
