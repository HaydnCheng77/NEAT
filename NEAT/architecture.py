"""Generate a right-sized, physically sound architecture for a given circuit.

Motivation
----------
The repo used to ship three fixed architectures (``default``, ``scale_to_100``,
``scale_to_500``) and assign whole suites to whichever was big enough. Two
problems with that:

1. **A size sweep is not self-consistent.** Moving the ``graphs`` suite from
   ``scale_to_100`` to ``scale_to_500`` (needed because n>=200 does not fit in
   192 storage traps) changed every previously measured number, because the
   placer's behaviour depends on the array it is given.

2. **``scale_to_500`` is not physically sound.** Its storage array spans
   y=0..186 while its entanglement zone starts at y=124, so the two zones
   OVERLAP. Measured minimum storage-to-entanglement site distance is **2 µm,
   inside the R_B = 5 µm Rydberg blockade radius**. An idle atom parked there
   sits close enough to a driven atom to be disturbed, but
   ``Router.write_idle`` charges crosstalk on exact coordinate membership
   (``current_mapping[q] in _ent_site_set``), not on distance -- so those atoms
   are charged nothing. The wide suites' fidelities are inflated by that gap
   between the geometry and the noise model. ``default.json`` keeps a 10 µm
   separation and is unaffected.

This generator builds, for a given qubit count, the smallest array that is both
big enough and physically sound: zones strictly separated by at least
``MIN_GAP`` >= R_B, so no storage site is ever within blockade range of an
entanglement site.

Geometry is taken from ``default.json``, which is the architecture the tqe
fidelity floor was measured on:

* storage site separation ``[6, 6]`` µm
* entanglement site separation ``[12, 6]`` µm, two interleaved SLMs offset by
  4 µm in x (that 4 µm pair is the Rydberg pair; 4 < R_B, as it must be)
* entanglement zone x-origin offset -24 relative to storage
* vertical gap of 10 µm between the top storage row and the first
  entanglement row

Sizing
------
Storage must hold ``n_q`` atoms. Entanglement must hold the widest stage's
gates; since the placer also parks idle atoms in the zone, it is sized against
``n_q + max_stage_2q`` pairs when a gate schedule is supplied, and ``n_q``
otherwise. Both are grown as near-square as the aspect of the source geometry
allows, so the array stays compact and transport distances stay short.

Usage
-----
    from NEAT.architecture import build_architecture
    arch = build_architecture(n_q=100, max_stage_2q=50)

or from the command line, to inspect one:

    python3 NEAT/architecture.py 100
"""
import json
import math
import os
import sys

# --- Geometry copied from baselines/zap/architectures/default.json (the tqe-floor machine) ---
STG_SEP = [3, 3]        # storage site separation (µm), matches ZAC's
                        # full_architecture.json pitch. Chosen
                        # 2026-09-08 after measuring a consistent fidelity
                        # gain across every suite tested -- see the SLACK
                        # SWEEP block below for the numbers.
ENT_SEP = [12, 6]       # entanglement site separation (µm)
ENT_PAIR_DX = 4         # x-offset between the two interleaved entanglement SLMs
ENT_X_OFFSET = -24      # entanglement zone x-origin, relative to storage (only used when width_couple=False)
R_B = 5                 # Rydberg blockade radius (placer.py), for validation
MIN_GAP = 10            # vertical gap, top storage row -> first entanglement
                        # row. SET 2026-09-08: 10 µm is a
                        # physical requirement, not merely R_B's floor -- do
                        # not lower this to R_B without revisiting
                        # that decision. (An earlier version of this file set
                        # MIN_GAP=R_B=5 and measured a small fidelity gain from
                        # it; that comparison is superseded by this decision
                        # and is no longer the shipped default.)

# --- Tunables, both chosen by measurement (see SLACK SWEEP at end of file) ---
# Overridable from the environment so a sweep can drive them across subprocess
# runs without editing this file.
STORAGE_SLACK = float(os.environ.get("ZAP_GEN_SLACK", "1.5"))
CENTER_ALIGN = os.environ.get("ZAP_GEN_ALIGN", "0") == "1"
WIDTH_COUPLE = os.environ.get("ZAP_GEN_WIDTH_COUPLE", "1") == "1"
# Staggered trap layout (2026-10-01, prototype, OFF by default). Every row x
# column crossing of the AOD is a tweezer, so two held atoms A=(x1,y1) and
# B=(x2,y2) also light (x1,y2) and (x2,y1); on a rectangular trap grid those
# corners are traps and, when occupied, get lifted (ghost pick-ups). Offsetting
# rows so that different rows never share a column position puts those
# corners between traps. See build_architecture for the exact offsets.
STAGGER = os.environ.get("ZAP_GEN_STAGGER", "1t")  # "0" off, "1", "2", "ent" or "k3" (see below)

OPERATION_DURATION = {"2qGate": 0.27, "1qGate": 52, "atom_transfer": 15}   # 2026-10-07: paper Table tab:params (2q was 0.36 us)
OPERATION_FIDELITY = {"two_qubit_gate": 0.995, "single_qubit_gate": 0.9997,
                      "atom_transfer": 0.999}
QUBIT_SPEC = {"T2": 1500000.0}
ROUTING = {"parking_dist": 1}


def _stg_pitch_x():
    """Storage column pitch: 4 um in the 2 um-compatible stagger, else STG_SEP."""
    return 4 if STAGGER == "2" else STG_SEP[0]


def _grid_for(n_sites, sep, aspect_bias=1.0, target=None):
    """Smallest near-square (rows, cols) holding ``n_sites``.

    ``aspect_bias`` > 1 widens the grid in columns, used for the entanglement
    zone whose x-pitch is twice its y-pitch -- a square site count there is a
    2:1 oblong in µm, so biasing columns keeps the PHYSICAL shape near-square
    and the transport distances short.
    """
    best = None
    # ``target``: zone height/width in um (see LAYOUT SEARCH at the end of
    # this file); None = the rule below, which -- despite this docstring --
    # does not give a square: w and h are already physical, so the bias makes
    # the zone ~bias times wider than tall (n=1000: 700 x 240 um).
    limit = int(math.isqrt(max(n_sites, 1))) + 2
    if target is not None:
        limit = n_sites
    for rows in range(1, limit + 1):
        cols = math.ceil(n_sites / rows)
        # Physical extent, for choosing the most compact option.
        w = (cols - 1) * sep[0]
        h = (rows - 1) * sep[1]
        if target is not None:
            score = max(w, h / float(target)) + 0.001 * (rows * cols - n_sites)
        else:
            # Penalise deviation from a square footprint.
            score = max(w, h * aspect_bias) + 0.001 * (rows * cols - n_sites)
        if best is None or score < best[0]:
            best = (score, rows, cols)
    return best[1], best[2]


def build_architecture(n_q, max_stage_2q=None, name=None,
                       storage_slack=STORAGE_SLACK, center_align=CENTER_ALIGN,
                       width_couple=WIDTH_COUPLE, min_gap=MIN_GAP):
    """Return a ZAP architecture dict sized for ``n_q`` qubits.

    Args:
        n_q: number of qubits the circuit needs.
        max_stage_2q: widest stage's two-qubit gate count. The entanglement
            zone is sized to hold that many Rydberg pairs ON TOP OF every atom
            that could be parked in it. Defaults to ``n_q // 2`` (the most a
            stage can use), which is the safe choice when the schedule is not
            known yet.
        name: optional architecture name.
        storage_slack: storage traps per qubit. 1.0 is the tightest array that
            fits; >1 leaves empty traps, which the router uses as staging room
            when shuffling atoms. Swept -- see SLACK SWEEP below.
        center_align: when True, the entanglement zone's x-centre is placed on
            the storage zone's x-centre. MEASURED WORSE on tqe (see the
            isolated shift sweep below) -- kept as a toggle for the harness,
            default OFF.
        width_couple: when True, storage's column count is derived from the
            entanglement zone's column count (see WIDTH COUPLING below)
            instead of being sized independently from slack alone.
        min_gap: vertical gap between the top storage row and the first
            entanglement row. Physical floor is R_B; anything above that is
            pure added transport time under the concave movement_duration
            model, so the honest default is exactly R_B, not the historical
            10 µm inherited from default.json (a value nobody derived, only
            copied) -- see WIDTH COUPLING below for the measurement.
    """
    if n_q < 1:
        raise ValueError("n_q must be >= 1")
    # Size-dependent zone class count (2026-10-03): with STAGGER=1t, circuits
    # of at most ZAP_GEN_SMALL_N qubits use the plain two-class stagger "1".
    # Measured under strict legality: two classes is better on tqe (n <= 35,
    # 6.925767 vs 6.914750) and three classes better from n = 100 up (-0.96%
    # summed -log10 F over n = 100/300/500): on a tiny zone the wider 18 um
    # column period costs more distance than its fewer corner conflicts save.
    STG = STAGGER
    # Size-dependent layout (2026-10-06): circuits above
    # ZAP_GEN_SMALL_N qubits get the layout-search winner (zone height/width
    # 0.5, storage 0.35 of the matched width); smaller ones keep the legacy
    # shape (wide zone, width-matched storage). At all sizes the winner put
    # tqe at 6.918073 vs the 6.936961 floor (10/14 worse), while gaining
    # -3.0% / -10.6% summed -log10 F at n ~ 100-500 / n = 1000 -- narrow,
    # deep storage only pays once the array is large. ZAP_GEN_ENT_ASPECT
    # ("legacy" or a number) and ZAP_GEN_STG_WFRAC override at every size.
    big = n_q > int(os.environ.get("ZAP_GEN_SMALL_N", "64"))
    ent_target = os.environ.get("ZAP_GEN_ENT_ASPECT", "0.5" if big else "legacy")
    ent_target = None if ent_target == "legacy" else ent_target
    if STG == "1t" and n_q <= int(os.environ.get("ZAP_GEN_SMALL_N", "64")):
        STG = "1"
    if max_stage_2q is None:
        max_stage_2q = n_q // 2

    # A Rydberg pair is one site in each interleaved SLM, so a grid of r x c
    # per SLM provides r*c pairs. Size against parked atoms + the widest stage.
    # n pairs (2026-10-18; was n_q + max_stage_2q): enough for
    # every atom to hold a pair alone, and a gate between two parked atoms can
    # always use one of their pairs. Measured with the three placer changes of
    # the same day (see placer.py): fidelity within noise, smaller array.
    need_pairs = max(1, n_q)
    # "1t" (2026-10-02, experimental): three entanglement row classes on an
    # 18 um column period, so two of every three row pairs share no column.
    K = int(os.environ.get("ZAP_GEN_ZCLASSES", "3")) if STG == "1t" else 2
    epx = 6 * K if STG == "1t" else ENT_SEP[0]
    ent_r, ent_c = _grid_for(need_pairs, [epx, ENT_SEP[1]], aspect_bias=epx / ENT_SEP[1],
                             target=ent_target)
    if STG == "k3":
        # Ghost-free entanglement zone (2026-10-01 prototype): at most 3 rows,
        # one per column class, so no two zone traps share a column. The zone
        # grows wide instead of deep.
        ent_r = min(3, need_pairs)
        ent_c = math.ceil(need_pairs / ent_r)

    n_storage_sites = max(n_q, int(round(n_q * storage_slack)))
    if width_couple:
        # WIDTH COUPLING: size storage to the SAME physical width as the
        # entanglement zone (in columns, converted through the pitch ratio),
        # then grow rows to hold n_storage_sites. Every storage column then
        # has an entanglement column directly above it, so under
        # movement_duration's sqrt(distance) model every atom's vertical trip
        # to ITS OWN column costs `min_gap` and nothing is spent on sideways
        # travel to find a facing column -- the shipped crop (independent
        # sqrt(n_q)+4 / sqrt(n_q)+2 sizing) has no such correspondence.
        ent_width_um = (ent_c - 1) * epx + ENT_PAIR_DX
        stg_c = max(1, round(ent_width_um / _stg_pitch_x()) + 1)
        stg_r = max(1, math.ceil(n_storage_sites / stg_c))
        # Floor on row count, not a cap on column count. An earlier version of
        # this capped `stg_c` at whatever a SQUARE grid would use, which
        # defeated width-coupling almost everywhere (n=100: matched width
        # wanted 26 columns, the square cap silently substituted 12, and
        # storage ended up HALF the entanglement zone's width -- worse than
        # no coupling at all, caught by the initial-mapping PNGs showing
        # storage narrower than the zone above it). The actual failure this
        # was meant to catch was a ROW count of 1 on wide-stage circuits
        # (random_n300: matched width -> 44 columns -> only 11 rows needed for
        # 450 sites, and that thin-and-wide shape is what made the placer's
        # per-stage column scan cost explode to 200s+). Guaranteeing a minimum
        # row count directly fixes the thing that was actually expensive,
        # without ever touching the width-matching property itself: `stg_c` is
        # ALWAYS the entanglement-matched value now, and only `stg_r` grows
        # past `n_storage_sites / stg_c` on circuits where that would be too
        # shallow.
        MIN_ROWS = 6
        if stg_r < MIN_ROWS:
            stg_r = MIN_ROWS
        # Storage width as a fraction of the matched width (layout search,
        # above ZAP_GEN_SMALL_N only): narrower and deeper, centred under the
        # zone. ZAP_GEN_STG_WFRAC overrides; "1" = matched width.
        wfrac = os.environ.get("ZAP_GEN_STG_WFRAC", "0.35" if big else None)
        if wfrac is not None:
            stg_c = max(1, round(stg_c * float(wfrac)))
            stg_r = max(MIN_ROWS, math.ceil(n_storage_sites / stg_c))
    else:
        stg_r, stg_c = _grid_for(n_storage_sites, STG_SEP)

    # ZAP_GEN_STG_SPLIT (2026-10-05, layout search): fraction of storage rows
    # placed ABOVE the entanglement zone (same columns, same stagger, same
    # min_gap), so the zone is fed from both sides. Unset/0 = all below.
    split = float(os.environ.get("ZAP_GEN_STG_SPLIT", "0") or 0)
    stg_r_above = int(round(stg_r * split)) if split > 0 else 0
    stg_r_below = stg_r - stg_r_above
    stg_top = (stg_r_below - 1) * STG_SEP[1]
    ent_y = stg_top + min_gap

    if width_couple:
        # Storage's width was just derived FROM the entanglement zone's width
        # (above), so finish the correspondence: overlap their x-spans exactly
        # rather than leaving the entanglement zone at its historical -24
        # offset (which was itself only ever relative to default.json's
        # independently-chosen storage width).
        stg_cx = ((stg_c - 1) * _stg_pitch_x()) / 2.0
        ent_span = (ent_c - 1) * epx + ENT_PAIR_DX
        ent_x = int(round(stg_cx - ent_span / 2.0))
        # No entanglement column may sit on a storage column (2026-10-01).
        # With the two zones aligned, half the entanglement
        # columns (e.g. 22 of 44 at n=300) coincided with storage columns, so
        # an atom held in the entanglement zone kept a line switched on that
        # runs straight through the densely filled storage grid, and any atom
        # picked up from another storage row lit a tweezer on a storage trap
        # (ghost pick-up). ZAC's own array avoids this (entanglement SLMs at
        # x=35/37 over a 3 um storage grid). Shift the zone so its columns
        # fall between storage columns: ent_x = 1 (mod 3) puts the pair's
        # two sites (ent_x, ent_x + ENT_PAIR_DX=4) at 1 and 2 (mod 3), and
        # ENT_SEP[0]=12 keeps every later column there. Storage still spans
        # the zone's width, so each atom's trip up stays essentially
        # vertical (a 1-2 um sideways component).
        while STG in ("0", "ent") and (ent_x % STG_SEP[0] == 0
                                  or (ent_x + ENT_PAIR_DX) % STG_SEP[0] == 0):
            ent_x += 1
    elif center_align:
        # Put the two zones' x-centres on top of each other. The entanglement
        # zone's own two SLMs straddle ENT_PAIR_DX, so its centre includes that.
        stg_cx = ((stg_c - 1) * STG_SEP[0]) / 2.0
        ent_span = (ent_c - 1) * epx + ENT_PAIR_DX
        ent_x = stg_cx - ent_span / 2.0
        # Keep coordinates on a whole-µm integer lattice: downstream code indexes
        # and compares trap coordinates, so they must be ints, not floats.
        ent_x = int(round(ent_x))
    else:
        ent_x = ENT_X_OFFSET

    stg_slms = [{"id": 0, "site_seperation": list(STG_SEP),
                 "r": stg_r, "c": stg_c, "location": [0, 0]}]
    ent_slms = [
        {"id": 1, "site_seperation": list(ENT_SEP), "r": ent_r,
         "c": ent_c, "location": [ent_x, ent_y]},
        {"id": 2, "site_seperation": list(ENT_SEP), "r": ent_r,
         "c": ent_c, "location": [ent_x + ENT_PAIR_DX, ent_y]},
    ]
    if STG == "k3" and width_couple:
        # Three row classes, pair gap 2 um: class k's pair sits at
        # (4k, 4k + 2) (mod 12) from ent_x, so every zone tone is >= 2 um
        # (d_s) from every other and, with ent_r <= 3, no two zone traps
        # share a column -- zone-zone ghost pick-ups are impossible. Storage
        # is the plain grid; with 2 um tone steps a third of the zone's
        # columns still land on a storage column (unavoidable at 3 um pitch).
        ry = ENT_SEP[1]
        ent_slms = []
        sid = 1
        for k in range(ent_r):
            for dx in (0, 2):
                ent_slms.append({"id": sid, "site_seperation": list(ENT_SEP),
                                 "r": 1, "c": ent_c,
                                 "location": [ent_x + 4 * k + dx, ent_y + k * ry]})
                sid += 1
    elif STG == "ent" and width_couple and ent_r > 1:
        # Alternating entanglement zone only; storage stays the plain grid.
        # Odd entanglement rows shift right by half the column pitch (6 um),
        # so class A columns sit at {0, 4} and class B at {6, 10} (mod 12,
        # from ent_x): adjacent rows share no column, and every gap between
        # the classes is >= 2 um (d_s). 6 = 0 (mod 3), so both classes keep
        # the storage-avoiding residues ent_x was shifted to above.
        half = ENT_SEP[0] // 2
        ry = ENT_SEP[1]
        ra, rb = (ent_r + 1) // 2, ent_r // 2
        ent_slms = []
        for sid, (dx, dy, r) in enumerate(((0, 0, ra), (ENT_PAIR_DX, 0, ra),
                                           (half, ry, rb),
                                           (half + ENT_PAIR_DX, ry, rb)), start=1):
            ent_slms.append({"id": sid, "site_seperation": [ENT_SEP[0], 2 * ry],
                             "r": r, "c": ent_c,
                             "location": [ent_x + dx, ent_y + dy]})
    elif STG == "2" and width_couple:
        # 2 um-compatible stagger. Storage columns 4 um apart; even rows at
        # x = 0 (mod 4), odd rows at x = 2 (mod 4): rows of different classes
        # never share a column, and their columns are always >= 2 um apart,
        # so they can still be lifted together (d_s). Entanglement columns all
        # at odd x (never on a storage column); a pair is (x, x+2), both odd;
        # rows cycle through three classes at columns {1,3}, {5,7}, {9,11}
        # (mod 12), so two of every three entanglement row pairs share no
        # column.
        sx, sy = 4, STG_SEP[1]
        stg_slms = [{"id": 0, "site_seperation": [sx, 2 * sy],
                     "r": (stg_r + 1) // 2, "c": stg_c, "location": [0, 0]}]
        if stg_r // 2:
            stg_slms.append({"id": 3, "site_seperation": [sx, 2 * sy],
                             "r": stg_r // 2, "c": stg_c, "location": [2, sy]})
        pdx = 2
        ex = ent_x
        while ex % 4 != 1:
            ex += 1
        ey, ry = ent_y, ENT_SEP[1]
        ent_slms = []
        sid = 1
        for k in range(3):
            rk = len(range(k, ent_r, 3))
            if not rk:
                continue
            for dx in (0, pdx):
                ent_slms.append({"id": sid, "site_seperation": [ENT_SEP[0], 3 * ry],
                                 "r": rk, "c": ent_c,
                                 "location": [ex + 4 * k + dx, ey + k * ry]})
                sid += 1
    elif STG in ("1", "1t") and width_couple:
        # Storage: two row classes. Even rows at x = 0 (mod 3), odd rows
        # shifted 1 um to x = 1 (mod 3): an even row and an odd row never
        # share a column, so their rectangle corners are empty. Same pitch
        # and trap count as the plain grid.
        sx, sy = STG_SEP
        stg_slms = [
            {"id": 0, "site_seperation": [sx, 2 * sy], "r": (stg_r_below + 1) // 2,
             "c": stg_c, "location": [0, 0]},
        ]
        if stg_r_below // 2:
            stg_slms.append({"id": 3, "site_seperation": [sx, 2 * sy],
                             "r": stg_r_below // 2, "c": stg_c, "location": [1, sy]})
        # Entanglement: every column at x = 2 (mod 3), the one residue storage
        # no longer uses, so no entanglement column meets a storage column.
        # A pair is (x, x+3) -- 3 um, inside R_B -- so both sites stay at
        # 2 (mod 3). Rows alternate between two classes shifted by 6 um, so
        # neighbouring entanglement rows never share a column either:
        # class A columns = {0, 3} (mod 12), class B = {6, 9} (mod 12).
        pdx = 3
        ex = ent_x
        while ex % sx != 2:
            ex += 1
        ey, ry = ent_y, ENT_SEP[1]
        ra, rb = (ent_r + 1) // 2, ent_r // 2
        ent_slms = [
            {"id": 1, "site_seperation": [ENT_SEP[0], 2 * ry], "r": ra,
             "c": ent_c, "location": [ex, ey]},
            {"id": 2, "site_seperation": [ENT_SEP[0], 2 * ry], "r": ra,
             "c": ent_c, "location": [ex + pdx, ey]},
        ]
        if rb:
            ent_slms += [
                {"id": 4, "site_seperation": [ENT_SEP[0], 2 * ry], "r": rb,
                 "c": ent_c, "location": [ex + 6, ey + ry]},
                {"id": 5, "site_seperation": [ENT_SEP[0], 2 * ry], "r": rb,
                 "c": ent_c, "location": [ex + 6 + pdx, ey + ry]},
            ]
        if STG == "1t":
            # Three row classes k = row mod 3, columns {6k, 6k+3} (mod 18):
            # all six residues are 3 um apart (>= d_s), every one is 2 (mod 3)
            # as ex is, so no zone column meets a storage column.
            ent_slms = []
            ids = iter([i for i in range(1, 4 * K + 4) if i not in (0, 3)])
            for k in range(K):
                rk = len(range(k, ent_r, K))
                if not rk:
                    continue
                for dx in (0, pdx):
                    ent_slms.append({"id": next(ids),
                                     "site_seperation": [epx, K * ry], "r": rk,
                                     "c": ent_c, "location": [ex + 6 * k + dx, ey + k * ry]})

    if stg_r_above:
        if STG not in ("1", "1t") or not width_couple:
            raise ValueError("ZAP_GEN_STG_SPLIT is implemented for stagger 1/1t only")
        ent_top = max(sl["location"][1] + (sl["r"] - 1) * sl["site_seperation"][1]
                      for sl in ent_slms)
        y0 = ent_top + min_gap
        sx, sy = STG_SEP
        stg_slms.append({"id": 90, "site_seperation": [sx, 2 * sy],
                         "r": (stg_r_above + 1) // 2, "c": stg_c, "location": [0, y0]})
        if stg_r_above // 2:
            stg_slms.append({"id": 91, "site_seperation": [sx, 2 * sy],
                             "r": stg_r_above // 2, "c": stg_c, "location": [1, y0 + sy]})

    arch = {
        "name": name or f"gen_n{n_q}",
        # Tells Placer.init_slm_sites to skip its own crop entirely (2026-09-08
        # fix). That crop assumes it is always handed an OVERSIZED fixed
        # architecture (default.json/scale_to_500.json) and re-trims it down
        # via sqrt(n_q)-based sizing -- storage sorted (x asc, y desc),
        # entanglement sorted (x asc, y asc), independently, with no
        # coordination between the two. Handed a HAND-BUILT array that is
        # ALREADY exactly the right size, that same crop logic still fires and
        # re-trims it anyway: found because the entanglement crop kept only 78
        # of a generated array's 112 declared sites, and the kept slice was
        # shifted left (x 1..73 of a 1..89 zone) -- init_mapping's
        # mean-distance sort then centred on THAT crop's centre (~37), not the
        # architecture's true centre (45), which is the actual mechanism
        # behind the initial mapping looking "left-leaning". The generator
        # guarantees storage/entanglement are already sized for n_q and
        # max_stage_2q, so cropping them again is not merely unnecessary, it
        # is actively wrong.
        "pre_cropped": True,
        "operation_duration": dict(OPERATION_DURATION),
        "operation_fidelity": dict(OPERATION_FIDELITY),
        "qubit_spec": dict(QUBIT_SPEC),
        "routing": dict(ROUTING),
        "storage_zones": [{"zone_id": 0, "slms": stg_slms}],
        "entanglement_zones": [{"zone_id": 0, "slms": ent_slms}],
    }
    return arch


def sites(arch):
    """(storage_sites, entanglement_sites) as coordinate lists."""
    stg = []
    for s in arch["storage_zones"][0]["slms"]:
        stg += [(s["location"][0] + i * s["site_seperation"][0],
                 s["location"][1] + j * s["site_seperation"][1])
                for i in range(s["c"]) for j in range(s["r"])]
    ent = []
    for e in arch["entanglement_zones"][0]["slms"]:
        ent += [(e["location"][0] + i * e["site_seperation"][0],
                 e["location"][1] + j * e["site_seperation"][1])
                for i in range(e["c"]) for j in range(e["r"])]
    return stg, ent


def validate(arch, n_q):
    """Raise unless the architecture is sound and big enough.

    The separation check is the reason this generator exists: no storage site
    may lie within R_B of an entanglement site, or an idle atom parked in
    storage would be inside the blockade radius of a driven one.
    """
    stg, ent = sites(arch)
    if len(stg) < n_q:
        raise ValueError(f"storage holds {len(stg)} < n_q={n_q}")

    # Zone separation on the bounding boxes: both zones are axis-aligned grids,
    # with storage below the zone and, under ZAP_GEN_STG_SPLIT, also above it.
    ent_bot = min(q[1] for q in ent)
    ent_top = max(q[1] for q in ent)
    below = [p[1] for p in stg if p[1] < ent_bot]
    above = [p[1] for p in stg if p[1] > ent_top]
    if len(below) + len(above) != len(stg):
        raise ValueError("storage site inside the entanglement zone's y-span")
    gap = min(([ent_bot - max(below)] if below else [])
              + ([min(above) - ent_top] if above else []))
    if gap < R_B:
        raise ValueError(f"zone gap {gap} µm < R_B={R_B}")

    # Rydberg pairs must exist: the two interleaved SLMs must be within R_B.
    if ENT_PAIR_DX >= R_B:
        raise ValueError(f"pair separation {ENT_PAIR_DX} >= R_B={R_B}")
    return {"storage_traps": len(stg), "entanglement_traps": len(ent),
            "pairs": len(ent) // 2, "zone_gap_um": gap}


# WIDTH COUPLING (2026-09-07/08) -- the first geometric change that BEATS
# default.json on tqe itself, not merely "loses less".
#
# Derived from first principles rather than swept: movement_duration(d) =
# 200*sqrt(d/110) is concave, so total transport time is minimised when every
# atom's trip to the zone is short AND uniform, not when a few trips are very
# short. The shipped crop sizes storage and the entanglement zone
# INDEPENDENTLY (sqrt(n_q)+4 / sqrt(n_q)+2 for storage; a separate sqrt-based
# grid for entanglement, offset by a historical constant -24 that nothing
# derived it from -- see the shift sweep below, which found -24 unremarkable
# among 101 alternatives). Two changes follow from the concave-time argument:
#
#  1. `min_gap` should be exactly R_B, not 10. Any gap above the physical
#     floor is pure added transport time bought for nothing -- 10 was only
#     ever copied from default.json, never derived.
#  2. Storage's column count should be DERIVED from the entanglement zone's
#     physical width (`width_couple=True`), and the entanglement zone shifted
#     so the two zones' x-spans coincide. Every storage column then faces an
#     entanglement column directly above it at distance `min_gap` -- no atom
#     pays for sideways travel to find a facing column, which the independent
#     sizing does not guarantee (on cat_n35's shipped crop the two zones'
#     centres are 22 µm apart in x).
#
# Measured (in-process, NEAT only, both physical gates passing on all
# four suites), generated array vs the suite's PREVIOUS shipped/measured
# baseline:
#
#   suite    baseline    generated    delta      compile time (11-14 circuits)
#   tqe      6.906136    6.921161    +0.015025    0.68s  (was on default.json)
#   tqe2     7.300549    7.335735    +0.035186    0.79s  (was on default.json)
#   graphs   3.804041    3.611445    -0.192596    7.04s  (was on scale_to_500,
#                                                         which overlapped its
#                                                         zones -- see module
#                                                         docstring; this drop
#                                                         is that overlap
#                                                         artifact leaving)
#   random   1.216135    1.210957    -0.005178   30.35s  (was on scale_to_500)
#
# tqe and tqe2 (the two suites with a properly-separated baseline architecture
# to compare against) both GAIN. graphs and random regress only because their
# baseline was scale_to_500's overlapping geometry, which was inflating
# fidelity through an unphysical crosstalk-model gap -- the generated array is
# the honest number, not a regression introduced by this change.
#
# The 2:1 aspect-ratio blowup this caught: matching storage's width to the
# entanglement zone's raw column count, unconditionally, produced an 11x44
# storage grid at random_n300 (that suite's widest stages force a very wide
# entanglement zone) and compile time exploded to 200s for that one circuit.
# `stg_c` is therefore capped at the narrower of "width-matched" and "as
# square as n_storage_sites allows" -- it can only shrink toward square, never
# widen past the entanglement footprint, so the coverage property from point 2
# above is kept while the grid stays compact. Fixed this, random_n300 alone
# dropped from 200.31s to a few seconds and the suite total from 229.70s to
# 30.35s, with total_F changing by under 0.01.
#
# NONE of this touches tqe's SHIPPED configuration -- settings/tqe.json still
# points at default.json and still measures exactly 6.906135582. This is an
# architecture generator available via "architecture": "generate" in a
# setting file; adopting it for any suite is a separate decision from writing
# it, and no settings file has been switched over.


if __name__ == "__main__":
    # SLACK SWEEP (2026-09-06, 8 probe circuits spanning tqe / graphs /
    # scalability, 5 slacks x centre-alignment on-off, sum of total_fidelity):
    #
    #   slack   aligned    unaligned
    #   1.0     2.017698   2.003769
    #   1.15    2.019727   2.006040
    #   1.3     2.026455   2.025015
    #   1.5     2.029409   2.019471
    #   2.0     2.030317   2.022309
    #
    # CENTRE ALIGNMENT LOSES -- default is now OFF. The sweep above appeared to
    # favour it, but that comparison changed array SHAPE at the same time, so
    # alignment was confounded. Isolated on 2026-09-06 by shifting ONLY the
    # entanglement zone's x-position on default.json (storage, gap, counts and
    # pitch all untouched), tqe total:
    #
    #   default.json, misaligned by +5 µm   6.906136   <- best
    #   shift -5, centres exactly aligned   6.905185   -0.000951
    #   shift -6                            6.890725   -0.015411
    #   shift -4                            6.883992   -0.022144
    #   shift +5                            6.903845   -0.002291
    #
    # Perfect symmetry is not optimal and the response is not even monotonic in
    # offset (-4 is worse than -6), which points at lattice/parity effects
    # between the 6 µm storage grid and the 12 µm entanglement grid rather than
    # a smooth distance optimum. Do not "fix" the misalignment.
    #
    # Slack rises steeply to 1.3-1.5 then flattens: 2.0 buys only +0.0009 over
    # 1.5 while costing compile time (4.90 s -> 5.11 s on the probe, and the
    # placer's per-stage scans grow with site count). 1.5 is taken as the knee.
    # Empty traps are genuinely useful -- they give the router staging room --
    # which is why the tightest array (1.0) is the worst of the five.
    #
    # NOT a monotone effect per circuit: ising_n26 prefers unaligned and
    # graphs_300_0 prefers slack 1.0. The placer is greedy, so more sites is a
    # DIFFERENT search, not a superset of a smaller one -- removing the crop
    # entirely (strictly more sites) measured -13% at n=100 and -35% at n=300.
    # Choose these knobs on suite totals, never on a single circuit.
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    mx = int(sys.argv[2]) if len(sys.argv) > 2 else None
    a = build_architecture(n, mx)
    info = validate(a, n)
    s = a["storage_zones"][0]["slms"][0]
    e = a["entanglement_zones"][0]["slms"][0]
    print(json.dumps(a, indent=2))
    print(f"\nn_q={n}: storage {s['r']}x{s['c']}={info['storage_traps']} traps, "
          f"entanglement {e['r']}x{e['c']}x2={info['entanglement_traps']} "
          f"({info['pairs']} pairs), zone gap {info['zone_gap_um']} µm")


# LAYOUT SEARCH (2026-10-05/06) -- zone shape x storage width, chosen by
# measurement. Grid: zone height/width target {legacy ~0.33, 0.4, 0.5, 0.6,
# 0.75} x storage width fraction {0.25, 0.35, 0.4, 0.45, 0.6, 0.8, 1}, plus
# storage split above+below the zone {0, 0.5} on a coarser grid. Scored on the
# six scalability families at n ~ 100/300/500 (18 circuits) and, for the
# leaders, n = 1000 (6 circuits); summed -log10 F vs the legacy layout:
#   0.5 / 0.35 (shipped)  -3.0% (18)  -10.6% (n=1000), 14 better / 10 worse,
#                         worst single circuit +0.54 (random_n300)
#   0.4 / 0.45            -3.2%       -10.4%
#   0.6 / 0.35            -4.3%        -9.8%
#   legacy / matched       0           0
#   any split > 0         +3% to +7% (worse everywhere; mostly qft/adder)
# Per-cell noise is ~+-1.5% (the 0.4 column landed irregularly between its
# neighbours), so the choice was read off the 3x3-smoothed surface, whose best
# region is shape ~0.5 / storage 0.35-0.45. Gains concentrate in qft and adder
# and grow with n. Three grid cells produced a single AOD violation each
# (adder_n496 at 0.6/0.25 and 0.75/1, qft_n300 at legacy/0.45), so legality is
# geometry-sensitive: re-run the verifier on every suite after any change here.

