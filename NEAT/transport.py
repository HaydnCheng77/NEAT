"""The single authoritative model of AOD transport.

``Placer`` decides where atoms should go; ``Router`` turns those decisions into
physically valid move instructions. Both need the same three facts about the
hardware — how long a move takes, whether two moves may run simultaneously, and
how a set of moves packs into waves — and both used to carry their own copy of
each. A copy that drifts turns the placer's cost model into a prediction of a
router that no longer exists, silently, with no test able to see it. Everything
here is therefore defined once and imported by both.

Nothing in this module holds state or knows about traps, qubits or stages: it
is pure geometry over move vectors ``(start_x, end_x, start_y, end_y)``.
"""

import bisect
import math

__all__ = ["movement_duration", "compatible_2D", "batch_moves", "absorb_move",
           "MIN_TONE_SEP", "set_min_tone_sep"]

# --- Minimum AOD tone separation (d_s), added 2026-09-27 -------------------
# Implication 3 of the hardware model: because the microscope generating the
# tweezers has finite optical resolution, two distinct rows or columns of the
# SAME array cannot be closer than ``d_s`` (2 um in the reference hardware).
# The AOD is one array, so every pair of atoms held in one wave must sit
# either on the SAME tone or at least ``d_s`` apart, on each axis, at BOTH
# ends of the move.
#
# ``compatible_2D`` used to enforce tone ORDERING only, never separation, so
# traces routinely emitted tones 1 um apart (769 cases on tqe, ~220k on
# scalability/random). Audited 2026-09-27.
#
# 0.0 disables the constraint and restores the exact pre-2026-09-27 predicate,
# which is what the historical fidelity floor was measured under.
MIN_TONE_SEP = 0.0


def set_min_tone_sep(ds):
    """Set the module-wide minimum AOD tone separation (um).

    Module-level rather than per-instance because ``compatible_2D`` is a pure
    module function bound as a staticmethod onto both ``Router`` and
    ``Placer`` (see the binding comments there) precisely so the hot path has
    no forwarding frame and no ``self`` lookup; threading a parameter through
    it would undo that. Set once from the architecture's ``routing`` block
    before placing/routing begins.
    """
    global MIN_TONE_SEP
    MIN_TONE_SEP = float(ds or 0.0)


def movement_duration(d: float) -> float:
    """Wall time (µs) to move an atom ``d`` µm.

    Concave (sqrt), and that shape drives real decisions: splitting a move into
    two legs of the same total distance costs MORE wall time than one leg, which
    is why the router's waypoint and chain machinery tries to keep legs long and
    few rather than merely numerous.
    """
    if d <= 0:
        return 0.0
    return 200 * ((d / 110) ** 0.5)


def compatible_2D(a, b) -> bool:
    """May move vectors ``a`` and ``b`` run in the same AOD wave?

    An AOD holds atoms on shared row and column "tones". A tone is a rigid line:
    it cannot be split mid-flight, and it cannot cross another tone. So two
    moves may run together only if their ordering survives on both axes —
    atoms starting on one line must end on one line, atoms starting on
    different lines must keep their relative order. This one predicate is the
    source of wave batching, collision parking, waypoint splitting and chaining
    alike.

    Vectors are ``(start_x, end_x, start_y, end_y)``.
    """
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    # Same eight tests as before, reassociated per axis: for each axis the pair
    # is compatible iff the start comparison and the end comparison agree in
    # sign, i.e. the tone ordering is preserved. Equality on one end forces
    # equality on the other (a tone cannot split); strict order on one end
    # forces the same strict order on the other (tones cannot cross).
    # Unpacking once and comparing the two orderings directly is measurably
    # cheaper than eight indexed comparisons at 38.8M calls/circuit, and is
    # exactly equivalent. Verified 2026-09-03 against the previous eight-test
    # implementation: exhaustive over all 3^8 = 6561 integer vector pairs, plus
    # 200k random float pairs — 0 mismatches. Re-run that check (both forms
    # side by side, compare on every pair) before editing this predicate.
    if a0 == b0:
        if a1 != b1:
            return False
    elif a0 < b0:
        if a1 >= b1:
            return False
    elif a1 <= b1:  # a0 > b0
        return False
    if a1 == b1 and a0 != b0:
        return False

    if a2 == b2:
        if a3 != b3:
            return False
    elif a2 < b2:
        if a3 >= b3:
            return False
    elif a3 <= b3:  # a2 > b2
        return False
    if a3 == b3 and a2 != b2:
        return False

    # d_s: distinct tones must clear MIN_TONE_SEP, on each axis and at both
    # ends. Ordering is already guaranteed above, so a nonzero gap here is a
    # genuine two-tone separation and not a sign question. Checked last, and
    # skipped entirely when the constraint is off, so the disabled path costs
    # one float compare on top of the original predicate -- this predicate runs
    # ~39M times per circuit on the wide suites.
    ds = MIN_TONE_SEP
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


def axis_compatible(a0, a1, b0, b1) -> bool:
    """One axis of ``compatible_2D``: tones ``a0 -> a1`` and ``b0 -> b1``
    keep their order (no split, no cross) and clear MIN_TONE_SEP at both
    ends. ``compatible_2D(a, b)`` is exactly
    ``axis_compatible(a[0], a[1], b[0], b[1]) and
    axis_compatible(a[2], a[3], b[2], b[3])`` (verified 2026-10-03)."""
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
    ds = MIN_TONE_SEP
    if ds:
        if a0 != b0 and abs(a0 - b0) < ds:
            return False
        if a1 != b1 and abs(a1 - b1) < ds:
            return False
    return True


def _duration(v):
    """Duration of one move vector ``(sx, tx, sy, ty)``."""
    return movement_duration(math.dist((v[0], v[2]), (v[1], v[3])))


def batch_moves(vectors):
    """Greedy first-fit packing of ``vectors`` into waves.

    Returns ``[[vectors_in_wave, bottleneck_duration_us], ...]``: append each
    vector to the first wave every member of which it is compatible with, else
    open a new one. This is the placer's model of what the hardware will do with
    a set of moves, and every cost estimate in the placer is built on it.

    Do NOT "improve" this to match the router's wave-selection algorithm more
    faithfully, or to LPT (longest-first) ordering. Both were tried against
    the router's PRIOR degree-sorted-MIS wave selection (since replaced
    2026-09-13 by `Router.find_chain_aware_independent_set` — a different
    algorithm, so this comparison's numbers are historical, not necessarily
    reproducible against the current router) and both measured net WORSE on
    tqe (6.9231 -> 6.9081 and 6.8901 -> 6.8894): the different tie-breaking
    cascades through the placer's greedy decisions, and first-fit and the old
    router MIS disagreed on only 3.2% of moving stages anyway. First-fit is
    cheap, order-stable, and empirically fine; re-measure before assuming
    this conclusion still holds against the new router algorithm if this is
    ever revisited.
    """
    # Sorted tone index per batch (2026-10-03). The members of a batch are
    # pairwise compatible, so on each axis their tones form a strictly
    # increasing map start -> end. A new vector is then compatible with every
    # member iff, on each axis, it matches an existing tone exactly or sits
    # strictly between its two neighbouring tones (by start) with at least
    # MIN_TONE_SEP to each at both ends -- O(log m) per batch instead of m
    # compatible_2D calls. Exactly equivalent to the plain first-fit loop
    # (verified against it on random inputs, ds on and off); batches are
    # returned in the same order with the same members.
    return _first_fit([], [], [], vectors)


def _first_fit(batches, index, owner, vectors, start=0):
    """First-fit core of ``batch_moves``, resumable: pack ``vectors[start:]``
    into the existing ``batches`` (``index`` holds each batch's sorted tones:
    ``[xs, xe, ys, ye]``), appending each vector's batch number to ``owner``.
    """
    ds = MIN_TONE_SEP
    bl = bisect.bisect_left
    def axis_ok(st, en, s, e):
        i = bl(st, s)
        if i < len(st) and st[i] == s:
            return en[i] == e
        if i > 0:
            ps, pe = st[i - 1], en[i - 1]
            if not pe < e:
                return False
            if ds and (s - ps < ds or e - pe < ds):
                return False
        if i < len(st):
            ns, ne = st[i], en[i]
            if not ne > e:
                return False
            if ds and (ns - s < ds or ne - e < ds):
                return False
        return True
    def add(st, en, s, e):
        i = bl(st, s)
        if i < len(st) and st[i] == s:
            return
        st.insert(i, s)
        en.insert(i, e)
    # Least-increase fit (2026-10-18): each vector joins the
    # compatible batch whose bottleneck it raises least (first such on ties),
    # so insertion matches how the placer prices a move. First-fit measured
    # within noise of it (tqe -0.00004, qasmbench_std +0.0009, 20 scalability
    # circuits -0.16% -log10 F). Still prefix-stable, so FirstFitCache holds.
    for i in range(start, len(vectors)):
        v = vectors[i]
        dur = _duration(v)
        pick = None
        for bi, (batch, (xs, xe, ys, ye)) in enumerate(zip(batches, index)):
            if axis_ok(xs, xe, v[0], v[1]) and axis_ok(ys, ye, v[2], v[3]):
                inc = max(0.0, dur - batch[1])
                if pick is None or inc < pick[0]:
                    pick = (inc, bi)
                    if inc == 0.0:
                        break
        if pick is None:
            owner.append(len(batches))
            batches.append([[v], dur])
            index.append([[v[0]], [v[1]], [v[2]], [v[3]]])
            continue
        bi = pick[1]
        batch, (xs, xe, ys, ye) = batches[bi], index[bi]
        batch[0].append(v)
        if dur > batch[1]:
            batch[1] = dur
        add(xs, xe, v[0], v[1])
        add(ys, ye, v[2], v[3])
        owner.append(bi)
    return batches


class FirstFitCache:
    """``batch_moves`` for a run of calls whose vector lists share long
    prefixes, e.g. the placer's per-gate base plan (2026-10-03). First-fit
    places each vector looking only at the vectors before it, so the packing
    of a shared prefix is identical in both calls: keep it, drop everything
    after the first difference, and resume from there. Returns exactly
    ``batch_moves(vectors)``, as fresh lists every call.
    """

    def __init__(self):
        self.vectors, self.owner, self.batches, self.index = [], [], [], []

    def __call__(self, vectors):
        old = self.vectors
        k, n = 0, min(len(old), len(vectors))
        while k < n and old[k] == vectors[k]:
            k += 1
        if k < len(old):
            # Batches that hold a prefix vector, in creation order, each with
            # its prefix members in their original order.
            keep = {}
            for i in range(k):
                keep.setdefault(self.owner[i], []).append(old[i])
            order = sorted(keep)
            remap = {b: j for j, b in enumerate(order)}
            self.owner = [remap[b] for b in self.owner[:k]]
            self.batches, self.index = [], []
            for b in order:
                mem = keep[b]
                self.batches.append([mem, max(_duration(v) for v in mem)])
                xs = sorted({(v[0], v[1]) for v in mem})
                ys = sorted({(v[2], v[3]) for v in mem})
                self.index.append([[a for a, _ in xs], [e for _, e in xs],
                                   [a for a, _ in ys], [e for _, e in ys]])
        self.vectors = list(vectors)
        _first_fit(self.batches, self.index, self.owner, self.vectors, start=k)
        return [[list(b[0]), b[1]] for b in self.batches]


def absorb_move(vector, batches):
    """Insert one ``vector`` into an existing ``batches`` structure, in place.

    Same least-increase rule as ``batch_moves``, applied incrementally: candidate
    placements differ from each other by only one or two vectors, so re-packing
    a whole stage per candidate is O(V^2) work repeated per candidate. Returns
    nothing — callers that also want the marginal cost compute it before
    calling, against the batches as they stood.
    """
    if vector is None:
        return
    dur = _duration(vector)
    pick = None   # least-increase fit, as in _first_fit
    for batch in batches:
        for u in batch[0]:  # plain loop, same reason as in batch_moves
            if not compatible_2D(vector, u):
                break
        else:
            inc = max(0.0, dur - batch[1])
            if pick is None or inc < pick[0]:
                pick = (inc, batch)
                if inc == 0.0:
                    break
    if pick is None:
        batches.append([[vector], dur])
        return
    pick[1][0].append(vector)
    if dur > pick[1][1]:
        pick[1][1] = dur
