import bisect
import heapq
import math
import os
import numpy as np

from . import transport

R_B = 5  # Rydberg interaction radius


class Placer:
    """
    Class to determine a qubit layout based on minimal distances between qubits
    and available interaction sites within a specified Rydberg interaction radius.
    """

    # The transport model is shared with Router, not reimplemented here — see
    # NEAT/transport.py. Bound as staticmethods so the existing
    # ``self.movement_duration(...)`` / ``self._compatible_2d(...)`` call sites
    # reach the module function directly, with no forwarding frame: these sit
    # inside O(V^2) inner loops and are the hottest code in the compiler.
    movement_duration = staticmethod(transport.movement_duration)
    _compatible_2d = staticmethod(transport.compatible_2D)

    def __init__(self,
                 slm_sites: list,
                 results_code: dict,
                 list_full_gates: list,
                 qubit_mapping: list,
                 architecture: dict = None):
        """
        Args:
            slm_sites: ``[storage_traps, entanglement_traps]``.
            results_code: Must include ``n_q``, ``stages`` from the scheduler.
            list_full_gates: Gates grouped per schedule stage.
            qubit_mapping: User-specified traps, or empty for automatic init.
            architecture: Optional fidelity/T2/routing weights for cost heuristics.
        """
        self.stg_slm_sites, self.ent_slm_sites = slm_sites
        self.results_code = results_code
        self.stages = results_code['stages']['stage']
        self.n_q = results_code['n_q']
        self.list_full_gates = list_full_gates
        self.qs_status = results_code['stages']['qs_status']

        architecture = architecture or {}
        self._pre_cropped = bool(architecture.get('pre_cropped', False))
        operation_fidelity = architecture.get('operation_fidelity', {})
        qubit_spec = architecture.get('qubit_spec', {})
        routing_cfg = architecture.get('routing', {})

        two_q_fid = operation_fidelity.get('two_qubit_gate', 0.999)
        self.fidelity_2q_idle = 1 - (1 - two_q_fid) / 2

        self.fidelity_atom_transfer = operation_fidelity.get('atom_transfer', 0.999)

        self.time_coherence = qubit_spec.get('T2', 1.5e6)
        self.time_atom_transfer = architecture.get('operation_duration', {}).get('atom_transfer', 15)
        # NOTE: ``parallel_priority_weight`` is retired. AOD parallelism is now
        # priced in microseconds by _marginal_wave_cost
        # instead of by a magic conflicts-times-1000 penalty added to a length.
        # REMOVED 2026-09-13: ``initial_mapping_parallel_lookahead``. Never
        # read back out anywhere in this tree (P3 note, 2026-09-02) — the
        # initial mapping deliberately seeds a vector for every future mover
        # with no lookahead cutoff, so this knob had no effect regardless of
        # value. settings/tqe.json and settings/graphs.json's explicit
        # `initial_mapping_parallel_lookahead: 0` keys are now unread; left
        # in those files as historical record rather than edited, since 0
        # was always the default anyway (bit-identical either way).
        #
        # REMOVED 2026-09-13: the ``routing_strategy`` constructor parameter
        # (and its ``always_move``/``always_stay`` forced-policy values). It
        # only ever gated the crosstalk-vs-move_cost comparison in
        # _move_unentangled_qubits (now unconditional — see that function),
        # for an A/B contest measured and removed 2026-09-02; no shipped
        # config ever set it to anything but "baseline". Router still
        # accepts and stores its own routing_strategy for logging/CLI
        # purposes, but no longer threads it into Placer.
        #
        # REMOVED 2026-09-12: ``idle_cost_alpha``, a weight on move_cost's
        # decoherence term. Every settings file in this repo left it at its
        # 1.0 default; only tools/sweep_configs.py ever swept it away from
        # 1.0, and per that sweep's own docstring it was "the knob the suite
        # splits on hardest" — a chaotic, non-monotonic single-circuit-driven
        # response (see git history for the sweep's own comment), not a
        # dial with a real optimum to preserve. Hardcoded to 1.0 (its
        # shipped default, so this is not a behavior change) and the config
        # key removed to stop it being tunable at all.
        # REMOVED 2026-09-05: ``future_travel_weight`` and its
        # ``_future_travel_cost`` term. The fixed config adopted 2026-08-31
        # set the weight to 0.0, which made the whole term identically zero —
        # its own block comment recorded 195438 calls across three circuits
        # returning 0.0 every time, and both of its call sites had already
        # grown an explicit ``if self.future_travel_weight:`` branch to skip
        # it. Deleting the dead weighted branch is bit-identical on all 14 tqe
        # circuits (verified; see the removal note in NEAT/neat.py).
        #
        # What it modelled, if it is ever wanted back: the wall-time a pair
        # site costs a qubit AFTER its gate — ``min(leave, stay)``, evacuate
        # to storage versus absorb crosstalk until the next gate. The ``min``
        # was load-bearing. An earlier version charged travel only when
        # ``stay > leave``, which is not monotone in distance (a farther site
        # has larger ``leave``, flipping the test to "stay", charging zero) and
        # so rewarded the far sites it meant to discourage — sat_n11 went from
        # 34 to 221 far-row gates. Recover it from git history, not from
        # scratch.
        # Free pairs fully scored per gate by the wave model (see _pair_options).
        # Runtime cut (B4): shortlist size swept 6/4/3/2 on tqe. 4 costs
        # -0.000966 total F (< the 0.002 noise budget) for ~10% less compile
        # time; 3 costs -0.001726 (still under budget but less margin); 2
        # costs -0.003658 (over budget, rejected). Kept at 4 for margin.
        self.pair_candidate_limit = int(
            routing_cfg.get("pair_candidate_limit", 4))
        # Spatial grid cell size for _pair_options (2026-09-16) -- see
        # placing()'s _pair_grid construction and _pair_options for the
        # mechanism and the fidelity/compile-time trade this makes on
        # purpose. Sized to a few pair-pitches so each queried cell usually
        # holds a handful of candidates, not zero and not the whole zone.
        self._grid_cell_size = float(routing_cfg.get("pair_grid_cell_size", 40.0))
        self._grid_search_radius = int(routing_cfg.get("pair_grid_search_radius", 1))
        # Same spatial-grid restriction, applied to _pick_ranked_site's
        # storage-site scan (2026-09-16) -- see _build_storage_grid and
        # _pick_ranked_site for the mechanism. Independent knobs from the
        # pair grid above since storage and entanglement zones have
        # different pitches and densities.
        self._stg_grid_cell_size = float(
            routing_cfg.get("storage_grid_cell_size", 40.0))
        self._stg_grid_search_radius = int(
            routing_cfg.get("storage_grid_search_radius", 1))
        # TRIED 2026-09-16 and REVERTED: a `wave_batch_scan_limit` config
        # capping how many already-planned AOD waves a move is checked
        # against in transport.batch_moves/absorb_move, targeting
        # `scalability/random`'s wide stages (414M compatible_2D calls
        # measured on random_n1000, its dominant placer cost). Swept
        # 10/30/50 on random_n1000: fidelity response was NOT monotone in
        # the cap (F worse by ~51 orders of magnitude at cap=10, ~19 at
        # cap=30, ~9 at cap=50 -- tighter cap did not mean worse fidelity
        # in a predictable way) and compile time was similarly erratic
        # (cap=50 measured SLOWER than uncapped). Consistent with this
        # file's repeated warnings elsewhere that the placer's greedy
        # decisions are chaotically sensitive to changes in this area
        # (see batch_moves' own docstring in NEAT/transport.py on the
        # LPT/router-MIS reordering attempts). Reverted
        # rather than searched further for a stable point --
        # the fidelity cost here was an order of magnitude larger than
        # every other change shipped at the time and the relationship
        # wasn't predictable enough to dial in safely. Not reintroduced;
        # recover from git history if ever revisited with a steadier
        # measurement methodology (the single-sample timings here were
        # also contaminated by machine contention noise seen throughout
        # at the time, which may explain some but not all of the swing).
        # REMOVED 2026-09-13: ``storage_window`` and ``cold_gap_threshold``.
        # Unread since the 2026-09-11 hot/cold-split unification
        # (_pick_ranked_site no longer branches on gap — see
        # _assign_storage_site_ranked's docstring for that measurement); the
        # only remaining reference, tools/sweep_configs.py, was itself
        # deleted 2026-09-13 as a sweep tool for knobs that no longer exist.
        # Modelled-cost differences below this (µs) are treated as noise and
        # settled by the tie-breaks in _place_gate. The wave model is a
        # heuristic, not a simulation, so sub-resolution differences are not
        # worth acting on. Measured: the suite is flat-to-better anywhere in
        # 5-50 µs and clearly worse below ~2 µs; the response inside the good
        # band is chaotic (about +/-0.003 total F), so 5 is the low end of a
        # plateau, not a tuned optimum.
        # REMOVED 2026-10-14: the gate-order search (gate_order_slacks,
        # gate_order_early_exit, ablate_gate_order) -- see
        # _process_two_qubit_gates.
        # REMOVED 2026-09-03: the adaptive best-first ("greedy") gate order,
        # along with its `gate_order_greedy` / `gate_order_greedy_max_k` knobs
        # and `_assign_gate_order_greedy`. It re-measured the stage cost after
        # every trial placement instead of ranking gates up front, at O(k^2)
        # placements per stage against O(k) for a static order.
        #
        # Removed on a five-suite on/off measurement showing it paid only on
        # narrow-stage circuits and cost real time on wide ones:
        #
        #   suite   sizes        dF (on - off)   b/w/same   compile on/off
        #   tqe     4-35q          +0.005081      4/1/9        1.22x
        #   tqe2    5-33q          +0.003808      3/0/11       1.19x
        #   graphs  10-100q x100   +0.038512     42/29/29      1.22x
        #   random  10-300q        +0.004907      2/0/9        1.00x
        #   adder   10-316q        +0.000801      2/2/7        1.00x
        #   qft     10-250q        +3.5e-7        1/0/5        1.51x
        #
        # The gains do not scale. On random every difference is n10/n20 — n40
        # through n300 were bit-identical; adder is the same, all movement at
        # n28-n64 and nothing from n118 up. That is the k<=12 cap turning
        # greedy off: a stage-width census found 80% of qft's 3000 stages and
        # 60% of random's above the cap (qft mean k=44, max 125), so on large
        # circuits it mostly never fired, and where it did the sign was a coin
        # flip — graphs was 42 better / 29 worse, its worst regression
        # (graphs_30_4, -0.0101) larger than its best gain, with no trend
        # across size buckets. qft was pure cost: +3.5e-7 (noise on qft_n50,
        # whose F is already 2.6e-6; n>=100 is a hard zero either way) for 51%
        # more compile time, 177s -> 337s on qft_n250 alone.
        #
        # The cost is tqe: 6.911217 -> 6.906136 (-0.005081), with vqc_n15,
        # qft_n10, wstate_n27 and multiplier_n15 regressing. That is below the
        # old floor and was accepted on 2026-09-03 as a deliberate
        # trade for a simpler placer and less compile time. This is the one documented case of
        # shipping under a floor — it is not a precedent.
        #
        # Do NOT add a cap on the STATIC orders in its place. Forcing circuit
        # order alone did not merely lose fidelity, it aborted random_n300 with
        # "No valid pair site found"; candidate breadth is what rescues a
        # dead-ended order (see the ValueError handler in
        # _process_two_qubit_gates).
        # REMOVED 2026-09-05: ``cost_quantum`` and ``_quantise``. The fixed
        # config adopted 2026-08-31 set it to 0.0, and ``_quantise`` returned
        # its argument unchanged for any value <= 0, so the sole call site in
        # _place_gate was an identity function. Bit-identical on tqe when
        # deleted (verified).
        #
        # What it did: bucketed modelled costs by ``math.floor(cost / q)`` so
        # sub-resolution differences compared equal and fell through to
        # ``pair_tiebreak``. Measured plateau was 5-50 µs — flat-to-better
        # anywhere inside it, clearly worse below ~2 µs, and chaotic within
        # (about +/-0.003 total F), so it was a plateau rather than a tuned
        # optimum. Note this made tie-break precision inconsistent: quantized
        # here but exact elsewhere (placer.md §"Inconsistent tie-break
        # precision"), which removal also resolves.
        # Debug hook for the live-occupancy A/B/C protocol (LIVE_OCCUPANCY_AND_
        # JOINT_PLACEMENT.md, Part A2): ZAP_FORCE_PLACEMENT="stage=<S>:q<idx>=(x,y)"
        # overrides qubit <idx>'s FINAL site for stage <S> only, after the normal
        # placement pipeline has run for that stage, then lets routing/simulation
        # proceed downstream unmodified. See _apply_force_override.
        self._force_target = self._parse_force_placement(
            os.environ.get("ZAP_FORCE_PLACEMENT", ""))

        # Phase-1 proxy audit; see proxy_stage_forecast and _departure_sites.
        # Measurement only — nothing gated on this flag changes a placement.
        # --- CUMULATIVE-ABLATION GATES (measurement only, all default OFF, so
        # the shipped compiler is bit-identical). Each disables one placer
        # component while leaving every other stage untouched; see
        # experiments/README.md for the measured contributions.
        # ablate_wave_cost_model: revert _marginal_wave_cost / _score_moves from
        # the microsecond wall-time model to the HISTORICAL predecessor this
        # file replaced -- parallel_priority_weight's `conflicts * 1000 +
        # distance` (see baselines/zap/placer.py::_single_move_rank). Mismatched
        # units by construction; that is the point of the comparison.
        self.ablate_wave_cost_model = bool(
            routing_cfg.get("ablate_wave_cost_model", False))
        self.parallel_priority_weight = float(
            routing_cfg.get("parallel_priority_weight", 1000.0))
        # ablate_fidelity_cost_model: price the idle stay/evacuate decision in
        # RAW COUNTS (stages idle vs transfer events) instead of commensurable
        # -log(fidelity) units -- the same unit-mismatch the wave model above
        # reverts to, applied to the physics side.
        self.ablate_fidelity_cost_model = bool(
            routing_cfg.get("ablate_fidelity_cost_model", False))
        self.ablate_pair_scoring = bool(routing_cfg.get("ablate_pair_scoring", False))
        self.ablate_home_traps = bool(routing_cfg.get("ablate_home_traps", False))
        self.ablate_idle_policy = bool(routing_cfg.get("ablate_idle_policy", False))
        # ablate_evac_rank: keep evacuation ON, but choose the destination the
        # way baseline zap does (home trap if free, else nearest by raw
        # distance) instead of ranking every free site by _move_rank's
        # marginal wave cost. This isolates the RANKING from the POLICY --
        # ablate_idle_policy above disables evacuation altogether, which is a
        # different and much larger intervention.
        self.ablate_evac_rank = bool(routing_cfg.get("ablate_evac_rank", False))

        self.proxy_audit = bool(routing_cfg.get("proxy_audit", False))
        self._audit_seed = self._blank_seed_audit()

        if len(qubit_mapping) > 0:
            self.current_mapping = qubit_mapping
        else:
            self.current_mapping = [(-1, -1) for _ in range(self.n_q)]
            self.init_slm_sites()
            self.init_mapping()

        from copy import deepcopy
        # Home traps for idle-vs-storage heuristics and final return-to-storage bias.
        self.initial_mapping = deepcopy(self.current_mapping)

        self.init_pairs()

        # Membership-test mirrors of the two site lists. ``site in self.xxx_slm_sites``
        # is a linear scan and sits inside several per-candidate inner loops.
        self._stg_set = set(self.stg_slm_sites)
        self._ent_set = set(self.ent_slm_sites)

        # Per-qubit sorted list of stages with a 2q gate. Computed once so the
        # per-stage urgency/gap lookups are O(log n) instead of a linear scan
        # over qs_status for every qubit at every stage.
        num_stages = self.results_code['stages']['num_stage']
        self._q_2q_stages = [
            [i for i in range(num_stages) if self.qs_status[q][i]['status'] == "2qGate"]
            for q in range(self.n_q)
        ]


    def _next_2q_stage(self, q, after_stage):
        """First stage strictly after ``after_stage`` where q does a 2q gate, else None."""
        lst = self._q_2q_stages[q]
        i = bisect.bisect_right(lst, after_stage)
        return lst[i] if i < len(lst) else None

    def _urgency(self, q):
        """Cache-style urgency: 1/(gap+1), gap = stages until q's next 2q gate.

        Why: a qubit needed next stage should get first claim on scarce
        near-boundary storage sites; one never needed again has urgency ~0 and
        takes whatever is left. The proximity gradient of the storage zone then
        emerges from matching urgency to proximity in sorted order.

        """
        nxt = self._next_2q_stage(q, self.index_list_gate)
        if nxt is None:
            return 0.0
        # Gap counted in 2q-stages (2026-10-18), the same k_q as the stay cost.
        pre = self._prefix_2q()
        return 1.0 / (pre[nxt] - pre[self.index_list_gate] + 1)

    def _prefix_2q(self):
        """pre[s] = number of stages before s that contain a 2q gate."""
        pre = getattr(self, '_pre2q', None)
        if pre is None:
            num_stages = self.results_code['stages']['num_stage']
            pre = [0]
            for st in range(num_stages):
                pre.append(pre[-1] + any(self.qs_status[o][st]['status'] == "2qGate"
                                         for o in range(self.n_q)))
            self._pre2q = pre
        return pre

    def init_slm_sites(self):
        if self._pre_cropped:
            # NEAT/architecture.py already sized storage and entanglement
            # exactly for this circuit (n_q and max_stage_2q respectively), so
            # the sqrt(n_q)-based re-crop below is not merely redundant, it is
            # actively harmful: it re-trims storage and entanglement with TWO
            # DIFFERENT, uncoordinated sort/window rules (storage sorts
            # (x asc, y desc); entanglement sorts (x asc, y asc) with its own
            # separate column count), so the kept entanglement slice can end
            # up smaller than and shifted relative to what was generated --
            # e.g. found on a generated 35-qubit array: 112 entanglement sites
            # declared, only 78 kept, the kept window shifted left (x 1..73 of
            # a 1..89 zone). init_mapping's mean-distance sort then optimises
            # against that shifted crop's centre, not the true array centre,
            # which was the actual cause of "why do atoms lean left" on a
            # WIDTH-COUPLED generated array (2026-09-08).
            return
        n_cols = [1 for _ in range(len(self.stg_slm_sites))]
        n_rows = [1 for _ in range(len(self.stg_slm_sites))]
        for i in range(len(self.stg_slm_sites)):
            for j in range(i+1, len(self.stg_slm_sites)):
                if self.stg_slm_sites[i][0] == self.stg_slm_sites[j][0]:
                    n_rows[i] += 1
                if self.stg_slm_sites[i][1] == self.stg_slm_sites[j][1]:
                    n_cols[i] += 1
        n_col = max(n_cols)
        n_row = max(n_rows)
        stg_n_col = min(int(np.sqrt(self.n_q)) + 4, n_col)
        stg_n_row = min(int(np.sqrt(self.n_q)) + 2, n_row)
        while True:
            if stg_n_col * stg_n_row >= self.n_q:
                break
            else:
                stg_n_col = min(stg_n_col+1, n_col)
                stg_n_row = min(stg_n_row+1, n_row)
                
        self.stg_slm_sites = sorted(self.stg_slm_sites, key=lambda item: (item[0], -item[1]))
        tmp = []
        for i in range(stg_n_col):
            for j in range(stg_n_row):
                tmp.append(self.stg_slm_sites[i * n_row + j])
        self.stg_slm_sites = tmp
  
        n_cols = [1 for _ in range(len(self.ent_slm_sites))]
        n_rows = [1 for _ in range(len(self.ent_slm_sites))]
        for i in range(len(self.ent_slm_sites)):
            for j in range(i+1, len(self.ent_slm_sites)):
                if self.ent_slm_sites[i][0] == self.ent_slm_sites[j][0]:
                    n_rows[i] += 1
                if self.ent_slm_sites[i][1] == self.ent_slm_sites[j][1]:
                    n_cols[i] += 1
        n_col = max(n_cols)
        n_row = max(n_rows)
        ent_n_col = min(stg_n_col + 4, n_col)
        ent_n_row = min(stg_n_row, n_row)

        # Grow the entanglement crop until it can hold the widest stage, 2026-09-05.
        #
        # Both crops above are sized from sqrt(n_q) — a working sub-grid, so the
        # placer does not pay O(sites) scans over an architecture far larger than
        # the circuit. That is fine for storage (it only has to hold n_q atoms)
        # but the entanglement zone's demand is set by the widest STAGE, not by
        # n_q, and nothing here consulted the circuit. On a 3-regular graph state
        # at n=500 the crop yielded 360 Rydberg pairs out of the 1088 the
        # architecture actually owns, and the compile died with "No gate order
        # could place stage 12: all 5 candidates ran out of entanglement pair
        # sites" — 96 gates wanting pairs with 4 free, while 446 atoms sat parked
        # in the zone. The hardware was never the limit; this crop was.
        #
        # A stage of k two-qubit gates needs k Rydberg pairs simultaneously. The
        # zone's two interleaved SLMs give one pair per (col, row) cell, so k
        # cells is the floor — but it is nowhere near enough in practice,
        # because the placer leaves idle atoms parked in the zone between their
        # gates. Measured on the n=500 failure: 446 of 500 atoms were resident
        # when a 96-gate stage needed pairs, and each resident blocks the pair
        # it sits on whether or not it is gating. So the zone must have room for
        # the widest stage ON TOP OF whatever is parked, and in the worst case
        # what is parked approaches n_q.
        #
        # Sizing against n_q + max_stage_2q covers that: enough room that even
        # a fully-resident circuit leaves the widest stage somewhere to go. This
        # is deliberately generous rather than tight — the cost of an oversized
        # crop is per-stage scan time over more sites, the cost of an undersized
        # one is a dead compile, and the sqrt(n_q) sizing above already keeps
        # small circuits small.
        #
        # Count in SITES, not cells. A Rydberg pair is two sites (one in each of
        # the two interleaved SLMs), so a crop of C columns x R rows holds C*R
        # sites and only C*R/2 pairs. Sizing the loop against cells was the
        # first version of this fix and it silently did nothing: at n=500 it
        # asked for 596 "cells", saw 720, and stopped — while those 720 sites
        # were just 360 pairs against 446 resident atoms.
        #
        # Only ever GROWS the crop, and only up to what the architecture has, so
        # any circuit that compiled before sees an identical or larger zone —
        # never a smaller one. Measured effect (2026-09-05):
        #
        #   tqe        bit-identical, 6.906135582, 14/14 unchanged (zone already
        #              big enough at these sizes, so the loop never fires)
        #   graphs     +0.041899 over 100 circuits (24 better, 16 worse)
        #   random     +0.000712 (5 better, 4 worse)
        #   qft        +0.000000 (4 circuits shifted, net zero)
        #   adder      -0.0000067 (2 better, 5 worse) — ~1e-5 of that suite's
        #              total, the placer's usual circuit-to-circuit chaos
        #
        # It is NOT bit-identical on the wide suites: a larger zone changes which
        # pair sites exist, which changes placement. The floor suite (tqe) is
        # untouched, and everything else nets positive except adder's rounding-
        # level loss.
        max_stage_2q = 0
        for stage_gates in self.list_full_gates:
            k = sum(1 for g in stage_gates if g[0] != g[1])
            if k > max_stage_2q:
                max_stage_2q = k
        # Sites needed = 2 per pair, for (parked atoms + this stage's gates).
        need_sites = 2 * (self.n_q + max_stage_2q)
        while ent_n_col * ent_n_row < need_sites:
            if ent_n_col >= n_col and ent_n_row >= n_row:
                break  # architecture is genuinely too small; fail downstream
            ent_n_col = min(ent_n_col + 1, n_col)
            ent_n_row = min(ent_n_row + 1, n_row)

        self.ent_slm_sites = sorted(self.ent_slm_sites, key=lambda item: (item[0], item[1]))
        tmp = []
        for i in range(ent_n_col):
            for j in range(ent_n_row):
                tmp.append(self.ent_slm_sites[i * n_row + j])
        self.ent_slm_sites = tmp
        
    def init_mapping(self):
        """Assign every qubit its home trap. The one non-stage-scoped decision.

        A sort and a zip: order the storage traps by mean distance to the
        entanglement zone, order the qubits by stage-decayed gate weight, and
        hand the closest traps to the busiest qubits.
        """
        # Mean distance, not min: a trap with a low mean sits centrally with
        # respect to the whole zone, so it is a decent start for whichever
        # entanglement column the qubit eventually needs. Min distance instead
        # rewards hugging one particular entrance, which packs qubits toward it
        # and leaves more of them loitering in the zone between gates.
        #
        # This sort also persists on ``self.stg_slm_sites`` for the rest of the
        # compile: _move_unentangled_qubits, _pick_ranked_site and
        # _move_to_target_zone all scan this list in order and take the first
        # acceptable site, so it doubles as the global tie-break policy for
        # every stage-scoped placement decision.
        # ABLATION (ablate_home_traps): skip the mean-distance sort entirely and
        # keep the architecture's raw site order, so qubits are handed traps by
        # index rather than by centrality. Combined with the priority ablation
        # below this makes the initial mapping an identity assignment.
        if not self.ablate_home_traps:
            self.stg_slm_sites = sorted(
                self.stg_slm_sites,
                key=lambda coord: float(np.mean(
                    [math.dist(coord, ez) for ez in self.ent_slm_sites]))
            )

        # Spatial grid over storage sites (2026-09-16), mirroring the pair
        # grid built for _pair_options -- see _pick_ranked_site. Built once
        # here since stg_slm_sites' membership is fixed for the whole
        # compile (only its sort order changed above); unlike available_pairs
        # this never needs rebuilding per-stage.
        stg_cell = self._stg_grid_cell_size
        self._stg_grid = {}
        for site in self.stg_slm_sites:
            key = (int(site[0] // stg_cell), int(site[1] // stg_cell))
            self._stg_grid.setdefault(key, []).append(site)

        # Stage-decayed gate count: a stage-0 gate is worth 1.0, stage-99 worth
        # 0.01, so a qubit that is busy early and often sorts first.
        self.qubit_weight = np.zeros(self.n_q)
        for stage, gates in enumerate(self.list_full_gates):
            weight = 1 / (stage + 1)
            for i in range(len(gates)):
                self.qubit_weight[gates[i][0]] += weight
                self.qubit_weight[gates[i][1]] += weight

        qubit_priority = (list(range(self.n_q)) if self.ablate_home_traps else
                          sorted(range(self.n_q),
                                 key=lambda i: self.qubit_weight[i],
                                 reverse=True))

        # Busiest qubit gets the most central free trap. No search, no
        # backtracking -- ``stg_slm_sites`` is already in the order we want.
        #
        # This replaced a greedy per-site search (removed 2026-09-10) that
        # scored every free site for every qubit by
        # ``rank = (_marginal_wave_cost(first trip), nearest_ent_dist)``, priced
        # against vectors already committed for higher-priority qubits. It was
        # O(n_q * sites) with an O(n_q^2) ``site not in assigned_sites`` list
        # scan inside, and it provably could not change the outcome:
        #
        #   1. The wave-cost key is a monotone function of the tie-break key.
        #      Opening a new wave costs ``2*time_atom_transfer +
        #      movement_duration(d)`` = ``30 + 200*sqrt(d/110)``, strictly
        #      increasing in the same ``d`` the tie-break sorted on, so both
        #      keys induce the same order on that branch. A histogram of all
        #      31,801 candidate evaluations across tqe put 87.8% there; the
        #      other 12.2% (rides an existing wave for 0.0, or extends one
        #      cheaply) never flipped the argmin, because a qubit whose first
        #      trip rides an existing wave is already one sitting close in.
        #   2. With the wave term inert, greedy "take the min free site" over a
        #      fixed per-site key is just that key's sort order. Verified on all
        #      14 tqe circuits: greedy picks == sorted picks.
        #
        # Removing it was bit-identical on tqe (14/14) and graphs (105/105,
        # n=10..1000), every circuit and not just the totals. Switching the
        # remaining key from min to mean distance then RAISED the floor
        # (+0.028339 tqe, +0.166904 graphs) by cutting crosstalk exposures
        # 349 -> 303.
        for q, site in zip(qubit_priority, self.stg_slm_sites):
            self.current_mapping[q] = site


    def init_params(self):
        """
        Initialize parameters used during placement operations.
        """
        self.execute_qubits_1q = []
        self.execute_qubits_2q = []
        self.gate_2q = []
        self.list_gate = self.list_full_gates[self.index_list_gate]
        # Planned movement vectors in this stage; used to favor parallel moves.
        self._planned_vectors = []
        # Incrementally-maintained batching of ``_planned_vectors`` (see
        # _commit_move / _absorb_into_batches). Avoids re-batching the whole
        # stage from scratch — O(V^2) — on every candidate site evaluation.
        self._planned_batches = []

    @staticmethod
    def _parse_force_placement(spec):
        if not spec:
            return None
        stage = qidx = site = None
        for part in spec.split(":"):
            if part.startswith("stage="):
                stage = int(part.split("=", 1)[1])
            elif part.startswith("q") and "=" in part:
                qname, coord = part.split("=", 1)
                qidx = int(qname[1:])
                xs, ys = coord.strip("()").split(",")
                # F5 (2026-09-02): every real trap coordinate is an integer
                # (Router.verify_path calls math.gcd on them). Parsing as
                # float let a fractional ZAP_FORCE_PLACEMENT slip through
                # here and fail deep in the router with a confusing error
                # instead of a clear one at parse time.
                site = (int(xs), int(ys))
        if stage is None or qidx is None or site is None:
            raise ValueError(f"malformed ZAP_FORCE_PLACEMENT spec: {spec!r}")
        return (stage, qidx, site)

    def _apply_force_override(self):
        """See ``_force_target`` / ZAP_FORCE_PLACEMENT. Rewrites qubit q's
        planned stage-net vector to end at the forced site instead of
        whatever the placement pipeline chose, then lets router/simulator
        proceed normally against the modified current_mapping."""
        if self._force_target is None:
            return
        stage, q, site = self._force_target
        if self.index_list_gate != stage:
            return
        start = self._stage_start_mapping[q]
        old_target = self.current_mapping[q]
        if old_target == site:
            return
        taken = {p for i, p in enumerate(self.current_mapping) if i != q}
        if site in taken:
            print(f"ZAP_FORCE_PLACEMENT infeasible: site {site} occupied "
                  f"at stage {stage} (q{q})")
            return
        old_vector = (start[0], old_target[0], start[1], old_target[1]) \
            if start != old_target else None
        if old_vector is not None:
            if old_vector in self._planned_vectors:
                self._planned_vectors.remove(old_vector)
            for batch in self._planned_batches:
                if old_vector in batch[0]:
                    batch[0].remove(old_vector)
                    if batch[0]:
                        batch[1] = max(
                            self.movement_duration(
                                math.dist((v[0], v[2]), (v[1], v[3])))
                            for v in batch[0])
                    else:
                        self._planned_batches.remove(batch)
                    break
        occ = getattr(self, "_occupied", None)
        if occ is not None:
            occ.discard(old_target)
            occ.add(start)
        self.current_mapping[q] = start
        self._commit_move(q, site)
        print(f"ZAP_FORCE_PLACEMENT applied: stage={stage} q{q} "
              f"{old_target} -> {site}")

    def _vector_for_move(self, q, target_site):
        sx, sy = self.current_mapping[q]
        tx, ty = target_site
        if (sx, sy) == (tx, ty):
            return None
        return (sx, tx, sy, ty)

    def _commit_move(self, q, target_site):
        vector = self._vector_for_move(q, target_site)
        if vector is not None:
            self._planned_vectors.append(vector)
            self._absorb_into_batches(vector, self._planned_batches)
        occ = getattr(self, "_occupied", None)
        if occ is not None:
            old_site = self.current_mapping[q]
            occ.discard(old_site)
            occ.add(target_site)
        self.current_mapping[q] = target_site

    def _absorb_vector(self, vector, batches):
        """Add ``vector`` to ``batches`` greedily; return the wall-time it costs.

        Same rule and same answer as re-running ``_batch_vectors`` on the whole
        set, but incremental: candidate placements differ only by one or two
        vectors, so re-batching the entire stage for each of them is O(V^2) work
        repeated per candidate. Mutates ``batches``.

        The insert itself is ``transport.absorb_move``; this only adds the
        marginal price, which has to be read off ``batches`` BEFORE the vector
        goes in.
        """
        if vector is None or self.ablate_wave_cost_model:
            cost = self._marginal_wave_cost(vector, batches)
            self._absorb_into_batches(vector, batches)
            return cost
        # One scan for both (2026-10-03, bit-identical): the price is read off
        # the compatible waves exactly as _marginal_wave_cost does, and the
        # vector goes into the compatible wave it lengthens least, exactly as
        # absorb_move puts it (least-increase since 2026-10-18; was the first
        # compatible wave) -- the two used to scan the same batches twice.
        dur = self.movement_duration(
            math.dist((vector[0], vector[2]), (vector[1], vector[3])))
        cost = 2 * self.time_atom_transfer + dur          # open a new wave
        compatible = self._compatible_2d
        first = None
        for batch in batches:
            for u in batch[0]:
                if not compatible(vector, u):
                    break
            else:
                if first is None:
                    first = batch
                if dur <= batch[1]:
                    cost = 0.0
                    first = batch
                    break
                if dur - batch[1] < cost:
                    cost = dur - batch[1]
                    first = batch
        if first is None:
            batches.append([[vector], dur])
        else:
            first[0].append(vector)
            if dur > first[1]:
                first[1] = dur
        return cost

    def _marginal_wave_cost(self, vector, batches):
        """Wall-time (µs) added if ``vector`` joins an already-planned wave set.

        A move that fits into a compatible wave whose bottleneck already exceeds
        its own duration adds exactly nothing — it happens in parallel, for
        free. One that fits nowhere opens a new wave and pays the
        activate/deactivate overhead plus its full duration.

        This is the honest version of what ``parallel_priority_weight`` was
        approximating. That knob scored a site as ``conflicts * 1000 +
        distance``: a count of pairwise incompatibilities added to a length in
        µm. Two objections. The units do not match, so the trade-off it encodes
        is arbitrary. And a conflict count is the wrong statistic anyway —
        conflicting with one already-planned move usually costs nothing, because
        the router simply puts this move in the next wave, which frequently
        exists already and is frequently slower. Pricing the marginal wave in
        microseconds removes the constant and the unit clash together, and it
        measured no worse across the suite (the knob's whole dynamic range,
        0 to 1000, spanned 0.008% of total fidelity).

        """
        if vector is None:
            return 0.0
        if self.ablate_wave_cost_model:
            # ABLATION: parallel_priority_weight's rule. Count pairwise AOD
            # incompatibilities against the already-planned vectors and add a
            # raw distance in µm -- no wave bottleneck, no transfer overhead,
            # no time units.
            conflicts = 0
            for batch in batches:
                for u in batch[0]:
                    if not self._compatible_2d(vector, u):
                        conflicts += 1
            return (conflicts * self.parallel_priority_weight
                    + math.dist((vector[0], vector[2]), (vector[1], vector[3])))
        dur = self.movement_duration(
            math.dist((vector[0], vector[2]), (vector[1], vector[3])))
        best = 2 * self.time_atom_transfer + dur          # open a new wave
        compatible = self._compatible_2d
        for batch in batches:
            vecs, bottleneck = batch[0], batch[1]
            # Plain loop rather than all(<genexpr>); same reason as
            # transport.batch_moves — 15.7M generator frames on qft_n150.
            for u in vecs:
                if not compatible(vector, u):
                    break
            else:
                if dur <= bottleneck:
                    # Rides this wave for free, and max(0.0, ...) floors the
                    # score at 0, so nothing later can beat it.
                    return 0.0
                if dur - bottleneck < best:
                    best = dur - bottleneck
        return best

    def _batch_vectors(self, vectors):
        """Greedy first-fit packing of ``vectors`` into modelled waves."""
        return transport.batch_moves(vectors)

    def _absorb_into_batches(self, vector, batches):
        """Insert one ``vector`` into ``batches`` in place, first-fit."""
        return transport.absorb_move(vector, batches)

    def _move_rank(self, q, target_site, batches, prefer_site=None):
        """Lexicographic cost of sending ``q`` to ``target_site``.

        Primary key is the wall-time this move adds to the stage. Secondary key
        is the move's own duration, and it is not redundant: a move that rides
        an existing wave adds exactly zero, so on a busy stage many candidate
        sites tie at zero and the primary key alone would leave the choice to
        list order. The atom still has to travel, and it still has to come back
        later, so among equally-free options the shorter trip wins.

        RE-TESTED 2026-09-12: all 6 orderings of (wave_cost, distance,
        home_trap_tiebreak) swept via a gated `move_rank_order` ablation
        (since reverted). Putting distance first (before wave_cost) measured
        +0.001428 on tqe (2/14 better, 0 regressed, both physical gates
        pass) but -0.112350 on graphs (105 circuits: 17 better/49 worse/39
        same) — a clear net loss, much larger than tqe's gain, in the same
        pattern as the hot/cold-split and reoptimize-pass findings above.
        Current order (wave_cost primary) kept.
        """
        vector = self._vector_for_move(q, target_site)
        prefer_penalty = 0.0 if prefer_site is not None and target_site == prefer_site else 1.0
        return (self._marginal_wave_cost(vector, batches),
                math.dist(self.current_mapping[q], target_site),
                prefer_penalty)

    def init_pairs(self):
        """
        Initialize all valid entanglement pairs based on distance constraints.
        """
        self.all_pairs = [
            (site1, site2)
            for i, site1 in enumerate(self.ent_slm_sites)
            for j, site2 in enumerate(self.ent_slm_sites)
            if i < j and math.dist(site1, site2) < R_B
            ]
        # site -> its partner sites, in all_pairs order. _free_partner_site used
        # to scan every pair to find one site's twin, an O(|all_pairs|) walk per
        # call (110k calls on qft_n150, where the zone holds 2176 traps). Built
        # once here; iteration order within a site's list matches the order the
        # linear scan visited, so "first free twin wins" picks the same site.
        self._twin_map = {}
        for p0, p1 in self.all_pairs:
            self._twin_map.setdefault(p0, []).append(p1)
            self._twin_map.setdefault(p1, []).append(p0)

    def placing(self, index_list_gate):
        """
        Perform qubit placement based on minimal distances.

        Args:
            index_list_gate (int): Index of the gate list to process.
        """
        self.index_list_gate = index_list_gate
        if self.proxy_audit:
            self._audit_seed = self._blank_seed_audit()
        self.init_params()
        # Snapshot of the stage-start mapping (see proxy_stage_forecast).
        self._stage_start_mapping = list(self.current_mapping)
        # Classify qubits into 1-qubit and 2-qubit gates
        self.assign_qubits()
        
        if self.stages[index_list_gate]['type'] != "1qGate":
            self._move_unentangled_qubits()

            # Computed AFTER evacuation, not before: an entanglement pair site
            # freed by an evacuee moving to storage this stage was previously
            # invisible to _process_two_qubit_gates for the rest of the
            # stage, because this list used to be built from the mapping as
            # it stood before any of this stage's moves were committed. Safe
            # to widen here — every move that could free a pair has already
            # been physically committed to current_mapping by this point, so
            # this reflects real state, not a prediction. (_pair_options
            # re-filters against live occupancy anyway; this only affects
            # which candidates are shortlisted in the first place.)
            occupied = set(self.current_mapping)
            self.available_pairs = [(p0, p1) for p0, p1 in self.all_pairs
                                    if p0 not in occupied and p1 not in occupied]
            # Spatial grid bucketing (2026-09-16): _pair_options used to scan
            # ALL of available_pairs (O(zone_size), which architecture/
            # generate.py sizes proportionally to n_q) for every gate, in
            # every gate-order candidate tried -- confirmed by direct
            # instrumentation to be the dominant cost behind NEAT's
            # ~n^1.75 compile-time scaling (ZAC, on a fixed-size architecture
            # with a large upfront cost instead, scales ~n^0.28 and wins the
            # crossover past ~n=550). This buckets available_pairs by a
            # coarse grid over each pair's midpoint so _pair_options can
            # search only the qubit's own cell and its neighbors instead of
            # the whole zone, at the explicit cost of missing a pair whose
            # midpoint falls just outside the searched cells -- a real,
            # accepted fidelity trade (see _pair_options), not a neutral
            # optimization like the _occupied set was.
            cell = self._grid_cell_size
            self._pair_grid = {}
            for p0, p1 in self.available_pairs:
                mx = (p0[0] + p1[0]) / 2
                my = (p0[1] + p1[1]) / 2
                key = (int(mx // cell), int(my // cell))
                self._pair_grid.setdefault(key, []).append((p0, p1))
            # Kept in sync by _commit_move from here on (2026-09-14) so
            # _pair_options stops rebuilding this
            # same set from scratch on every gate placement -- see their
            # docstrings for the measured cost. Rebuilt fresh once per stage
            # here, which is exactly the point their own re-filter already
            # treated as authoritative ("this reflects real state" above).
            self._occupied = occupied

            self._process_two_qubit_gates()

            # REMOVED 2026-09-11: a post-pass, _reoptimize_evacuee_storage,
            # used to run here — it re-ranked each evacuee's storage site
            # (chosen by _move_unentangled_qubits before this stage's 2q-gate
            # targets were known) against final post-gate occupancy, on the
            # theory that more storage was now provably free. Measured: net
            # POSITIVE to remove, on both suites it was checked against —
            # tqe +0.000657 (1 circuit better, 0 regressed), graphs
            # +0.039898 over 105 circuits (39 better, 19 worse, 47 same).
            # Both verify_traces.py and verify_crossings.py still pass with
            # it gone. The pass's own 2-cycle guard and re-ranking logic were
            # real and not a bug (see git history for the full mechanism);
            # it just measured worse than not re-ranking at all, similar in
            # flavor to init_mapping's rejected pair-affinity idea — a second
            # optimization pass fighting the first one's already-good
            # optimum rather than complementing it.

        self._apply_force_override()
        return self.current_mapping

    # Removed: a per-stage contest that ran the whole stage twice (once under
    # 'baseline', once under 'always_stay') and kept whichever scored cheaper
    # by _stage_move_cost() + _stage_crosstalk_cost() (both since deleted).
    # Measured on the tqe suite (14 circuits, 410 stages where the two
    # policies actually diverged): always_stay only won 20/410 (4.9%),
    # concentrated almost entirely in one circuit (sat_n11: 18/122), for a
    # suite total fidelity difference judged not worth 2x'ing every
    # contested stage's placement cost (full _process_two_qubit_gates runs
    # its own inner gate-order search, so re-running it is not cheap). A
    # 1-stage lookahead ON TOP of that removed contest was also tried and
    # rejected earlier (no measurable gain at weight 0.5, net worse at 3.0).
    #
    # ``_count_2q_stages`` and its backing ``_2q_stage_prefix`` array (a
    # per-circuit prefix-sum of stages firing >=1 2q gate) supported this
    # contest and a later crosstalk-count-faithfulness ablation (see the
    # RE-MEASURED 2026-09-12 note in _move_unentangled_qubits); both were
    # deleted 2026-09-13 as unused by any remaining code path. Recover from
    # git history if a similar per-stage-count mechanism is needed again.

    def _move_unentangled_qubits(self):
        """Move unentangled qubits to appropriate storage sites."""
        # Evacuees placed this call. Read by proxy_stage_forecast's
        # seed_reconsidered stat (audit-only); the pass that once revisited
        # these post-hoc (_reoptimize_evacuee_storage) was removed 2026-09-11.
        self._evacuated_this_stage = []
        num_stages = self.results_code['stages']['num_stage']
        # Urgency-sorted evaluation: qubits needed again soonest decide first, so
        # they probe (and implicitly claim, via the immediate current_mapping
        # update in _commit_move) the closest storage sites before less-urgent
        # qubits are considered. Ties keep index order (sorted is stable).
        qubit_order = sorted(range(self.n_q), key=lambda q: -self._urgency(q))
        for q in qubit_order:
            # Only consider qubits that are currently in the entanglement zone
            # and are not participating in a two-qubit gate at this stage.
            if (self.current_mapping[q] in self._ent_set and
                    self.qs_status[q][self.index_list_gate]['status'] != "2qGate"):

                # Find the next stage where this qubit participates in a 2q gate.
                next_2q_index = self._next_2q_stage(q, self.index_list_gate)

                # Determine the range of stages during which this qubit would
                # experience crosstalk if it stays in the entanglement zone.
                if next_2q_index is None:
                    # No future 2q gates for this qubit: consider until the end.
                    idle_end = num_stages
                else:
                    idle_end = next_2q_index
                # Leaving is charged the outbound trip only (r_q = 1, one
                # pick-up and one drop-off), also when q has a future gate
                # (2026-10-29). A staying atom usually moves to
                # its pair site at its next gate too, so the return trip is
                # not extra; charging it (r_q = 2) made evacuation too
                # reluctant. Measured end to end vs r_q = 2: tqe total F
                # 6.691258 -> 6.706651 (4 better, 3 worse: qram_n20 -0.0011,
                # multiplier_n15 -0.0005, sat_n11 -0.0003), qasmbench_std
                # summed -log10 F 6.3784 -> 6.3331 (dnn_n33 -0.040), n<=400
                # qaoa -3.6%, graphs -2.1%, adder +0.34%, qft/ising/random
                # ~0 (evacuation rarely fires there). Same day, rejected:
                # sharing each wave's 2 t_trans + t_b among its evacuees
                # (tqe 13/14 bit-identical, qasmbench_std 6.3749) -- a stage
                # rarely has two compatible evacuees, and gate moves joining
                # an evacuee's wave later cut its real cost only ~20%.
                num_transfer_events = 2

                # Stage count for the 'stay' cost. Deliberately counts EVERY
                # stage, not just the ~50% that fire 2q gates and actually
                # charge crosstalk in the simulator (Router.write_idle only
                # fires on a stage with >=1 2q gate elsewhere): the ~2x
                # overcount acts as a calibrated proxy for the unmodeled
                # congestion cost of idling in the zone (squatting on pair
                # sites degrades active gates' placement).
                #
                # RE-MEASURED 2026-09-12 across 5 suites (via a gated
                # `_count_2q_stages`-based ablation, since reverted):
                #   tqe                -0.017386 (9/14 regressed, 2 better)
                #   scalability/adder  -0.005387 (3/23 regressed, 1 better)
                #   scalability/qft    +0.002864 (1/26 better, 2 regressed)
                #   graphs             bit-identical (the faithful count
                #                      never differs from the overcount on
                #                      that suite's stage pattern)
                #   scalability/random not measured (run killed mid-compile
                #                      early; never resumed)
                # Net worse on the two suites where it changes anything
                # meaningfully. Mechanism, confirmed by a per-decision trace
                # on tqe/qft_n10, sat_n11, qram_n20: the faithful count is
                # almost exactly half the overcount (ratio 0.50-0.52,
                # matching the ~50% 2q-gate-stage fraction), which halves
                # crosstalk_cost while leaving move_cost completely
                # unchanged (move_cost has no dependence on stage count at
                # all — see its computation below). Shrinking one side of
                # `crosstalk_cost > move_cost` can only ever push decisions
                # toward "stay", never "move" — confirmed directly: every
                # flipped decision traced went move->stay, zero went the
                # other way (92/184 flips on sat_n11 alone). A qubit that
                # would have evacuated (freeing a pair site for actively-
                # gating qubits that stage) now squats instead — the
                # regression shows up as a placement-quality cost to OTHER
                # qubits' gates that stage, not a term on this qubit itself,
                # which is why it doesn't show up as one traceable number.
                # Keep the deliberate overcount.
                # CHANGED 2026-10-18 (faithful to the paper's
                # T_stay = k_q tau_idle): count only stages with a 2q gate,
                # the only ones that charge crosstalk. The all-stage overcount
                # described above measured better (tqe ~ +0.02, qasmbench_std
                # ~ +0.03 with it); traded for faithfulness on purpose.
                pre = self._prefix_2q()
                crosstalk_stage_count = pre[idle_end] - pre[self.index_list_gate]

                # --- Estimate crosstalk cost (log-fidelity loss) ---
                # Each affected stage applies fidelity_2q_idle once to this qubit.
                crosstalk_cost = crosstalk_stage_count * (-math.log(self.fidelity_2q_idle))

                # --- Estimate movement and decoherence cost if we move to storage ---
                # Atom-transfer cost (estimated).
                transfer_cost = num_transfer_events * (-math.log(self.fidelity_atom_transfer))

                # Estimate the site this qubit would actually be sent to, using
                # the same rank _assign_storage_site_ranked uses for the real
                # move (_pick_ranked_site: marginal wave cost, then distance,
                # then home-trap tiebreak) — one site-selection rule for both
                # "would evacuating be worth it" and "where would it go",
                # instead of a separate ad hoc rule for the estimate alone.
                #
                # REPLACED 2026-09-12 a two-branch predecessor (home trap only
                # if free, else nearest-by-raw-distance among ALL free sites —
                # "F4", 2026-09-02) that gave home
                # trap a much stronger role here than anywhere else in the
                # file: whenever free, it was the ONLY candidate considered,
                # not merely a tiebreak, inconsistent with home trap's role
                # everywhere else post the 2026-09-11 hot/cold-split removal
                # (see _assign_storage_site_ranked). Measured: tqe +0.000658
                # (1/14 better — sat_n11 — 0 regressed), both physical gates
                # pass; graphs not measured (user judged the tqe result
                # sufficient given this change's rationale is simplicity and
                # consistency, not raw fidelity gain).
                best_storage_site = self._pick_ranked_site(q)
                if best_storage_site is None:
                    # No available storage site: cannot move this qubit now.
                    continue
                # T_leave = r_q * Delta T_eqv({v}) (2026-10-14;
                # r_q = 1 since 2026-10-29, see above):
                # each of the r_q = num_transfer_events / 2 trips is priced by
                # the same marginal wave cost that chose best_storage_site,
                # against the waves already planned this stage. Until
                # 2026-10-14 each trip was priced t(d) alone, with no pick-up
                # / drop-off time. Measured on 37 sample circuits (tqe,
                # qasmbench_std, graphs/qft/random/adder/qaoa at n~100/300),
                # summed -log10 F vs t(d): tqe total F 6.695328 -> 6.695473
                # (1 better / 1 worse), qasmbench_std 4.652588 -> 4.651383,
                # graphs +4.0% (graphs_300 alone; over n=100..1000 ~+0.4%,
                # both signs), qaoa +1.0%, adder +1.0%, qft -0.2%, random 0.
                # 96-100% of evacuees open a new wave at decision time, so
                # this is ~ r_q (2 t_trans + t(d)). Rejected the same day:
                # outbound marginal + return t(d), a forecast of the stage's
                # gate-move bottleneck, a stage count of Rydberg stages only,
                # ZAC's evacuate-unless-needed-next rule, always evacuating,
                # deciding evacuations after pair selection, and a weight w
                # in {0.5, 0.75} on the whole leave cost.
                vector = self._vector_for_move(q, best_storage_site)
                approx_movement_time = (0.5 * num_transfer_events
                                        * self._marginal_wave_cost(vector, self._planned_batches))

                # Decoherence penalty: movement introduces additional time during which
                # some qubits are idle. For a small segment dt, -log(exp(-dt/T2)) ≈ dt/T2.
                # n_q, not n_q - 1 (2026-10-18): the paper's tau uses n.
                decoherence_cost = self.n_q * approx_movement_time / self.time_coherence

                move_cost = transfer_cost + decoherence_cost

                if self.ablate_fidelity_cost_model:
                    # ABLATION: strip the physics. Compare RAW COUNTS -- stages
                    # this qubit would sit in the zone, against the number of
                    # atom-transfer events evacuating would cost -- instead of
                    # converting both to commensurable -log(fidelity) units.
                    # The decision rule below is unchanged; only the units it
                    # compares are. This is the unit-mismatch the shipped model
                    # exists to remove, and isolates the cost MODEL from the
                    # stay/evacuate POLICY (ablate_idle_policy).
                    crosstalk_cost = float(crosstalk_stage_count)
                    move_cost = float(num_transfer_events)

                # If the expected crosstalk loss dominates the movement+decoherence loss,
                # we move this qubit to storage; otherwise keep it in the entanglement zone.
                # REMOVED 2026-09-13: routing_strategy's always_move/always_stay
                # forced-policy alternates and this function's policy_override
                # parameter, both used only for the A/B contest documented
                # below (measured and removed 2026-09-02) — no remaining call
                # site ever passed a non-None override, and no shipped config
                # ever set routing_strategy to anything but "baseline".
                # ABLATION (ablate_idle_policy): never evacuate -- idle atoms
                # stay in the entanglement zone and absorb crosstalk.
                if ((crosstalk_cost > move_cost
                     or os.environ.get("ZAP_FORCE_EVAC") == "1")
                        and not self.ablate_idle_policy):
                    if self._assign_storage_site_ranked(q):
                        self._evacuated_this_stage.append(q)

            # Storage residents are left where they are. A "retiring resident
            # drifts back to its home trap" move used to live here; removed
            # 2026-09-11 after measuring it fires 0 times on tqe/adder/random
            # and 21,644 times on graphs (105 circuits) yet leaves fidelity
            # bit-identical in every case (105/105 unchanged) — storage sites
            # carry no crosstalk cost, so the move's only payoff was shaving a
            # future hop by tens-to-hundreds of µs against T2 (~1.5e6 µs),
            # while its atom-transfer fidelity hit turned out to be too small
            # to register either; net measured effect is zero, not negative.
            # A prior cache-refresh variant that also repositioned *live*
            # (still-gating) residents was tried and measured net negative
            # (see git history) — that finding is unrelated to this removal.

    @staticmethod
    def _blank_seed_audit():
        # NOTE: reopt_saw_seeded/reopt_seeded_taken/reopt_seeded_cycle are
        # dead since _reoptimize_evacuee_storage was removed 2026-09-11 — the
        # only writer is gone, so these stay 0 forever. Kept (not deleted) so
        # tools/proxy_audit_report.py's reads of them don't KeyError; that
        # tool now always reports 0 for this section.
        return {'storage_placements': 0, 'with_extra_sites': 0, 'flips': 0,
                'flipped': {}, 'reopt_saw_seeded': 0,
                'reopt_seeded_taken': 0, 'reopt_seeded_cycle': 0}

    def _departure_sites(self, q):
        """Storage traps that are occupied now but provably free by stage end.

        Every qubit firing a 2q gate this stage must end up in the entanglement
        zone — that is the only place a 2q gate can execute — so a storage trap
        held by one of them is guaranteed to be vacated, and is knowable from
        the schedule before any placement decision is made. ``_move_unentangled_
        qubits`` runs before ``_process_two_qubit_gates``, so it sees these
        traps as occupied and will not send an evacuee to one.

        Audit-only: this is used to MEASURE how often that blindness changes a
        decision, not to change one. Acting on it directly inside
        ``_move_unentangled_qubits`` is the fix that was tried and reverted —
        the gate resident's own target is not yet known at that point, and if
        it lands on the trap the evacuee is vacating the two moves form a
        direct position swap, which ``compatible_2D`` forbids on every axis
        (double occupancy on 10/14 tqe circuits when tried). A post-pass,
        ``_reoptimize_evacuee_storage``, used to re-decide evacuee sites after
        gate targets were real (see the comments in ``placing``) — removed
        2026-09-11, net positive on both tqe and graphs; see that removal note.
        """
        sites = set()
        for other in range(self.n_q):
            if other == q:
                continue
            if self.qs_status[other][self.index_list_gate]['status'] != "2qGate":
                continue
            p = self.current_mapping[other]
            if p in self._stg_set:
                sites.add(p)
        return sites

    def _audit_departure_seeding(self, q, chosen, free):
        """Count evacuations whose site would change if departures were seeded."""
        self._audit_seed['storage_placements'] += 1
        extra = [s for s in self._departure_sites(q) if s not in free]
        if not extra:
            return
        self._audit_seed['with_extra_sites'] += 1
        home = self.initial_mapping[q]
        prefer = home if (home in free or home in extra) else None
        seeded = min(list(free) + extra,
                     key=lambda s: self._move_rank(q, s, self._planned_batches,
                                                   prefer_site=prefer))
        if seeded != chosen:
            self._audit_seed['flips'] += 1
            self._audit_seed['flipped'][q] = seeded

    def _assign_storage_site_ranked(self, q):
        """Cache-style storage assignment for a qubit leaving the entanglement zone.

        Instead of the nearest free site (urgency-blind), rank free sites by
        dmin. Because callers iterate qubits in descending urgency and each
        commit immediately claims its site, the most-urgent qubit gets first
        pick of the near-boundary sites and low-urgency qubits drift deep —
        the proximity gradient emerges without declaring any cache zone.

        residual_xtalk is deliberately 0 here: ZAP's simulator only charges
        crosstalk to atoms inside the entanglement zone (Router.write_idle
        checks y >= zone edge), so a near-boundary storage site is exactly as
        crosstalk-free as deep storage under the actual cost model.

        UNIFIED 2026-09-11: the former hot(gap<=1)/cold(gap>cold_gap_threshold)/
        windowed(gap 2-3) three-way split is gone — every qubit now goes
        through _pick_ranked_site's single unconditional rank over all free
        sites (home-trap preference included as _move_rank's tiebreak, not a
        separate branch). Measured, both suites where the split ever fired
        (tqe: 51 hot/windowed calls total per placer.md; graphs: far more):
        tqe +0.001133 (6.948373->6.949507, 4/14 moved: sat_n11 +0.001309,
        knn_n25 +0.000186 better, bv_n14 -0.000087, ghz_n30 -0.000275 worse),
        graphs -0.005198 (18.364621->18.359423 over 105 circuits, 11 better/
        7 worse/87 unchanged, worst single regression graphs_30_1 -0.006174).
        graphs's loss outweighs tqe's gain in magnitude; shipped anyway
        (2026-09-11) after weighing the trade,
        prioritising tqe as the floor suite over graphs as the secondary
        never-tuned-against check. Both verify_traces.py and
        verify_crossings.py still pass on both suites.
        """
        best = self._pick_ranked_site(q)
        if best is None:
            return False
        self._commit_move(q, best)
        return True

    def _baseline_storage_site(self, q):
        """Baseline zap's evacuation destination: home trap, else nearest free.

        Measurement path for ``ablate_evac_rank`` only. Mirrors
        ``baselines/zap/placer.py::_move_unentangled_qubits``: prefer the qubit's
        original trap when free, otherwise the closest free storage site by
        raw distance -- no wave-cost term anywhere.
        """
        taken = {p for i, p in enumerate(self.current_mapping) if i != q}
        home = self.initial_mapping[q]
        if home in self._stg_set and home not in taken:
            return home
        here = self.current_mapping[q]
        best, best_d = None, None
        for site in self.stg_slm_sites:
            if site in taken:
                continue
            d = math.dist(site, here)
            if best_d is None or d < best_d:
                best, best_d = site, d
        return best

    def _pick_ranked_site(self, q):
        """Choose (without committing) the ranked storage site for q.

        Ranks every free storage site by _move_rank (marginal AOD-wave cost,
        then raw distance, then a home-trap-preference tiebreak) and takes the
        best. See _assign_storage_site_ranked's docstring for the 2026-09-11
        measurement that replaced the former hot/cold/windowed gap-based
        split with this single unconditional rank.

        """
        if self.ablate_evac_rank:
            return self._baseline_storage_site(q)
        # One occupancy set instead of an O(n_q) scan per candidate site: this
        # comprehension was the second hottest line in the compiler.
        taken = {p for i, p in enumerate(self.current_mapping) if i != q}

        # Spatial grid restriction (2026-09-16), mirroring _pair_options:
        # only rank storage sites near q's current site instead of every
        # free site in the zone. Same deliberate compile-time/fidelity trade
        # -- a closer-but-outside-radius site can be missed. Expands radius
        # if the local neighborhood has nothing free, so it degrades
        # gracefully rather than ever returning nothing when a free site
        # exists somewhere in the zone.
        qx, qy = self.current_mapping[q]
        cell = self._stg_grid_cell_size
        gx = int(qx // cell)
        gy = int(qy // cell)
        grid = self._stg_grid
        radius = self._stg_grid_search_radius
        free = []
        while True:
            free = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for s in grid.get((gx + dx, gy + dy), ()):
                        if s not in taken:
                            free.append(s)
            if free or radius >= 12:
                break
            radius += 1
        if not free:
            return None

        batches = self._planned_batches
        best = min(free, key=lambda s: self._move_rank(q, s, batches))
        if self.proxy_audit:
            self._audit_departure_seeding(q, best, free)
        return best

    def proxy_stage_forecast(self):
        """Phase-1 proxy audit: what this stage's placement is PREDICTED to cost.

        Called by ``Router._record_proxy_audit`` right after ``placing()``
        returns, and only when ``routing.proxy_audit`` is on — it costs nothing
        on a normal compile. The prediction is the stage-net
        vectors, greedy first-fit packed by ``_batch_vectors``, one bottleneck
        duration plus one activate/deactivate pair per resulting wave.

        Also measures the audit's category (b), *sequential-commitment error*:
        the placer prices each move against the batches that existed when it
        was decided, which is a prefix of the stage. ``_planned_vectors`` holds
        those moves in commit order, so repricing move i against the prefix
        ``[0, i)`` reconstructs precisely what the placer saw, and repricing it
        against every OTHER move in the stage says what it would have paid with
        perfect hindsight. A move whose two prices straddle the new-wave price
        was mispriced by a later placement it could not see.
        """
        vectors = []
        for q in range(self.n_q):
            a = self._stage_start_mapping[q]
            b = self.current_mapping[q]
            if a != b:
                vectors.append((a[0], b[0], a[1], b[1]))
        batches = self._batch_vectors(vectors)

        # Commit order, restricted to the moves that survived into the final
        # stage-net set. Historically `_reoptimize_evacuee_storage` could
        # rewind and re-commit an evacuee, leaving its round-1 placeholder
        # vector behind in `_planned_vectors` (never physically made, so not
        # part of what the router routes) — that pass is gone (removed
        # 2026-09-11), so `stale_planned` below is now always 0, but the
        # filter is kept since it's still correct in general.
        net = set(vectors)
        commit_order, seen = [], set()
        for v in self._planned_vectors:
            if v in net and v not in seen:
                seen.add(v)
                commit_order.append(v)
        stale_planned = sum(1 for v in self._planned_vectors if v not in net)
        for v in vectors:            # committed outside the tracked path
            if v not in seen:
                seen.add(v)
                commit_order.append(v)

        n_misprice = n_misclass = 0
        for i, v in enumerate(commit_order):
            new_wave = 2 * self.time_atom_transfer + self.movement_duration(
                math.dist((v[0], v[2]), (v[1], v[3])))
            p_decision = self._marginal_wave_cost(
                v, self._batch_vectors(commit_order[:i]))
            p_hindsight = self._marginal_wave_cost(
                v, self._batch_vectors([u for u in commit_order if u != v]))
            if abs(p_decision - p_hindsight) > 1e-9:
                n_misprice += 1
            if ((p_decision >= new_wave - 1e-9)
                    != (p_hindsight >= new_wave - 1e-9)):
                n_misclass += 1

        return {
            'pred_waves': len(batches),
            'pred_move_time': sum(b[1] for b in batches),
            'pred_transfer_time': 2 * self.time_atom_transfer * len(batches),
            'pred_duration': (sum(b[1] for b in batches)
                              + 2 * self.time_atom_transfer * len(batches)),
            'pred_n_tr': 2 * len(vectors),
            'n_vectors': len(vectors),
            'n_misprice': n_misprice,
            'n_misclass': n_misclass,
            'n_stale_planned_vectors': stale_planned,
            # Departure-set seeding (Phase 2). `flips` counts storage
            # placements that WOULD have gone elsewhere had the round-1 pass
            # known which traps this stage's gate residents are about to
            # vacate. `seed_corrected` used to count how many of those the
            # now-removed `_reoptimize_evacuee_storage` pass landed on anyway
            # once gate targets were real — that pass is gone (2026-09-11),
            # so `current_mapping[q]` here is just q's original round-1 site
            # and `seed_corrected` measures nothing meaningful any more.
            'seed_storage_placements': self._audit_seed['storage_placements'],
            'seed_with_extra_sites': self._audit_seed['with_extra_sites'],
            'seed_flips': self._audit_seed['flips'],
            'seed_corrected': sum(
                1 for q, s in self._audit_seed['flipped'].items()
                if self.current_mapping[q] == s),
            # Was the trap seeding would have picked actually free once the
            # stage finished? If not, the "guaranteed departure" was claimed by
            # somebody else and seeding would have bought nothing here.
            'seed_still_free': sum(
                1 for q, s in self._audit_seed['flipped'].items()
                if s not in {p for i, p in enumerate(self.current_mapping)
                             if i != q}),
            'seed_reopt_saw': self._audit_seed['reopt_saw_seeded'],
            'seed_reopt_taken': self._audit_seed['reopt_seeded_taken'],
            'seed_reopt_cycle': self._audit_seed['reopt_seeded_cycle'],
            'seed_reconsidered': sum(
                1 for q in self._audit_seed['flipped']
                if q in self._evacuated_this_stage),
        }

    def _process_two_qubit_gates(self):
        """Assign entanglement pair sites to this stage's 2q gates.

        Gates claim pairs one at a time, longest span first (ties in circuit
        order): the gate needing the longest move claims its pair before
        shorter moves fragment the free pool.

        REPLACED 2026-10-14 the gate-order search, which placed
        each stage under several candidate orders (circuit order, most-
        constrained-first at slack 7, longest span first) and kept the
        cheapest by the placer's stage cost model. Measured on every paper
        circuit with the tools/regen timing rules: against the search, tqe
        6.724937 -> 6.695 total F and qasmbench_std 4.676 -> 4.653 (still above
        every other compiler on all 13); against circuit order, which the
        scalability families then used, summed -log10 F qft -11.5%, random
        -9.6%, graphs +2.5%, qaoa +1.5%, adder +1.1%. The search itself was
        better still where measured (qft_n500 156.7 vs 167.1) at ~1.3x compile
        time. Recover it from git history.
        """
        self._assign_gate_order(sorted(self.gate_2q, key=lambda g: -math.dist(
            self.current_mapping[g[0]], self.current_mapping[g[1]])))

    def _twin_and_holder(self, site, ignore_q):
        """Return (twin_site, occupying_qubit) for ``site``'s Rydberg pair.

        Used to identify a third-party qubit squatting the twin of a resident's
        site — the common blocker preventing zone residency.
        """
        for p0, p1 in self.all_pairs:
            if site not in (p0, p1):
                continue
            twin = p1 if site == p0 else p0
            for i, m in enumerate(self.current_mapping):
                if m == twin and i != ignore_q:
                    return twin, i
        return None, None


    def _free_partner_site(self, site, ignore_q=None):
        """The unoccupied twin of ``site`` in some Rydberg pair, or None.

        ``ignore_q`` treats that qubit's trap as free — used when the qubit
        about to move INTO the twin is the one currently sitting there or
        elsewhere, so its own position must not veto the assignment.
        """
        twins = self._twin_map.get(site)
        if not twins:
            return None
        occupied = {m for i, m in enumerate(self.current_mapping) if i != ignore_q}
        for twin in twins:
            if twin not in occupied:
                return twin
        return None

    def _assign_gate_order(self, gate_order):
        """Run the pair-site assignment loop for one candidate gate order.

        Each gate is satisfied by the cheapest of its enumerated placements (see
        ``_pair_options`` / ``_place_gate``). Anchoring — holding a qubit that is
        already in the zone and bringing only its partner — is no longer a
        special branch but simply one of those candidates, so it is taken when
        it is actually cheaper rather than whenever it is merely possible, and
        WHICH qubit anchors is decided by cost instead of by position in the
        gate tuple.
        """
        for q0, q1 in gate_order:
            # Already sitting on a valid Rydberg pair: nothing to do, and no
            # candidate can beat moving zero atoms.
            if ((self.current_mapping[q0], self.current_mapping[q1]) in self.all_pairs or
                    (self.current_mapping[q1], self.current_mapping[q0]) in self.all_pairs):
                continue

            if self._place_gate(q0, q1):
                continue

            # No candidate at all: every twin is taken and no free pair is left.
            # The blocker may be an idle third qubit squatting the twin of a
            # qubit already in the zone. Evicting it costs its 2 transfers,
            # while anchoring saves the resident's 2 transfers AND both long
            # moves — worth it when the squatter is not gating this stage.
            for cand, other in ((q0, q1), (q1, q0)):
                site = self.current_mapping[cand]
                if site not in self._ent_set:
                    continue
                twin, squatter = self._twin_and_holder(site, other)
                if twin is None or squatter is None:
                    continue
                if self.qs_status[squatter][self.index_list_gate]['status'] == "2qGate":
                    continue  # squatter needs the zone itself
                self._move_to_target_zone(squatter, "storage")
                if self.current_mapping[squatter] != twin:
                    break

            if not self._place_gate(q0, q1):
                raise ValueError(
                    f"No valid pair site found for gate ({q0}, {q1})")
         
         
    def assign_qubits(self):
        """
        Classify gates into 1-qubit and 2-qubit gates.
        """
        self.execute_qubits = list(set(q for gate in self.list_gate for q in gate))
        for gate in self.list_gate:
            q0, q1 = gate
            if q0 == q1:
                self.execute_qubits_1q.append(q0)
            else:
                self.gate_2q.append((q0, q1))
                self.execute_qubits_2q += [q0, q1]

    def _move_to_target_zone(self, q, zone_type):
        """Move qubit q into the storage zone, preferring its original trap.

        Args:
            q (int): Index of the qubit to move.
            zone_type (str): Target zone; only ``"storage"`` is used.

        Raises:
            ValueError: If no available site is found in the specified zone.
        """
        # One occupancy set instead of an O(n_q) scan per candidate site (see
        # _pick_ranked_site for the same fix).
        taken = {p for i, p in enumerate(self.current_mapping) if i != q}
        available_sites = [site for site in self.stg_slm_sites if site not in taken]

        # The original trap is a preference, not a hard constraint.
        original_trap = self.initial_mapping[q]
        prefer_site = original_trap if (
            original_trap in self._stg_set and original_trap in available_sites
        ) else None
        batches = self._planned_batches
        available_sites = sorted(
            available_sites,
            key=lambda site: self._move_rank(q, site, batches, prefer_site=prefer_site)
        )

        # Update mapping or raise error if no valid site is found
        if available_sites:
            if self.proxy_audit and zone_type == "storage":
                self._audit_departure_seeding(q, available_sites[0],
                                              set(available_sites))
            self._commit_move(q, available_sites[0])
        else:
            raise ValueError(f"No available sites in {zone_type} zone for qubit {q}")

    # ---- unified pair-site selection -------------------------------------
    #
    # Every way of satisfying a 2q gate — hold q0 and fetch q1, hold q1 and
    # fetch q0, or move both to a free pair — is enumerated as a candidate and
    # priced by ONE cost function in wall-time microseconds. This replaces a
    # branch cascade whose tie-breaks were arbitrary (it anchored whichever
    # qubit appeared first in the gate tuple) and a ``conflicts * 1000 +
    # distance`` score that added a count to a length.

    def _site_storage_dist(self, site):
        """Distance (µm) from an entanglement site to the nearest storage trap."""
        cache = self.__dict__.setdefault('_site_stg_cache', {})
        if site not in cache:
            cache[site] = min(math.dist(site, s) for s in self.stg_slm_sites)
        return cache[site]

    def _stage_base_plan(self, exclude):
        """This stage's committed movement, ignoring the qubits in ``exclude``.

        Vectors are NET displacement since the stage began, which is what the
        router actually emits: it diffs the current mapping against the final
        one once per stage, so a qubit shuffled twice costs one move and a qubit
        that returns to where it started costs none. Scoring candidates against
        per-commit vectors instead measurably loses fidelity, because it charges
        for intermediate positions the hardware never visits.

        Returns ``(batches, n_movers)``. Computed once per gate; each candidate
        then only has to add its own one or two vectors.
        """
        vectors = []
        for q in range(self.n_q):
            if q in exclude:
                continue
            a, b = self._stage_start_mapping[q], self.current_mapping[q]
            if a != b:
                vectors.append((a[0], b[0], a[1], b[1]))
        # Consecutive gates' base plans share long prefixes in qubit order;
        # FirstFitCache resumes the packing after the shared prefix and returns
        # exactly _batch_vectors(vectors) (2026-10-03, bit-identical).
        cache = getattr(self, "_base_plan_cache", None)
        if cache is None:
            cache = self._base_plan_cache = transport.FirstFitCache()
        return cache(vectors), len(vectors)

    def _score_moves(self, targets, base_batches, base_movers, extra_cost=0.0):
        """Total stage rearrangement cost (µs) if ``targets`` were committed.

        Same model as ``proxy_stage_forecast`` — greedy AOD wave batching, each
        wave's bottleneck plus its activate/deactivate overhead, plus every
        moving atom's pickup/drop loss converted to decoherence-equivalent time
        — evaluated on the base plan with this gate's qubits added back.

        Pricing AOD parallelism this way, in microseconds, is what replaces the
        ``parallel_priority_weight`` conflict count: a move that rides an
        existing wave costs nothing because ``movement_duration`` is concave,
        and one that forces a new wave pays a number the hardware model
        actually produces.

        ``base_batches`` is consumed (pass a copy).
        """
        if self.ablate_wave_cost_model:
            # ABLATION: score the gate by conflicts+distance, matching
            # _marginal_wave_cost's ablated form (and baseline _pair_move_rank).
            conflicts = 0.0; dist = 0.0
            for q, target in targets.items():
                a = self._stage_start_mapping[q]
                if a == target:
                    continue
                v = (a[0], target[0], a[1], target[1])
                dist += math.dist(a, target)
                for batch in base_batches:
                    for u in batch[0]:
                        if not self._compatible_2d(v, u):
                            conflicts += 1
            return conflicts * self.parallel_priority_weight + dist + extra_cost
        movers = base_movers
        for q, target in targets.items():
            a = self._stage_start_mapping[q]
            if a != target:
                movers += 1
                self._absorb_vector((a[0], target[0], a[1], target[1]), base_batches)
        transfer_logloss = 2 * (-math.log(self.fidelity_atom_transfer)) * movers
        return (sum(b[1] for b in base_batches)
                + 2 * self.time_atom_transfer * len(base_batches)
                + transfer_logloss * self.time_coherence / max(1, self.n_q)
                + extra_cost)

    def _baseline_closest_pair(self, q0, q1):
        """Baseline zap's pair choice: all free pairs, conflicts*1000+distance.

        Measurement path for ``ablate_pair_scoring`` only. Mirrors
        ``baselines/zap/placer.py::_find_closest_pair`` including its orientation
        rule (assign the pair the way that minimises summed inbound distance,
        then rank). Returns a one-element option list so ``_place_gate``
        commits it directly.
        """
        occupied = self._occupied
        m0, m1 = self.current_mapping[q0], self.current_mapping[q1]
        best_rank, best = None, None
        for p0, p1 in self.available_pairs:
            if p0 in occupied or p1 in occupied:
                continue
            d1 = math.dist(p0, m0) + math.dist(p1, m1)
            d2 = math.dist(p1, m0) + math.dist(p0, m1)
            a, b = (p0, p1) if d1 <= d2 else (p1, p0)
            v0 = self._vector_for_move(q0, a)
            v1 = self._vector_for_move(q1, b)
            conflicts = 0
            for v in (v0, v1):
                if v is None:
                    continue
                for batch in self._planned_batches:
                    for u in batch[0]:
                        if not self._compatible_2d(v, u):
                            conflicts += 1
            if v0 is not None and v1 is not None and not self._compatible_2d(v0, v1):
                conflicts += 1
            rank = (conflicts * self.parallel_priority_weight
                    + math.dist(m0, a) + math.dist(m1, b))
            if best_rank is None or rank < best_rank:
                best_rank, best = rank, [(q0, a), (q1, b)]
        return [best] if best else []

    def _pair_options(self, q0, q1, limit=None):
        """Every candidate placement for gate ``(q0, q1)``.

        Anchor options come first: holding a qubit already in the zone saves its
        pickup+drop and both long moves, which is usually decisive. Free pairs
        are shortlisted by a cheap proxy of the real score (inbound distance
        plus the future-travel term) so the expensive wave-model scoring runs on
        a handful of candidates rather than every free pair in the zone.
        """
        limit = self.pair_candidate_limit if limit is None else limit
        # ABLATION (ablate_pair_scoring): shortlist exactly one candidate -- the
        # nearest free pair by the distance proxy -- so _place_gate's wave-model
        # scoring has nothing to choose between and the placer falls back to
        # "put the gate on the closest free pair".
        if self.ablate_pair_scoring:
            # ABLATION: baseline zap's _find_closest_pair -- scan EVERY free
            # pair and rank by _pair_move_rank (conflicts*1000 + distance),
            # rather than wave-model-scoring a 4-candidate shortlist. Handing
            # _place_gate a single pre-ranked option reproduces that choice
            # exactly, since zap commits the argmin with no further scoring.
            return self._baseline_closest_pair(q0, q1)
        opts = []
        for anchor, mover in ((q0, q1), (q1, q0)):
            site = self.current_mapping[anchor]
            if site not in self._ent_set:
                continue
            twin = self._free_partner_site(site, ignore_q=mover)
            if twin is None:
                continue
            opts.append([(mover, twin)])

        shortlist = []
        # Incrementally maintained by _commit_move (2026-09-14), not rebuilt
        # here -- this used to be `set(self.current_mapping)` on every call,
        # an O(n_q) rebuild on top of the O(zone_size) scan below, at up to
        # ~94k calls on the largest circuits. See _commit_move and the
        # bulk-restore points in _process_two_qubit_gates for what keeps it
        # exact across gate-order-candidate rollbacks.
        occupied = self._occupied
        m_q0 = self.current_mapping[q0]
        m_q1 = self.current_mapping[q1]
        # Hot loop: every free pair in the zone, twice. Both distances are
        # measured from two fixed points, so the coordinates are hoisted and
        # the hypotenuses computed inline — math.dist builds and unpacks two
        # tuples per call, which at 31.9M calls/circuit outweighs the sqrt.
        # Same value, same order.
        #
        # This used to be the ``else`` of a test on future_travel_weight, whose
        # weighted branch added a _future_travel_cost term to each proxy. Both
        # the knob and the term were removed 2026-09-05 (see Placer.__init__)
        # because the shipped weight of 0.0 made that term identically zero.
        x0, y0 = m_q0
        x1, y1 = m_q1
        sqrt = math.sqrt
        append = shortlist.append
        # Spatial grid restriction (2026-09-16): scan only the cells around
        # the gate's own midpoint instead of every pair in the zone -- see
        # placing()'s _pair_grid construction for the mechanism. Expands the
        # search radius if too few raw candidates turn up (an empty or
        # sparse neighborhood, e.g. near the zone edge), so this degrades
        # gracefully rather than silently returning nothing; it still never
        # falls back to a true full-zone scan, which is the deliberate
        # fidelity/compile-time trade being made here.
        cell = self._grid_cell_size
        gx = int(((x0 + x1) / 2) // cell)
        gy = int(((y0 + y1) / 2) // cell)
        grid = self._pair_grid
        radius = self._grid_search_radius
        seen_ids = set()
        while True:
            n_candidates = 0
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for p0, p1 in grid.get((gx + dx, gy + dy), ()):
                        if p0 in occupied or p1 in occupied:
                            continue
                        n_candidates += 1
                        pid = (p0, p1)
                        if pid in seen_ids:
                            continue
                        seen_ids.add(pid)
                        a0, a1 = p0
                        b0, b1 = p1
                        da0 = sqrt((a0 - x0) ** 2 + (a1 - y0) ** 2)
                        da1 = sqrt((a0 - x1) ** 2 + (a1 - y1) ** 2)
                        db0 = sqrt((b0 - x0) ** 2 + (b1 - y0) ** 2)
                        db1 = sqrt((b0 - x1) ** 2 + (b1 - y1) ** 2)
                        append((da0 + db1, (p0, p1)))
                        append((db0 + da1, (p1, p0)))
            if n_candidates >= limit or radius >= 6:
                break
            radius += 1
        # Only `limit` distinct sites are ever taken (default 4), but the zone
        # can hold thousands of free pairs — 2176 entanglement traps on
        # scale_to_500 — so a full sort orders thousands to consume a handful.
        # heapq.nsmallest is O(n log limit) against sort's O(n log n) and, like
        # sort, is stable: equal proxies keep insertion order, so the candidate
        # actually chosen is unchanged. Duplicate (a, b) entries are possible
        # (a pair and its reverse can tie), so ask for `limit` DISTINCT sites by
        # over-fetching to 2*limit — the dedup loop below still guards the count.
        # Verified bit-identical on tqe.
        if len(shortlist) > 2 * limit:
            shortlist = heapq.nsmallest(2 * limit, shortlist, key=lambda t: t[0])
        else:
            shortlist.sort(key=lambda t: t[0])
        seen = set()
        for _proxy, (a, b) in shortlist:
            if len(seen) >= limit:
                break
            if (a, b) in seen:
                continue
            seen.add((a, b))
            opts.append([(q0, a), (q1, b)])
        return opts

    def _place_gate(self, q0, q1):
        """Commit the lowest-cost placement for gate ``(q0, q1)``; False if none.

        Ranking is lexicographic: (modelled cost, total distance moved,
        distance to storage). The primary key is the modelled cost; the
        other two settle what the cost model is indifferent about, and that
        indifference is common — anchoring q0 and anchoring q1 often cost
        exactly the same, because the two atoms move the same distance. The
        distance-to-storage key then breaks a same-cost/same-distance tie by
        preferring the resting position a row closer to everything both
        atoms must reach later ("move q2 up rather than q0 down" — same
        distance travelled, same gate, worse resting position otherwise).

        This tiebreak can never override a real cost difference, so it is not
        a blanket preference for near-storage sites — that variant was
        measured (``storage_proximity_weight``) and cost -0.010 net across
        the suite. Two other tiebreak orderings (distance-to-storage-first,
        and no tiebreak at all — first candidate wins) were config-driven
        alternates (``pair_tiebreak``) used only for A/B sweeps in the now-
        deleted ``tools/sweep_configs.py``; removed 2026-09-13 since no
        shipped config ever set this to anything but the ordering below.
        """
        opts = self._pair_options(q0, q1)
        if not opts:
            return False
        # Batch everything this stage has already committed once; candidates
        # differ only in where q0 and q1 end up, so each is priced by adding
        # two vectors to a cheap copy rather than re-batching the whole stage.
        base_batches, base_movers = self._stage_base_plan({q0, q1})
        best = None
        for moves in opts:
            move_map = dict(moves)
            targets = {q0: move_map.get(q0, self.current_mapping[q0]),
                       q1: move_map.get(q1, self.current_mapping[q1])}
            local = [[list(b[0]), b[1]] for b in base_batches]
            cost = self._score_moves(targets, local, base_movers)
            distance = sum(math.dist(self.current_mapping[q], t) for q, t in moves)
            to_storage = sum(self._site_storage_dist(s) for s in targets.values())
            rank = (cost, distance, to_storage)
            if best is None or rank < best[0]:
                best = (rank, moves)
        for q, target in best[1]:
            self._commit_move(q, target)
        return True
