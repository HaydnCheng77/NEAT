import math
import json
import os
import time
from tqdm import tqdm

from NEAT.placer import Placer
from NEAT import transport



class Router:
    """Per-stage placement (via ``Placer``) and AOD move planning; emits the instruction list."""

    # Shared with Placer rather than reimplemented — see
    # NEAT/transport.py. Bound as staticmethods so the existing
    # ``self.movement_duration(...)`` / ``self.compatible_2D(...)`` call sites
    # reach the module function with no forwarding frame; ``compatible_2D`` in
    # particular runs O(V^2) times per wave.
    movement_duration = staticmethod(transport.movement_duration)
    compatible_2D = staticmethod(transport.compatible_2D)

    def __init__(self,
                 slm_sites: list,
                 results_code: dict,
                 list_full_gates: list,
                 qubit_mapping: list,
                 architecture: dict,
                 routing_strategy: str='baseline',
                 write_code: bool=True
                 ):
        """
        Args:
            slm_sites: ``[storage_sites, entanglement_sites]`` as lists of ``(x, y)`` traps (µm).
            results_code: Shared dict with ``n_q``, ``stages``, ``benchmark``, ``output_dir``; extended with ``instructions``.
            list_full_gates: Scheduled gates per stage (pairs of qubit indices).
            qubit_mapping: Optional initial trap assignment per qubit index.
            architecture: Hardware timing/fidelity JSON plus optional ``routing`` overrides.
            routing_strategy: Stored for logging/CLI display only — NOT
                forwarded to ``Placer`` (which never accepted this parameter;
                see ``Placer.__init__``'s 2026-09-13 removal note). Historical
                values ``"always_move"``/``"always_stay"`` are inert here.
            write_code: Emit the instruction trace here. ``Zap`` passes False and
                writes the trace itself once routing returns.
        """
        self.architecture = architecture
        operation_duration = self.architecture.get("operation_duration", {})
        routing_cfg = self.architecture.get("routing", {})

        self.time_atom_transfer = operation_duration.get("atom_transfer", 15)
        self.time_2q_gate = operation_duration.get("2qGate", 0.25)
        self.time_1q_gate = operation_duration.get("1qGate", 0.5)

        self.PARKING_DIST = routing_cfg.get("parking_dist", 1)
        # d_s (Implication 3): distinct tones of one array cannot be closer than
        # this. ON by default at the reference hardware's 2 um since
        # 2026-09-30 : with it off, tqe traces carried ~750
        # tone gaps under 2 um (tools/verify_aod.py rule c). Costs about
        # 0.012 of tqe total F. 0.0 disables it. Also pushed into
        # transport.compatible_2D, which the placer's cost model shares.
        self._min_tone_sep = float(routing_cfg.get("min_tone_sep", 2.0) or 0.0)
        transport.set_min_tone_sep(self._min_tone_sep)
        if self._min_tone_sep and self.PARKING_DIST < self._min_tone_sep:
            # Every parking/waypoint offset is an integer multiple of
            # PARKING_DIST, so a step below d_s can only ever generate illegal
            # tone gaps. Raise it to the smallest legal step rather than
            # silently emitting them.
            # Kept integral: PARKING_DIST feeds range() bounds and integer
            # trap arithmetic elsewhere in the router.
            self.PARKING_DIST = math.ceil(self._min_tone_sep)
        # Rolling-chain tail (see _run_chain_legs). RAISED 2026-09-14: the old
        # default of 8 was tuned against tqe alone ("F rises 6.9200 (1 leg)
        # -> 6.9206 (4) -> 6.9229 (6) -> 6.9231 (8), flat to 12"), where tqe's
        # narrow stages mean the chain almost never gets deep enough to feel
        # a cap at all — confirmed by testing effectively unbounded
        # (100000) against tqe: bit-identical, 14/14, both physical gates
        # pass, i.e. the real stopping condition (see below) already governs
        # tqe entirely and the count never mattered there. On `graphs`
        # (wider stages, more atoms in flight per wave) the same unbounded
        # test measured +0.014874 total over 105 circuits (36 better, 3
        # regressed by <0.0004 each, 66 unchanged), both physical gates
        # still passing — the old cap of 8 really was truncating productive
        # chains on wider circuits.
        #
        # The loop already has a real, non-arbitrary stopping condition that
        # does not need an iteration count at all: each leg either finishes
        # a rider outright (_park_riders) or extends `partial` only via
        # _best_waypoint, which itself requires >= min_frac (2%) real
        # progress toward the target to accept a split — so every
        # leg that does anything shrinks `pending` by at least one
        # genuinely-progressed qubit, and the loop's own `if not partial and
        # not full: break` fires the instant a leg accomplishes nothing. The
        # loop is therefore already bounded by len(deferred) productive legs
        # with real per-leg progress; it cannot spin on infinitesimal gains.
        # `chain_max_legs` is kept as a purely defensive ceiling (should
        # never bind on any circuit tested; if it ever does, that is a sign
        # something upstream is producing a pathological chain, not evidence
        # the ceiling itself needs raising further) rather than a tuned
        # parameter — set high enough that no real circuit should approach
        # it, not picked to trade off against fidelity.
        self.chain_max_legs = int(routing_cfg.get("chain_max_legs", 1000))
        # Held-atom chaining (2026-10-03, experiment; see _held_leg_plan).
        # ZAP_HELD_CHAIN: "0" off; "1" always chain held atoms in legs;
        # "wave" also relax wave-level banking (Fig. 4b: bank a prefix even
        # when its remainder conflicts with another banked remainder);
        # "pick" route each stage as needed under off / leg / leg+wave and
        # keep the cheapest by the Simulator's own cost (_route_stage_pick).
        # Default "1" since 2026-10-03 : leg chaining only, at
        # the usual compile time; "pick" measured better but ~2x slower.
        self._hc_mode = os.environ.get("ZAP_HELD_CHAIN", "1")
        self._pick_stage = self._hc_mode == "pick"
        self._held_chain = self._hc_mode in ("1", "wave")
        self._wave_relax = self._hc_mode == "wave"
        self._remainder_rejected = False
        self._held_conflict = False
        # Ghost pick-up planning (2026-10-01, experimental). ghost_lookahead:
        # rank waypoints by the bystanders their lines would lift when the
        # chain's pending riders are picked up later (_best_waypoint).
        # ghost_refuse: a rider whose pick-up would still lift someone waits
        # for a later leg instead of joining this one.
        self.ghost_lookahead = bool(routing_cfg.get(
            "ghost_lookahead", os.environ.get("ZAP_GHOST_LOOKAHEAD", "0") == "1"))
        self.ghost_refuse = bool(routing_cfg.get(
            "ghost_refuse", os.environ.get("ZAP_GHOST_REFUSE", "0") == "1"))
        # ghost_leg_rounds: a chain leg lifts its full and partial riders as
        # one set, in clean rounds with a carrying slide between them
        # (_pick_up_rounds), instead of full riders then partial riders blind.
        self.ghost_leg_rounds = bool(routing_cfg.get(
            "ghost_leg_rounds", os.environ.get("ZAP_GHOST_LEG_ROUNDS", "1") == "1"))
        # ABLATION GATES (measurement only, both default OFF => shipped
        # behaviour is bit-identical). Set from a setting file's `routing_cfg`
        # to disable one transport mechanism while leaving every other stage of
        # the pipeline untouched, so an ablation measures that mechanism alone.
        #   ablate_waypoints: plan_waypoints returns {} => no deferred vector
        #     ever banks a partial prefix; it simply waits for a later wave.
        #   ablate_chain: the closing Park runs as a single straight leg per
        #     held atom (the pre-chain behaviour) instead of a rolling chain
        #     that hosts riders.
        # Note these two are COUPLED: _run_chain_legs itself calls
        # _best_waypoint to extend legs, so ablating waypoints also removes the
        # chain's ability to bank partial rider progress. Report the 2x2, not
        # just the two single ablations.
        # WAYPOINT DECOMPOSITION is ONE mechanism with two application sites:
        # the BigMove prefix (plan_waypoints) and the rolling Park chain
        # (_run_chain_legs), which extends its legs by calling _best_waypoint
        # -- the same banking rule applied to the Park phase. Ablating the
        # mechanism therefore disables both; `ablate_chain` remains available
        # on its own only for the historical 2x2 interaction measurement.
        self.ablate_waypoints = bool(routing_cfg.get("ablate_waypoints", False))
        self.ablate_chain = bool(routing_cfg.get("ablate_chain", False)) or self.ablate_waypoints
        # Cumulative-ablation gate: degree-greedy MIS over compatible_2D only,
        # dropping the dependency-ordered admission of
        # find_chain_aware_independent_set.
        self.ablate_chain_aware_mis = bool(
            routing_cfg.get("ablate_chain_aware_mis", False))
        # Router-block ablation gate: anchor collision diversions at the
        # ORIGIN, as baseline zap does, instead of at the destination.
        self.ablate_park_anchor = bool(routing_cfg.get("ablate_park_anchor", False))
        # When true, moves are emitted as AOD-compatible batches instead of one
        # simultaneous instruction.
        #
        # Now ON by default. Most of what this used to catch has been fixed at
        # the source (destination-anchored collision parking, per-column
        # pre-park fan-out, compatibility-screened diversion offsets), which
        # took the tqe suite from 111 emitted crossings to 64; the remainder are
        # mostly in the closing Park phase, which is assembled from
        # per-qubit fix-ups and never planned as a wave. Splitting those costs
        # 0.007 of total fidelity across 14 circuits and buys a trace with zero
        # AOD row/column crossings — worth it, because a plan that violates the
        # hardware's ordering constraint is not actually cheaper, it just
        # doesn't pay. Set false to reproduce baseline ZAP's unchecked emission
        # for a strictly like-for-like fidelity table (baseline itself emits 26).
        self.enforce_move_compatibility = bool(
            routing_cfg.get("enforce_move_compatibility", True))

        # Phase-0 instrumentation (wave-count reframing investigation). OFF by
        # default so normal runs pay nothing; set routing.collect_stats=true to
        # populate results_code['stats'] with per-circuit wave/transfer/crosstalk
        # counters. See PLACER_ALGORITHM_SUMMARY.md / ROUTER_ALGORITHM_SUMMARY.md
        # for the mechanism these numbers are meant to expose.
        self.collect_stats = bool(routing_cfg.get("collect_stats", False))
        if self.collect_stats:
            self.stats = {
                'waves_per_stage': [],   # one entry per stage: # of physical AOD waves emitted
                'total_waves': 0,
                'total_atom_transfers': 0,   # sum of len(qs) over Activate + Deactivate
                'total_crosstalk_exposures': 0,  # sum of len(qs) over Crosstalk instructions
                'move_time_us': 0.0,     # sum of BigMove + Park durations (bottleneck per instr)
                'transfer_time_us': 0.0,  # sum of Activate + Deactivate durations (bottleneck per instr)
                'gate1q_time_us': 0.0,
                'gate2q_time_us': 0.0,
            }
            self._stage_wave_count = 0

        # Phase-1 proxy audit (PLACER_ROUTER_COST_MODEL_INSTRUCTIONS.md §1).
        # OFF by default; set routing.proxy_audit=true to record, per stage, the
        # placer's predicted wave/duration/transfer cost against what this
        # router actually emitted, plus the mechanism counters the divergence
        # is categorised with. Costs a per-stage max-clique on the conflict
        # graph, so it is a measurement mode, not something to leave on.
        self.proxy_audit = bool(routing_cfg.get("proxy_audit", False))
        self.audit_stages = [] if self.proxy_audit else None
        self._audit = None

        self.slm_sites = slm_sites
        # Every physical trap. An atom sitting on one is caught by the SLM when
        # its AOD tone is ramped down; an atom anywhere else is not, which is
        # what makes partial deposits delicate (see process_rearrangement).
        self._slm_set = set(map(tuple, slm_sites[0])) | set(map(tuple, slm_sites[1]))
        # Exact entanglement-zone site membership, for write_idle's crosstalk
        # check. NOT a y-coordinate threshold (see write_idle) — on
        # baselines/zap/architectures/scale_to_500.json the storage zone's row count runs its
        # y-extent (0-186) past where the entanglement zone starts (124), so a
        # y >= edge proxy misclassifies storage-resident qubits as
        # entanglement-zone occupants. Confirmed: 110,288 of 111,415 crosstalk
        # charges on adder_n226 were against qubits actually sitting in a
        # storage trap. baselines/zap/architectures/default.json (the certified tqe suite)
        # has no such overlap (storage 0-42, entanglement 52-94), so this is a
        # no-op there — verified bit-identical on tqe after this fix.
        self._ent_site_set = set(map(tuple, slm_sites[1]))
        self.results_code = results_code
        self.n_q = results_code['n_q']
        self.list_full_gates = list_full_gates
        self.qubit_mapping = qubit_mapping

        self.routing_strategy = routing_strategy

        tmp = time.time()
        self.route_qubit()
        route_wall_s = time.time() - tmp
        self.results_code['compilation_time'] += route_wall_s
        if self.collect_stats:
            self.stats['route_wall_time_s'] = route_wall_s
            n_stages = len(self.stats['waves_per_stage'])
            self.stats['mean_waves_per_stage'] = (
                self.stats['total_waves'] / n_stages if n_stages else 0.0)
            self.results_code['stats'] = self.stats
        if self.proxy_audit:
            self.results_code['proxy_audit'] = self.audit_stages
        if write_code:
            self.log_results()

    def log_results(self):
        """
        Saves the results in JSON format within the specified directory.
        """
        output_dir = os.path.join("results", self.results_code['output_dir'], "code")
        output_path = os.path.join(
            output_dir, f"{self.results_code['benchmark']}_code.json"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.results_code, f)

    def write_init_instruction(self):
        """
        Writes initialization instructions for qubit mapping.
        """
        self.results_code['instructions'].clear()
        self.results_code['instructions'].append(
            {
                'type': "Init",
                'duration': [0 for _ in range(self.n_q)],
                'locs': [{
                    'id': q,
                    'x': self.current_mapping[q][0],
                    'y': self.current_mapping[q][1],
                } for q in range(self.n_q)]
            }
        )

    def write_1q_gate_instruction(self, gate_1q: list):
        """
        Writes instructions for single-qubit gates.

        Args:
            gate_1q (list): List of qubits for single-qubit gates.
        """
        qs = gate_1q
        locs = [
            {
                'id': q,
                'x': self.current_mapping[q][0],
                'y': self.current_mapping[q][1]
            } for q in qs
        ]
        self.results_code['instructions'].append({
            'type': "1qGate",
            'stage': self.index_list_gate,
            'duration': [self.time_1q_gate for _ in range(len(qs))],
            'qs': qs,
            'gates': gate_1q,
            'locs': locs
        })

    def write_2q_gate_instruction(self, gate_2q: list):
        """
        Writes instructions for two-qubit gates.

        Args:
            gate_2q (list): List of qubit pairs for two-qubit gates.
        """
        locs, qs = [], []
        for q0, q1 in gate_2q:
            qs += [q0, q1]
            locs.extend([{
                'id': q,
                'x': self.current_mapping[q][0],
                'y': self.current_mapping[q][1]
            } for q in [q0, q1]])

        self.results_code['instructions'].append({
            'type': "2qGate",
            'stage': self.index_list_gate,
            'duration': [self.time_2q_gate for _ in range(len(qs))],
            'qs': qs,
            'gates': gate_2q,
            'locs': locs
        })

    def write_activate_instruction(self, qs: list):
        """
        Writes instructions for activating qubits.
        """
        if not qs:
            return
        self._held.update(qs)
        self.results_code['instructions'].append({
            'type': "Activate",
            'qs': qs,
            'duration': [self.time_atom_transfer for _ in range(len(qs))],
            'locs': [{
                'id': q,
                'x': self.current_mapping[q][0],
                'y': self.current_mapping[q][1]
            } for q in qs]
        })
        if self.collect_stats and qs:
            self.stats['total_atom_transfers'] += len(qs)
            self.stats['transfer_time_us'] += self.time_atom_transfer

    def write_deactivate_instruction(self, qs: list):
        """
        Writes instructions for deactivating qubits.
        """
        # An atom may already have been released early (_release_blockers);
        # callers that release by their own bookkeeping skip it here.
        qs = [q for q in qs if q in self._held]
        if not qs:
            return
        self._held.difference_update(qs)
        self.results_code['instructions'].append({
            'type': "Deactivate",
            'qs': qs,
            'duration': [self.time_atom_transfer for _ in range(len(qs))],
            'locs': [{
                'id': q,
                'x': self.current_mapping[q][0],
                'y': self.current_mapping[q][1]
            } for q in qs]
        })
        if self.collect_stats and qs:
            self.stats['total_atom_transfers'] += len(qs)
            self.stats['transfer_time_us'] += self.time_atom_transfer

    def write_move_instruction(self, qs: list, type: str, end_locs: list):
        """
        Writes instructions for moving qubits.
        """
        if not qs:
            return
        # Held-atom guard (2026-09-30). An atom the AOD holds but does not move
        # in this instruction still has its row and column tones on, so every
        # mover must be compatible_2D with it as a zero-length vector -- the
        # check tools/verify_held.py applies after the fact. Counted, not yet
        # raised: 41 such crossings remain on tqe (mid-trip atoms that
        # write_move_batched splits into separate batches); switch this to a
        # hard error once those are fixed.
        movers = [(self.current_mapping[q][0], end_locs[q][0],
                   self.current_mapping[q][1], end_locs[q][1]) for q in qs
                  if tuple(self.current_mapping[q]) != tuple(end_locs[q])]
        if movers:
            moving = set(qs)
            for h in self._held:
                if h in moving:
                    continue
                hx, hy = self.current_mapping[h]
                if any(not self.compatible_2D(v, (hx, hx, hy, hy)) for v in movers):
                    self.results_code['held_crossings'] = (
                        self.results_code.get('held_crossings', 0) + 1)
                    if os.environ.get("ZAP_DEBUG_RAISE_HELD") == "1":
                        raise RuntimeError(f"held crossing: moving {qs} while q{h} held "
                                           f"at {self.current_mapping[h]}")
        distance_list = [math.dist(self.current_mapping[q], end_locs[q]) for q in qs]
        duration_list = [self.movement_duration(d) for d in distance_list]
        self.results_code['instructions'].append(
            {
                'type': type,
                'qs': qs,
                'distance': distance_list,
                'duration': duration_list,
                'locs': [{
                    'id': q,
                    'x_begin': self.current_mapping[q][0],
                    'y_begin': self.current_mapping[q][1],
                    'x_end': end_locs[q][0],
                    'y_end': end_locs[q][1],
                    'movement': distance_list[i],
                } for i, q in enumerate(qs)]
            }
        )
        if self._line_atoms is not None:
            by_col, by_row = self._line_atoms
            for q in qs:
                (x0, y0), (x1, y1) = self.current_mapping[q], end_locs[q]
                if x0 != x1:
                    by_col[x0].discard(q)
                    by_col.setdefault(x1, set()).add(q)
                if y0 != y1:
                    by_row[y0].discard(q)
                    by_row.setdefault(y1, set()).add(q)
        for q in qs:
            self.current_mapping[q] = end_locs[q]
        if self.collect_stats and qs:
            self.stats['total_waves'] += 1
            self._stage_wave_count += 1
            self.stats['move_time_us'] += max(duration_list) if duration_list else 0.0

    def write_move_batched(self, qs: list, type: str, end_locs: list,
                           allow_defer: bool = False, guard: bool = True):
        prev = getattr(self, '_guard_on', False)
        self._guard_on = guard
        try:
            return self._write_move_batched(qs, type, end_locs, allow_defer)
        finally:
            self._guard_on = prev

    def _write_move_batched(self, qs: list, type: str, end_locs: list,
                            allow_defer: bool = False):
        """Emit a move as one or more physically valid batches.

        ``write_move_instruction`` emits every qubit in a single simultaneous
        move. That is only physical if the moves are pairwise AOD-compatible
        (rows/columns may not cross). Two call sites can violate it: the main
        BigMove, whose vectors are verified when the wave is chosen but are
        then DIVERTED by collision parking; and the closing Park, which was
        never checked. Both produced real crossings.

        Splitting into batches must respect a second constraint: once the moves
        are sequential, a qubit may not land on a trap whose occupant has not
        moved yet. So each batch takes a maximal set that is both pairwise
        compatible AND has free targets. Compatible sets contain no permutation
        cycles (``compatible_2D`` rejects the crossings a cycle requires), so a
        valid order always exists and the loop terminates.

        Every batch must also clear the atoms the AOD holds but does not move
        in it (2026-09-30, see _legal_batch). When no batch can:
          * a held atom that has not moved yet and still sits on its trap is
            put back down and picked up again when its own batch comes
            (wstate_n27 inst 107: q16 on (36,31) blocked q18 leaving column 36
            while q18 blocked q16's target);
          * otherwise, with ``allow_defer``, the stuck moves are returned
            unmoved for the caller's next leg, where the atom blocking them
            may already be home and released (multiplier_n15 inst 830).
        Returns the deferred qubits (empty unless ``allow_defer``).
        """
        if not self.enforce_move_compatibility:
            # Default is True (see __init__); this branch only runs if a
            # config explicitly sets enforce_move_compatibility=false, to
            # reproduce baseline ZAP's unchecked emission (26 real crossings
            # across the tqe suite) for a strictly like-for-like fidelity
            # comparison (a measurement mode, never the default).
            self.write_move_instruction(qs, type, end_locs)
            return []

        movers = [q for q in qs
                  if tuple(self.current_mapping[q]) != tuple(end_locs[q])]
        if not movers:
            self.write_move_instruction(qs, type, end_locs)
            return []

        # Longest leg first. The split's cost is the SUM of the batches'
        # bottlenecks, so the one unavoidable long leg should absorb as many
        # short legs as it can rather than being stranded in its own batch
        # behind them. First-fit in caller order does the opposite whenever a
        # short 1 µm fix-up happens to be listed first and blocks the long leg
        # out of batch 1.
        remaining = sorted(
            movers,
            key=lambda q: -math.dist(self.current_mapping[q], end_locs[q]))
        guard = 0
        while remaining and guard <= 3 * len(movers) + 3:
            guard += 1
            # Home atoms still held whose tones this call's remaining moves
            # would cross are released first (2026-09-30).
            self._release_blockers(remaining, end_locs)
            self._release_corner_lines({q: tuple(end_locs[q]) for q in remaining})
            self._retarget_hazards(remaining, end_locs)
            batch = self._legal_batch(remaining, end_locs)
            if batch and os.environ.get("ZAP_RELIFT_CHECK", ("1" if os.environ.get("ZAP_STRICT", "1") == "1" else "0")) == "1":
                # Re-lifting put-back atoms used to be unchecked; atoms whose
                # re-lift would put a lit corner on a bystander wait for a
                # later batch when the batch has anything else to move.
                relift = [q for q, _v in batch if q not in self._held]
                if relift:
                    okr = set(self._clean_subset(relift))
                    if len(okr) < len(relift):
                        # Re-batch with the excluded atoms as stationary: a
                        # member may have been landing on a trap only one of
                        # them was vacating in this same move (ising_n1000:
                        # q684 onto q682's trap after q682 was dropped).
                        excl = set(relift) - okr
                        kept = self._legal_batch([r for r in remaining if r not in excl],
                                                 end_locs)
                        if kept:
                            batch = kept
            if batch:
                # Atoms put back down by the re-pick step below are lifted
                # again just before their own batch.
                relift = [q for q, _v in batch if q not in self._held]
                if relift:
                    if os.environ.get("ZAP_SAFE_LIFT", ("1" if os.environ.get("ZAP_STRICT", "1") == "1" else "0")) == "1" and (self._lift_ghosts(relift) or self._persist()):
                        if self._safe_lift(relift, future=[r for r in remaining if r not in self._held and r not in relift]):
                            continue   # positions changed: re-batch
                    else:
                        self.write_activate_instruction(relift)
                self.write_move_instruction([q for q, _v in batch], type, end_locs)
                done = {q for q, _v in batch}
                remaining = [r for r in remaining if r not in done]
                continue
            if self._put_back_unmoved(remaining, end_locs):
                continue
            if allow_defer:
                return remaining
            # No batch is legal against the held atoms: fall back to the old
            # selection, which checks only the movers against each other. The
            # held-atom guard in write_move_instruction counts what follows.
            # Traps still held by qubits that have not moved in this call.
            held = {tuple(self.current_mapping[r]): r for r in remaining}
            batch = []
            for q in remaining:
                sx, sy = self.current_mapping[q]
                tx, ty = end_locs[q]
                target = (tx, ty)
                # A stationary qubit on the target always blocks.
                if any(tuple(self.current_mapping[o]) == target
                       for o in range(self.n_q)
                       if o != q and o not in remaining):
                    continue
                v = (sx, tx, sy, ty)
                if not all(self.compatible_2D(v, u) for _bq, u in batch):
                    continue
                # A target still held by a PENDING mover is fine as long as that
                # mover leaves in this same simultaneous instruction — the AOD
                # applies one rigid row/column transform, so a chain A->B while
                # B->C never collides. That is precisely what compatible_2D
                # already certifies, and it is checked just above. Blocking it
                # unconditionally serialised every chain the placer had gone to
                # the trouble of making compatible (multiplier_n15 stage 154:
                # q5 (24,58)->(24,52) behind q6 (24,52)->(24,42), two waves
                # where one suffices). Only a mover left for a LATER batch is a
                # real hazard, so the batch is re-checked for that below.
                batch.append((q, v))
            # Drop any member whose target is vacated by a mover that did NOT
            # make it into this batch; it would land on a still-occupied trap.
            while True:
                in_batch = {q for q, _v in batch}
                keep = [(q, v) for q, v in batch
                        if (v[1], v[3]) not in held
                        or held[(v[1], v[3])] in in_batch
                        or held[(v[1], v[3])] == q]
                if len(keep) == len(batch):
                    break
                batch = keep
            if not batch:
                break  # cannot sequentialise safely; fall through below
            # Atoms put back on their traps by _put_back_unmoved are lifted
            # again before they move (2026-10-01; ising_n26 inst 10 moved q16
            # after it had been released).
            relift = [q for q, _v in batch if q not in self._held]
            if relift:
                self.write_activate_instruction(relift)
            self.write_move_instruction([q for q, _v in batch], type, end_locs)
            done = {q for q, _v in batch}
            remaining = [r for r in remaining if r not in done]

        if remaining:
            # Deadlocked (should not happen for compatible, cycle-free sets):
            # emit the rest together rather than dropping the move. This is the
            # one path in the router that writes a move WITHOUT a compatibility
            # check, so the audit counts it: if it ever fires, the trace can
            # contain a crossing that nothing upstream would have caught.
            if self.proxy_audit and self._audit is not None:
                self._audit['unchecked_emit'] += 1
            relift = [q for q in remaining if q not in self._held]
            if relift:
                self.write_activate_instruction(relift)
            self.write_move_instruction(remaining, type, end_locs)
        return []

    def _put_back_unmoved(self, remaining, end_locs):
        """Put down held atoms of ``remaining`` that still sit on a trap and
        block another remaining move; they are re-lifted for their own batch.
        Released by row or column, whichever carries no other held atom.
        Returns True if anything was put down."""
        vecs = {q: (self.current_mapping[q][0], end_locs[q][0],
                    self.current_mapping[q][1], end_locs[q][1]) for q in remaining}
        occupied = {tuple(self.current_mapping[q]) for q in remaining}
        cands = []
        for h in remaining:
            if h not in self._held:
                continue
            if tuple(self.current_mapping[h]) not in self._slm_set:
                continue
            st = self._stationary(h)
            if all(self.compatible_2D(v, st) for q, v in vecs.items() if q != h):
                continue
            x, y = self.current_mapping[h]
            others = [o for o in self._held if o != h]
            if (all(self.current_mapping[o][1] != y for o in others)
                    or all(self.current_mapping[o][0] != x for o in others)):
                # Prefer an atom that could not move now anyway (its target is
                # still occupied): putting it down costs it nothing it had.
                blocked = (vecs[h][1], vecs[h][3]) in occupied
                cands.append((not blocked, h))
        if not cands:
            return False
        # One at a time: each put-down costs two transfers, and freeing one
        # atom's tones is usually enough for the rest to find a legal order.
        self.write_deactivate_instruction([min(cands)[1]])
        return True

    def _finishable(self, held, targets):
        """Can every atom in ``held`` still be brought home legally?

        ``held`` maps each atom the AOD will hold to its position, ``targets``
        to its final trap. Simulates the simplest legal finish: release every
        home atom whose row or column carries no other held atom, then move
        one mid-air atom straight home, provided its move clears every other
        held atom as a stationary atom (its tones are still on) and its trap
        is free; repeat. True if the AOD empties. This is the planning-time
        form of the held-atom rule (2026-09-30): a waypoint, diversion or
        rider that would leave mid-air atoms blocking each other is refused
        when chosen, instead of being emitted as a crossing later.
        """
        from collections import Counter
        held = {q: tuple(p) for q, p in held.items()}
        nonheld = {tuple(p) for i, p in enumerate(self.current_mapping)
                   if i not in held}
        rows = Counter(p[1] for p in held.values())
        cols = Counter(p[0] for p in held.values())
        # A corner on an unheld atom with an unfinished atom on both of its
        # lines could only be cleared by dragging it (_corner_hazard).
        nh_cols = {p[0] for q, p in held.items() if p != targets[q]}
        nh_rows = {p[1] for q, p in held.items() if p != targets[q]}
        by_col = self._lines()[0]
        if any(self.current_mapping[a][1] in nh_rows
               for c in nh_cols for a in by_col.get(c, ()) if a not in held):
            return False

        def release():
            again = True
            while again:
                again = False
                for q in list(held):
                    p = held[q]
                    if p == targets[q] and (rows[p[1]] == 1 or cols[p[0]] == 1):
                        del held[q]
                        rows[p[1]] -= 1
                        cols[p[0]] -= 1
                        nonheld.add(p)
                        again = True

        for _ in range(2 * len(held) + 2):
            release()
            if not held:
                return True
            taken = set(held.values())
            moved = False
            for q, p in held.items():
                t = targets[q]
                if p == t or t in nonheld or t in taken:
                    continue
                v = (p[0], t[0], p[1], t[1])
                if all(self.compatible_2D(v, (o[0], o[0], o[1], o[1]))
                       for r, o in held.items() if r != q):
                    held[q] = t
                    rows[p[1]] -= 1
                    cols[p[0]] -= 1
                    rows[t[1]] += 1
                    cols[t[0]] += 1
                    moved = True
                    break
            if not moved:
                return False
        return False

    def _occupied_lines(self):
        """Rows and columns an unheld atom sits on (cached per instruction)."""
        key = len(self.results_code['instructions'])
        if getattr(self, '_occ_lines_key', None) != key:
            held = self._held
            rows, cols = set(), set()
            for q, (x, y) in enumerate(self.current_mapping):
                if q not in held:
                    rows.add(y)
                    cols.add(x)
            self._occ_lines = (rows, cols)
            self._occ_lines_key = key
        return self._occ_lines

    def _ghosts(self, new):
        """Atoms an Activate of ``new`` would lift (ghost pick-ups).

        With the atoms already held still held, every intersection of an
        active row and an active column is a tweezer. Returns the atoms outside
        ``held | new`` sitting on such an intersection -- the same rule as
        tools/verify_aod.py (d). 2026-10-01.
        """
        members = set(self._held) | set(new)
        cols = {self.current_mapping[q][0] for q in members}
        rows = {self.current_mapping[q][1] for q in members}
        return [q for q, (x, y) in enumerate(self.current_mapping)
                if q not in members and x in cols and y in rows]

    def _new_corners(self, xy, cols, rows, occ, members):
        """Bystanders that lighting an atom at ``xy`` would newly lift, given
        the lines already lit (``cols``/``rows``). ``occ`` maps positions to
        the atoms that could be lifted; ``members`` are atoms lifted on
        purpose. Corners that are already lit are not counted again: they
        are the same lift whether or not this atom joins.
        """
        x, y = xy
        out = set()
        if x not in cols:
            for r in rows | {y}:
                a = occ.get((x, r))
                if a is not None and a not in members:
                    out.add(a)
        if y not in rows:
            for c in cols | {x}:
                a = occ.get((c, y))
                if a is not None and a not in members:
                    out.add(a)
        return out

    def _sep_ok(self, xy, cols, rows):
        """Lifting an atom at ``xy`` keeps every lit tone >= d_s from every
        other (rule c): its column/row is either an already-lit line or at
        least d_s from all of them (2026-10-02)."""
        ds = self._min_tone_sep
        if not ds:
            return True
        x, y = xy
        if x not in cols and any(abs(x - c) < ds for c in cols):
            return False
        if y not in rows and any(abs(y - r) < ds for r in rows):
            return False
        return True

    def _clean_order(self, cands, pos, occ):
        """``cands`` sorted fewest-clashes-first, the order ``_clean_round``
        tries them in. Depends only on the candidates' own positions and
        ``occ``, so a caller trying many slides of OTHER atoms computes it
        once (2026-10-03, bit-identical)."""
        at = lambda q: pos.get(q, self.current_mapping[q])
        def clash(a, b):
            (xa, ya), (xb, yb) = at(a), at(b)
            return (xa, yb) in occ or (xb, ya) in occ
        deg = {a: sum(clash(a, b) for b in cands if b != a) for a in cands}
        return sorted(cands, key=lambda q: deg[q])

    def _clean_round(self, cands, pos, held=None, occ=None, order=None):
        """Greedy set of ``cands`` that can be lifted now adding no new ghost
        pick-up, given the held atoms at ``pos`` (overrides). Fewest conflicts
        first -- the usual greedy for a large independent set, so fewer rounds.

        Incremental on purpose (2026-10-01): an atom joins if it adds no NEW
        lifted corner. Testing the whole grid instead meant that once any held
        corner was dirty nothing was ever clean again, and the pick-up fell
        back to lifting everyone at once -- 56,185 of ising_n1000's 58,723
        ghosts came from that fallback.
        """
        held = self._held if held is None else held
        at = lambda q: pos.get(q, self.current_mapping[q])
        cols = {at(h)[0] for h in held}
        rows = {at(h)[1] for h in held}
        occ = self._occ_nonheld(held) if occ is None else occ
        cset = set(cands)
        if order is None:
            order = self._clean_order(cands, pos, occ)
        rnd, members = [], set(held)
        closure = os.environ.get("ZAP_ROUND_CLOSURE", "1") == "1"
        for q in order:
            if q in members:
                continue
            if not self._sep_ok(at(q), cols, rows):
                continue
            if not closure:
                members.add(q)
                if self._new_corners(at(q), cols, rows, occ, members):
                    members.discard(q)
                    continue
                rnd.append(q)
                cols.add(at(q)[0]); rows.add(at(q)[1])
                continue
            # Closure (2026-10-02): a corner that lands on another rider of
            # this leg is not a ghost if that rider joins this round too --
            # it is lifted in this leg anyway, and leg riders are pairwise
            # move-compatible. Grow the group until no corner lands on a
            # rider outside it; reject it if any lands on a bystander.
            tc, tr, mem = set(cols), set(rows), members | {q}
            queue, group, ok = [q], [], True
            while queue:
                a = queue.pop()
                if not self._sep_ok(at(a), tc, tr):
                    ok = False
                    break
                hits = self._new_corners(at(a), tc, tr, occ, mem)
                if any(h not in cset for h in hits):
                    ok = False
                    break
                group.append(a)
                tc.add(at(a)[0]); tr.add(at(a)[1])
                for h in hits:
                    mem.add(h); queue.append(h)
            if not ok:
                continue
            rnd.extend(group)
            cols, rows, members = tc, tr, mem
        return rnd

    def _occ_nonheld(self, held=None):
        held = self._held if held is None else held
        return {tuple(p): q for q, p in enumerate(self.current_mapping)
                if q not in held}

    def _damage_round(self, cands, held=None, over=None, with_cost=False):
        """Fallback when nobody can be lifted cleanly: the atom that adds the
        fewest lifts, plus every other atom that then adds none."""
        held = self._held if held is None else held
        over = {} if over is None else over
        at = lambda q: over.get(q, self.current_mapping[q])
        cols = {at(h)[0] for h in held}
        rows = {at(h)[1] for h in held}
        occ = self._occ_nonheld(held)
        cost = {q: len(self._new_corners(at(q), cols, rows, occ, set(held) | {q}))
                for q in cands}
        first = min(cands, key=lambda q: (not self._sep_ok(at(q), cols, rows), cost[q]))
        rnd = [first]
        cols.add(at(first)[0]); rows.add(at(first)[1])
        mem = set(held) | {first}
        for q in cands:
            if q == first:
                continue
            if not self._sep_ok(at(q), cols, rows):
                continue
            mem.add(q)
            if self._new_corners(at(q), cols, rows, occ, mem):
                mem.discard(q)
                continue
            rnd.append(q)
            cols.add(at(q)[0]); rows.add(at(q)[1])
        return (rnd, cost[first]) if with_cost else rnd

    def _held_slide(self, movable, remaining, leg_qs, ends, reach=3,
                    held=None, over=None):
        """One common slide of ``movable`` (held atoms that move this leg)
        that lets the most of ``remaining`` be lifted cleanly afterwards.

        A common vector keeps every moved tone's order and spacing, so the
        slide itself cannot cross, split or merge among the moved atoms. It is
        also checked against held atoms that stay (no shared line on the
        moved axis, order kept, >= d_s), against the riders still to be lifted
        (>= d_s or exactly their line), must not land on an atom, and must keep
        every leg move compatible with the ones it was compatible with.
        Returns the new positions, or None if no slide lifts more cleanly than
        standing still. Ties: fewer persisting lifted corners, shorter, more
        progress toward ``ends``.
        """
        ds = max(self._min_tone_sep, 1e-9)
        held = self._held if held is None else held
        over = {} if over is None else over
        cm = self.current_mapping
        class _P:
            def __getitem__(_s, q):
                return over.get(q, cm[q])
        pos = _P()
        mv = [h for h in movable]
        if not mv:
            return None
        mset = set(mv)
        stay = [h for h in held if h not in mset]
        stay_x = {pos[h][0] for h in stay}
        stay_y = {pos[h][1] for h in stay}
        sh_x = any(pos[h][0] in stay_x for h in mv)
        sh_y = any(pos[h][1] in stay_y for h in mv)
        occ = self._occ_nonheld(held)
        rx = {pos[r][0] for r in remaining}
        ry = {pos[r][1] for r in remaining}
        vec = lambda q, p: (p[0], ends[q][0], p[1], ends[q][1])
        others = [u for u in leg_qs if u not in mset and u in ends]
        # Slides move only held atoms, so the riders' clash order is the
        # same for every offset: compute it once (bit-identical).
        order = (self._clean_order(remaining, over, occ)
                 if not mset.intersection(remaining) else None)
        base_n = len(self._clean_round(remaining, over, held=held, occ=occ, order=order))
        # Leg compatibility per axis, once per call instead of per offset:
        # compatible_2D is the AND of two independent per-axis tests, and a
        # slide changes only the start of each axis, so a moved atom breaks
        # some leg move at (dx, dy) iff one of the moves it was compatible
        # with fails on x at dx or on y at dy (bit-identical, 2026-10-03).
        _ax = transport.axis_compatible
        bad_x, bad_y = {}, {}
        for h in mv:
            if h not in ends:
                continue
            ov = vec(h, pos[h])
            us = [vec(u, pos[u]) for u in others if u in ends]
            us = [vu for vu in us if self.compatible_2D(ov, vu)]
            px, py = pos[h]; ex, ey = ends[h]
            bad_x[h] = {d for d in range(-reach, reach + 1)
                        if any(not _ax(px + d, ex, vu[0], vu[1]) for vu in us)}
            bad_y[h] = {d for d in range(-reach, reach + 1)
                        if any(not _ax(py + d, ey, vu[2], vu[3]) for vu in us)}
        persist_first = os.environ.get("ZAP_PERSIST_FIRST", ("1" if os.environ.get("ZAP_STRICT", "1") == "1" else "0")) == "1"
        base_persist = self._persist(held, over, occ=occ) if persist_first else 0
        leg_all = [u for u in leg_qs if u in ends]
        max_leg = max([math.dist(pos[u], ends[u]) for u in leg_all] or [0.0])
        rest_max = max([math.dist(pos[u], ends[u]) for u in leg_all if u not in mset]
                       or [0.0])
        best = None
        st = self.results_code.setdefault('ghost_stats', {})
        why = {}
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                if (dx, dy) == (0, 0):
                    continue
                if (dx and sh_x) or (dy and sh_y):
                    why['split'] = why.get('split', 0) + 1
                    continue
                npos = {h: (pos[h][0] + dx, pos[h][1] + dy) for h in mv}
                bad = False
                reason = None
                for h in mv:
                    nx, ny = npos[h]
                    if (nx, ny) in occ:
                        reason = 'occupied'; break
                    if dx and any(abs(nx - o) < ds or (pos[h][0] < o) != (nx < o)
                                  for o in stay_x):
                        reason = 'stay_x'; break
                    if dy and any(abs(ny - o) < ds or (pos[h][1] < o) != (ny < o)
                                  for o in stay_y):
                        reason = 'stay_y'; break
                    if any(nx != o and abs(nx - o) < ds for o in rx) or any(
                            ny != o and abs(ny - o) < ds for o in ry):
                        reason = 'ds_riders'; break
                    if h in ends and (dx in bad_x[h] or dy in bad_y[h]):
                        reason = 'leg_compat'; break
                if reason:
                    why[reason] = why.get(reason, 0) + 1
                    continue
                if persist_first:
                    pers = self._persist(held, {**over, **npos}, occ=occ)
                    # Never leave more lit corners on atoms than before, and
                    # only slide if it removes some or lets more riders join.
                    # (Corner count first: it is cheap, and a slide that adds
                    # corners is rejected whatever n_next would be.)
                    if pers > base_persist:
                        why['no_gain'] = why.get('no_gain', 0) + 1
                        continue
                n_next = len(self._clean_round(remaining, {**over, **npos}, held=held,
                                               occ=occ, order=order))
                if persist_first:
                    if pers == base_persist and n_next <= base_n:
                        why['no_gain'] = why.get('no_gain', 0) + 1
                        continue
                elif n_next <= base_n:
                    why['no_gain'] = why.get('no_gain', 0) + 1
                    continue
                cols = {npos[h][0] for h in mv} | stay_x
                rows = {npos[h][1] for h in mv} | stay_y
                persist = sum(1 for c in cols for r in rows if (c, r) in occ)
                prog = sum(math.dist(pos[h], ends[h]) - math.dist(npos[h], ends[h])
                           for h in mv if h in ends)
                if os.environ.get("ZAP_SLIDE_TIME", "1") == "1":
                    # Wall time this slide really adds: its own duration plus
                    # the change in the leg Park that follows (bottlenecked by
                    # its longest move), so a slide along the longest mover's
                    # path is cheaper than an equally short one across it.
                    far0 = max_leg
                    far1 = max([math.dist(npos[h], ends[h]) for h in mv if h in ends]
                               + [rest_max])
                    added = (self.movement_duration(math.hypot(dx, dy))
                             + self.movement_duration(far1)
                             - self.movement_duration(far0))
                    key = ((pers, -n_next, round(added, 3), -prog) if persist_first
                           else (-n_next, persist, round(added, 3), -prog))
                else:
                    key = (-n_next, persist, abs(dx) + abs(dy), -prog)
                if best is None or key < best[0]:
                    # Tested last: only a slide that would be chosen pays for it.
                    if self._corner_hazard(npos, held=held, over=over) is not None:
                        why['corner'] = why.get('corner', 0) + 1
                        continue
                    best = (key, npos)
        if best is None and os.environ.get("ZAP_DEBUG_SLIDE"):
            dbg = os.environ["ZAP_DEBUG_SLIDE"]
            try:
                n = sum(1 for _ in open(dbg))
            except OSError:
                n = 0
            if n < 40:
                with open(dbg, "a") as f:
                    f.write(json.dumps({
                        "mv": {h: list(pos[h]) for h in mv},
                        "mv_end": {h: list(ends[h]) for h in mv},
                        "stay": {h: list(pos[h]) for h in stay},
                        "rem": {r: list(pos[r]) for r in remaining},
                        "rem_end": {r: list(ends[r]) for r in remaining if r in ends},
                        "why": why, "base_n": base_n}) + "\n")
        if best is None:
            for k, v in why.items():
                st['why_' + k] = st.get('why_' + k, 0) + v
            st['slide_none'] = st.get('slide_none', 0) + 1
            if not mv:
                st['why_nomov'] = st.get('why_nomov', 0) + 1
        return None if best is None else best[1]

    def _lift_ghosts(self, qs):
        """Bystanders that lifting ``qs`` now would newly put under a lit
        corner (incremental, given what is held). 2026-10-02."""
        held = self._held
        cols = {self.current_mapping[h][0] for h in held}
        rows = {self.current_mapping[h][1] for h in held}
        occ = self._occ_nonheld(held)
        mem = set(held) | set(qs)
        hit = set()
        for q in qs:
            xy = self.current_mapping[q]
            hit |= self._new_corners(xy, cols, rows, occ, mem)
            cols.add(xy[0]); rows.add(xy[1])
        return hit

    def _clean_subset(self, qs):
        """Greedy subset of ``qs`` that can be lifted now adding no lit
        corner on a bystander and keeping tones >= d_s apart."""
        held = self._held
        cols = {self.current_mapping[h][0] for h in held}
        rows = {self.current_mapping[h][1] for h in held}
        occ = self._occ_nonheld(held)
        mem = set(held) | set(qs)
        ok = []
        for q in qs:
            xy = self.current_mapping[q]
            if not self._sep_ok(xy, cols, rows) or self._new_corners(xy, cols, rows, occ, mem):
                continue
            ok.append(q)
            cols.add(xy[0]); rows.add(xy[1])
        return ok

    def _safe_lift(self, qs, reach=10, allow_home=False, future=(), skip_fail=False):
        """Lift ``qs`` with no ghost pick-up, whatever is held (2026-10-02).

        The guaranteed fallback for the rare lifts every other mechanism left
        dirty: lift the atoms one at a time, and before each one, if it (or a
        corner the last move left lit) would put a tweezer on a bystander,
        slide EVERY held atom by one common vector to lines that leave the
        new atom's row and column, and each other, clear of occupied traps.
        Moving all held atoms together cannot split, cross or merge tones;
        the slide must also keep tones >= d_s from the new atom's lines and
        not land on an atom. Returns True if it had to slide (callers that
        planned positions should re-plan). Lifts anyway only if no vector in
        range works.
        """
        slid = False
        self._safe_skipped = []
        self._release_corner_lines()
        ds = max(self._min_tone_sep, 1e-9)
        qs = list(qs)
        for i_x, x in enumerate(qs):
            if x in self._held:
                continue
            # Lines of atoms this context will still lift after x: a slid
            # line must stay >= d_s from them and not merge onto them, or the
            # later Activate breaks rule c (ising_n1000 inst 1418).
            fut = [f for f in list(qs[i_x + 1:]) + list(future)
                   if f not in self._held and f != x]
            fx = {self.current_mapping[f][0] for f in fut}
            fy = {self.current_mapping[f][1] for f in fut}
            def fut_ok(h, nx, ny, fx=fx, fy=fy):
                ox, oy = self.current_mapping[h]
                for c in fx:
                    if (nx != c and abs(nx - c) < ds) or (nx == c and ox != c):
                        return False
                for r in fy:
                    if (ny != r and abs(ny - r) < ds) or (ny == r and oy != r):
                        return False
                return True
            xy = tuple(self.current_mapping[x])
            _cols = {self.current_mapping[h][0] for h in self._held}
            _rows = {self.current_mapping[h][1] for h in self._held}
            if (not self._lift_ghosts([x]) and not self._persist()
                    and self._sep_ok(xy, _cols, _rows)):
                self.write_activate_instruction([x])
                continue
            held = list(self._held)
            if not held:
                self.write_activate_instruction([x])
                continue
            # Atoms already home stay put (a slide would knock a finished
            # atom off its trap after the wave counted it done -- adder_n298
            # q268). The rest move by one vector, checked against them.
            home = [h for h in held
                    if tuple(self.current_mapping[h]) == tuple(self.final_mapping[h])]
            if home and not allow_home:
                # Release finished atoms first where legal: the line ramped
                # down (row or column) must carry no other held atom away
                # from its own final trap.
                midair = [h for h in held if h not in home]
                mr = {self.current_mapping[h][1] for h in midair}
                mc = {self.current_mapping[h][0] for h in midair}
                rel = [h for h in home if self.current_mapping[h][1] not in mr
                       or self.current_mapping[h][0] not in mc]
                if rel:
                    self.write_deactivate_instruction(rel)
                    held = list(self._held)
                    home = [h for h in home if h not in rel]
                    if not self._lift_ghosts([x]) and not self._persist() and \
                            self._sep_ok(xy, {self.current_mapping[h][0] for h in held},
                                         {self.current_mapping[h][1] for h in held}):
                        self.write_activate_instruction([x])
                        continue
            if allow_home:
                # Inside a chain leg every held atom is re-checked against its
                # final trap each leg, so a home atom moved aside is brought
                # back; in the wave code it would be lost (adder q268).
                home = []
            mvh = [h for h in held if h not in home]
            if not mvh:
                st = self.results_code.setdefault('ghost_stats', {})
                st['safe_fail'] = st.get('safe_fail', 0) + 1
                self.write_activate_instruction([x])
                continue
            home_x = {self.current_mapping[h][0] for h in home}
            home_y = {self.current_mapping[h][1] for h in home}
            occ = self._occ_nonheld()
            occ_pos = set(occ)
            cur = {h: tuple(self.current_mapping[h]) for h in self._held}
            # No-deadlock check (2026-10-31, _can_empty): refuse a slide that
            # leaves the held atoms unable to all be released or flown home.
            nd_guard = self._can_empty(cur)
            best = None
            for r in range(1, reach + 1):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if max(abs(dx), abs(dy)) != r:
                            continue
                        npos = {h: (self.current_mapping[h][0] + dx,
                                    self.current_mapping[h][1] + dy) for h in mvh}
                        if any(p in occ_pos for p in npos.values()):
                            continue
                        if not all(fut_ok(h, *npos[h]) for h in mvh):
                            continue
                        if (dx and any(self.current_mapping[h][0] in home_x for h in mvh)) or \
                           (dy and any(self.current_mapping[h][1] in home_y for h in mvh)):
                            continue   # would split a tone shared with a home atom
                        if not all(self.compatible_2D(
                                (self.current_mapping[h][0], npos[h][0],
                                 self.current_mapping[h][1], npos[h][1]),
                                self._stationary(o)) for h in mvh for o in home):
                            continue
                        cols = {p[0] for p in npos.values()} | home_x
                        rows = {p[1] for p in npos.values()} | home_y
                        if any(c != xy[0] and abs(c - xy[0]) < ds for c in cols):
                            continue
                        if any(rw != xy[1] and abs(rw - xy[1]) < ds for rw in rows):
                            continue
                        if xy[0] in cols or xy[1] in rows:
                            continue
                        allc = cols | {xy[0]}
                        allr = rows | {xy[1]}
                        if any((c, rw) in occ_pos and occ[(c, rw)] != x
                               for c in allc for rw in allr):
                            continue
                        if nd_guard and not self._can_empty({**cur, **npos}):
                            continue
                        if self._corner_hazard(npos) is not None:
                            continue
                        best = npos
                        break
                    if best:
                        break
                if best:
                    break
            if best:
                st = self.results_code.setdefault('ghost_stats', {})
                st['safe_slides'] = st.get('safe_slides', 0) + 1
                self.write_move_instruction(sorted(best), "Park", best)
                slid = True
            elif self._shift_offending_tones(x, reach, allow_home, fut_ok):
                st = self.results_code.setdefault('ghost_stats', {})
                st['safe_tone_shifts'] = st.get('safe_tone_shifts', 0) + 1
                slid = True
            else:
                st = self.results_code.setdefault('ghost_stats', {})
                st['safe_fail'] = st.get('safe_fail', 0) + 1
                if skip_fail:
                    # Last tier (2026-10-03): no slide or tone shift can make
                    # this lift clean (ising_n550 q373: 13 atoms held mid-air,
                    # every vector blocked). Do not lift it now; the caller
                    # defers it to a later leg. Anything bound for its trap
                    # waits, since every landing is occupancy-checked.
                    self._safe_skipped.append(x)
                    st['safe_skipped'] = st.get('safe_skipped', 0) + 1
                    continue
            self.write_activate_instruction([x])
        return slid

    def _shift_offending_tones(self, x, reach, allow_home=False, fut_ok=None):
        """Second-stage fallback for _safe_lift (2026-10-02): move only the
        tones that put a tweezer on a bystander, one tone at a time. A row
        tone carries every held atom on that row and moves in y only; a
        column tone moves in x only -- so no tone is split. Each move must be
        compatible with every held atom that stays, must not land on an
        atom, and is kept only if it reduces the lifts; done when x lifts
        clean (qft_n100: held row 37 (q70, q77) x q86's column hit q89)."""
        pos = self.current_mapping
        xy = tuple(pos[x])
        def bad():
            cols = {pos[h][0] for h in self._held}
            rows = {pos[h][1] for h in self._held}
            return (len(self._lift_ghosts([x])) + self._persist()
                    + (0 if self._sep_ok(xy, cols, rows) else 1000))
        cur = bad()
        moved = False
        for _ in range(6):
            if cur == 0:
                return True
            occ_pos = set(self._occ_nonheld())
            held = list(self._held)
            hpos = {h: tuple(pos[h]) for h in held}
            # No-deadlock check (2026-10-31, _can_empty): refuse a slide that
            # leaves the held atoms unable to all be released or flown home.
            nd_guard = self._can_empty(hpos)
            best = None
            for axis in (1, 0):
                for line in sorted({pos[h][axis] for h in held}):
                    S = [h for h in held if pos[h][axis] == line]
                    if not allow_home and any(
                            tuple(pos[h]) == tuple(self.final_mapping[h]) for h in S):
                        continue   # never knock a finished atom off its trap
                    rest = [h for h in held if h not in S]
                    for k in range(1, reach + 1):
                        for d in (k, -k):
                            npos = {h: ((pos[h][0], pos[h][1] + d) if axis == 1
                                        else (pos[h][0] + d, pos[h][1])) for h in S}
                            if any(p in occ_pos for p in npos.values()):
                                continue
                            if fut_ok and not all(fut_ok(h, *npos[h]) for h in S):
                                continue
                            if not all(self.compatible_2D(
                                    (pos[h][0], npos[h][0], pos[h][1], npos[h][1]),
                                    self._stationary(o)) for h in S for o in rest):
                                continue
                            if nd_guard and not self._can_empty({**hpos, **npos}):
                                continue
                            if self._corner_hazard(npos) is not None:
                                continue
                            old = {h: tuple(pos[h]) for h in S}
                            for h in S:
                                pos[h] = npos[h]
                            score = bad()
                            for h in S:
                                pos[h] = old[h]
                            if score < cur and (best is None or (score, k) < best[0]):
                                best = ((score, k), npos)
                    if best and best[0][0] == 0:
                        break
                if best and best[0][0] == 0:
                    break
            if best is None:
                if os.environ.get("ZAP_DEBUG_TONE"):
                    with open(os.environ["ZAP_DEBUG_TONE"], "a") as _f:
                        _f.write(json.dumps({"x": x, "xy": list(xy), "cur": cur,
                            "ghosts": sorted(self._lift_ghosts([x])), "persist": self._persist(),
                            "held": {h: list(pos[h]) for h in self._held},
                            "final": {h: list(self.final_mapping[h]) for h in self._held},
                            "occ_near": [list(p) for p in self._occ_nonheld()
                                         if abs(p[0]-xy[0]) < 40 and abs(p[1]-xy[1]) < 40]}) + "\n")
                return moved and cur == 0
            self.write_move_instruction(sorted(best[1]), "Park", best[1])
            moved = True
            cur = best[0][0]
        return cur == 0

    def _persist(self, held=None, over=None, occ=None):
        """Lit corners (held row x held column) resting on an atom that is not
        held -- the tweezers that would lift it at the next Activate or drag
        it at the next move (2026-10-02). Most of K3DW's remaining ghosts were
        such corners left behind by a move and counted at the next lift."""
        held = self._held if held is None else held
        over = {} if over is None else over
        at = lambda q: over.get(q, self.current_mapping[q])
        cols = {at(h)[0] for h in held}
        rows = {at(h)[1] for h in held}
        occ = self._occ_nonheld(held) if occ is None else occ
        # Same count either way; iterate whichever side is smaller.
        if len(cols) * len(rows) < len(occ):
            return sum(1 for c in cols for r in rows if (c, r) in occ)
        return sum(1 for (x, y) in occ if x in cols and y in rows)

    def _lines(self):
        """Every atom by column and by row, kept up to date by
        write_move_instruction so a corner check costs only the lines it
        touches, not a scan of all atoms."""
        if self._line_atoms is None:
            by_col, by_row = {}, {}
            for q, (x, y) in enumerate(self.current_mapping):
                by_col.setdefault(x, set()).add(q)
                by_row.setdefault(y, set()).add(q)
            self._line_atoms = (by_col, by_row)
        return self._line_atoms

    def _corner_hazard(self, ends, held=None, over=None):
        """The atom whose move to ``ends`` ({q: (x, y)}) would drag an unheld
        atom, or None.

        A move drags if it carries a lit corner off an unheld atom, or leaves a
        new corner on one with an atom away from its final trap on both lines:
        neither line could then be switched off, so a later move would have to
        drag it. A corner with a line of arrived atoms only is cleared by
        releasing them (_release_corner_lines). 2026-11-04.
        """
        held = self._held if held is None else held
        over = {} if over is None else over
        cm, fm = self.current_mapping, self.final_mapping
        at = lambda q: tuple(over.get(q, cm[q]))
        by_col, by_row = self._lines()
        idle_rows = lambda c: (cm[a][1] for a in by_col.get(c, ())
                               if a not in held and a not in ends)
        idle_cols = lambda r: (cm[a][0] for a in by_row.get(r, ())
                               if a not in held and a not in ends)
        cols0 = {at(h)[0] for h in held}
        rows0 = {at(h)[1] for h in held}
        for q, e in ends.items():
            if q not in held:
                continue
            x, y = at(q)
            if e[0] != x and any(r in rows0 for r in idle_rows(x)):
                return q
            if e[1] != y and any(c in cols0 for c in idle_cols(y)):
                return q
        nh_cols, nh_rows = set(), set()
        for q in set(held) | set(ends):
            p = tuple(ends.get(q, at(q)))
            if p != tuple(fm[q]):
                nh_cols.add(p[0])
                nh_rows.add(p[1])
        for q, e in ends.items():
            x, y = at(q)
            if e[0] != x and e[0] in nh_cols and any(r in nh_rows for r in idle_rows(e[0])):
                return q
            if e[1] != y and e[1] in nh_rows and any(c in nh_cols for c in idle_cols(e[1])):
                return q
        return None

    def _retarget_hazards(self, remaining, end_locs):
        """Give an atom whose planned stop would drag a nearby stop instead.

        A wave's waypoints are planned against each other, but an atom dropped
        from the wave afterwards stays on its trap, where their lines can now
        cross (ising_n550: q171's row and q407's column on dropped q383).
        Making the atom wait splits the wave and can jam the AOD, so one of the
        unfinished atoms stops at the best stop that is legal against the rest
        of the move, starting with the one named; its row may be locked to a
        row-mate, leaving only its partner free to change (q407 there). Never
        for a move to the final mapping, which is not this call's to change.
        """
        if end_locs is self.final_mapping:
            return
        cm, fm = self.current_mapping, self.final_mapping
        ends = {q: tuple(end_locs[q]) for q in remaining}
        for _ in range(len(remaining)):
            bad = self._corner_hazard(ends)
            if bad is None:
                return
            finals = {q: tuple(fm[q]) for q in set(self._held) | set(ends)}
            after = {q: tuple(cm[q]) for q in self._held}
            after.update(ends)
            cands = [bad] + [q for q in ends if q != bad and ends[q] != finals[q]]
            for c in cands:
                if c not in self._held:
                    continue
                v = (cm[c][0], fm[c][0], cm[c][1], fm[c][1])
                batch = [(cm[q][0], e[0], cm[q][1], e[1]) for q, e in ends.items() if q != c]
                occ = {tuple(p) for i, p in enumerate(cm) if i not in ends} | {
                    e for q, e in ends.items() if q != c}

                def ok(w, c=c, ends=ends, after=after, finals=finals):
                    return (self._corner_hazard({**ends, c: w}) is None
                            and self._finishable({**after, c: w}, finals))

                w = self._best_waypoint(v, batch, occ, occ, [], 0.02, ok=ok)
                if w is not None:
                    ends[c] = tuple(w)
                    end_locs[c] = tuple(w)
                    return
            return

    def _release_corner_lines(self, ends=None):
        """Switch off lines of arrived atoms that keep a lit corner on an
        unheld atom (2026-11-04). With ``ends``, only corners those moves would
        carry off. Returns True if anything was released."""
        if not self._held:
            return False
        cm, fm = self.current_mapping, self.final_mapping
        line = {0: {}, 1: {}}
        for h in self._held:
            line[0].setdefault(cm[h][0], []).append(h)
            line[1].setdefault(cm[h][1], []).append(h)
        moved = None
        if ends is not None:
            moved = ({cm[q][0] for q, e in ends.items() if e[0] != cm[q][0]},
                     {cm[q][1] for q, e in ends.items() if e[1] != cm[q][1]})
        drop = set()
        by_col = self._lines()[0]
        for col in line[0]:
            for a in by_col.get(col, ()):
                x, y = cm[a]
                if a in self._held or y not in line[1]:
                    continue
                if moved is not None and x not in moved[0] and y not in moved[1]:
                    continue
                for axis, c in ((0, x), (1, y)):
                    if all(tuple(cm[h]) == tuple(fm[h]) for h in line[axis][c]):
                        drop.update(line[axis][c])
                        break
        if drop:
            self.write_deactivate_instruction(sorted(drop))
        return bool(drop)

    def _best_slide(self, last_rnd, remaining, leg_qs, ends, held=None, over=None):
        """Carrying slide for the pick-up: try sliding only the round just
        lifted (earlier rounds already sit off the trap grid and should stay
        there) and sliding every held atom that moves this leg; keep the one
        that lets more of ``remaining`` be lifted cleanly next (2026-10-02).
        Sliding everything by one vector pushed earlier rounds back ONTO the
        grid whenever it moved the newest round off it (ising_n300: q0 at
        (49,11) onto storage row 12)."""
        held = self._held if held is None else held
        over = {} if over is None else over
        leg_set = set(leg_qs)
        sets = []
        lr = [q for q in (last_rnd or []) if q in leg_set and q in held]
        if lr:
            sets.append(lr)
        allm = [h for h in held if h in leg_set]
        if allm and set(allm) != set(lr):
            sets.append(allm)
        best = None
        occ0 = self._occ_nonheld(held)
        hpos = {h: tuple(over.get(h, self.current_mapping[h])) for h in held}
        # No-deadlock check (2026-10-31, _can_empty): refuse a slide that
        # leaves the held atoms unable to all be released or flown home,
        # after the leg move that follows the pick-up (2026-11-01).
        leg = {q: tuple(ends[q]) for q in held if q in leg_set and q in ends}
        nd_guard = self._can_empty(hpos, leg)
        for mv in sets:
            npos = self._held_slide(mv, remaining, leg_qs, ends, held=held, over=over)
            if npos is None:
                continue
            if nd_guard and not self._can_empty({**hpos, **npos}, leg):
                continue
            n = len(self._clean_round(remaining, {**over, **npos}, held=held, occ=occ0))
            d = max(abs(npos[q][0] - (over.get(q, self.current_mapping[q]))[0])
                    + abs(npos[q][1] - (over.get(q, self.current_mapping[q]))[1])
                    for q in npos)
            key = (-n, d, len(npos))
            if os.environ.get("ZAP_PERSIST_FIRST", ("1" if os.environ.get("ZAP_STRICT", "1") == "1" else "0")) == "1":
                key = (self._persist(held, {**over, **npos}, occ=occ0),) + key
            if best is None or key < best[0]:
                best = (key, npos)
        return None if best is None else best[1]

    def _plan_pick_up(self, new, leg_qs, ends):
        """Dry run of _pick_up_rounds: the clean lifts and carrying slides it
        would emit, and the riders it could not lift cleanly (2026-10-02).
        Nothing is emitted; positions of held atoms evolve in ``over``."""
        held = set(self._held)
        over = {}
        leg_set = set(leg_qs)
        remaining = list(new)
        steps = []
        slides = 0
        last = []
        mode = os.environ.get("ZAP_SLIDE_SET", "best")
        cap = int(os.environ.get("ZAP_MAX_SLIDES", "1000000"))
        if os.environ.get("ZAP_PERSIST_FIRST", ("1" if os.environ.get("ZAP_STRICT", "1") == "1" else "0")) == "1" and remaining and self._persist(held, over):
            # Corners left on atoms by the previous move would be counted at
            # this leg's first lift: slide them off first if possible.
            npos = self._best_slide([], remaining, leg_qs, ends, held=held, over=over)
            if npos is not None:
                steps.append(('slide', npos))
                over.update(npos)
                slides += 1
        for _ in range(2 * len(new) + 2):
            if not remaining:
                break
            rnd = self._clean_round(remaining, over, held=held)
            if not rnd:
                if slides > len(new):
                    break
                if slides >= cap:
                    npos = None
                elif mode == "all":
                    npos = self._held_slide([h for h in held if h in leg_set],
                                            remaining, leg_qs, ends,
                                            held=held, over=over)
                else:
                    npos = self._best_slide(last, remaining, leg_qs, ends,
                                            held=held, over=over)
                if npos is None and os.environ.get("ZAP_FALLBACK", "all") == "all":
                    # Stuck: lift everyone left in ONE Activate. Fewer
                    # Activates than one damage round each, and the checker
                    # re-counts every persisting dirty corner per Activate.
                    steps.append(('damage', list(remaining), 1))
                    held |= set(remaining)
                    remaining = []
                    break
                if npos is None:
                    # Stuck: one least-damage round, then keep going (slides
                    # and clean rounds may work again with it held).
                    rnd, c = self._damage_round(remaining, held=held, over=over,
                                                with_cost=True)
                    steps.append(('damage', rnd, c))
                    held |= set(rnd)
                    last = list(rnd)
                    remaining = [q for q in remaining if q not in rnd]
                    continue
                slides += 1
                steps.append(('slide', npos))
                over.update(npos)
                last = []
                continue
            steps.append(('lift', rnd))
            held |= set(rnd)
            last = list(rnd)
            remaining = [q for q in remaining if q not in rnd]
        return steps, remaining

    def _pick_up_planned(self, new, leg_qs, ends):
        """Plan the leg's pick-up, defer the riders it cannot lift cleanly to
        a later leg, then emit the plan (2026-10-02). Returns the deferred.

        A rider can only be deferred if no other leg member is about to land
        on the trap it keeps occupying; a deferral that strands such a member
        defers that member too (if it is a rider) or is undone (if it is an
        atom already held). At least one rider always stays, so the chain
        never ends just because of this. Riders that cannot be deferred are
        lifted with the fewest extra lifts (_damage_round).
        """
        st = self.results_code.setdefault('ghost_stats', {})
        bump = lambda k, n=1: st.__setitem__(k, st.get(k, 0) + n)
        bump('legs'); bump('riders', len(new))
        self._release_corner_lines()
        new_set = set(new)
        drop = set()
        forced = set()
        thr = int(os.environ.get("ZAP_DEFER_MIN", "1"))
        for _ in range(len(new) + 2 if os.environ.get("ZAP_DEFER", "1") == "1" else 0):
            cands = [q for q in new if q not in drop]
            steps, residual = self._plan_pick_up(cands, leg_qs, ends)
            residual = [q for st_ in steps if st_[0] == 'damage' and st_[2] >= thr
                        for q in st_[1]] + list(residual)
            residual = [q for q in residual if q not in forced]
            if not residual:
                break
            add = set(residual)
            changed = True
            while changed:
                changed = False
                stay = {tuple(self.current_mapping[r]): r for r in drop | add}
                # Anyone about to land on a deferred rider's trap: this leg's
                # endpoint, or the FINAL trap a split atom is driven to by a
                # later leg of this chain (qaoa_n300: q163, mid-air bound for a
                # waypoint, later landed on deferred q13's trap).
                for m in list(leg_qs) + [h for h in self._held if h not in leg_qs]:
                    if m in drop or m in add:
                        continue
                    r = stay.get(tuple(ends[m]))
                    if r is None:
                        r = stay.get(tuple(self.final_mapping[m]))
                    if r is None:
                        continue
                    if m in new_set and m not in forced:
                        add.add(m); changed = True
                    elif m in self._held and os.environ.get("ZAP_DEFER_HELD", "0") == "1":
                        # A held atom bound for r's trap simply waits: every
                        # landing is occupancy-checked (write_move_batched
                        # batches; the set-down finish above), so it lands
                        # once r has left. 2026-10-02.
                        continue
                    else:
                        add.discard(r); forced.add(r); changed = True
                    break
            if len(drop | add) >= len(new) and os.environ.get("ZAP_ALLOW_EMPTY_LEG", "0") != "1":
                keep = next(q for q in new if q in add)
                add.discard(keep); forced.add(keep)
            if not add:
                break
            drop |= add
        cands = [q for q in new if q not in drop]
        safe = os.environ.get("ZAP_SAFE_LIFT", ("1" if os.environ.get("ZAP_STRICT", "1") == "1" else "0")) == "1"
        skipped_all = set()
        for _round in range(len(cands) + 2):
            todo = [q for q in cands if q not in self._held and q not in skipped_all]
            if not todo:
                break
            steps, residual = self._plan_pick_up(todo, leg_qs, ends)
            replan = False
            for stp in steps:
                kind, arg = stp[0], stp[1]
                if kind == 'damage':
                    bump('dirty_lifts', len(arg))
                    bump('dirty_forced', len([q for q in arg if q in forced]))
                if kind in ('lift', 'damage'):
                    bump('rounds' if kind == 'lift' else 'damage_rounds')
                    if safe and (self._lift_ghosts(arg) or self._persist()):
                        slid_ = self._safe_lift(arg, allow_home=True, skip_fail=True,
                                                future=[q for q in todo if q not in arg])
                        skipped_all |= set(self._safe_skipped)
                        if slid_ or self._safe_skipped:
                            replan = True
                            break
                    else:
                        self.write_activate_instruction(arg)
                else:
                    bump('slides')
                    self.write_move_batched(list(arg), "Park", arg, guard=False)
            if replan:
                continue
            rest = [q for q in residual if q not in self._held]
            if rest:
                if safe:
                    self._safe_lift(rest, allow_home=True, skip_fail=True)
                    skipped_all |= set(self._safe_skipped)
                else:
                    while rest:
                        rnd = self._damage_round(rest)
                        bump('damage_rounds'); bump('damage_riders', len(rnd))
                        self.write_activate_instruction(rnd)
                        rest = [q for q in rest if q not in rnd]
            break
        drop |= skipped_all
        bump('deferred', len(drop))
        return sorted(drop)

    def _pick_up_rounds(self, new, leg_qs, ends):
        """Lift ``new`` (all on their traps) with as few ghost pick-ups as the
        held lines allow (2026-10-01).

        Rounds: lift the largest set that adds no lifted corner; if nobody
        can be lifted cleanly, or more remain after a round, slide every held
        atom that moves this leg ONCE by a common vector to positions whose
        lines let the most remaining riders join cleanly (_held_slide); only
        when no slide helps, lift the least damaging atom and those that add
        nothing more (_damage_round). The old fallback lifted everyone at
        once, so every rider's column met every rider's row.
        """
        remaining = list(new)
        leg_set = set(leg_qs)
        slides = 0
        st = self.results_code.setdefault('ghost_stats', {})
        bump = lambda k, n=1: st.__setitem__(k, st.get(k, 0) + n)
        bump('legs'); bump('riders', len(new))
        for _ in range(2 * len(new) + 2):
            if not remaining:
                return
            rnd = self._clean_round(remaining, {})
            if not rnd and slides <= len(new):
                movable = [h for h in self._held if h in leg_set]
                npos = self._held_slide(movable, remaining, leg_qs, ends)
                if npos is not None:
                    slides += 1
                    bump('slide_unblock')
                    self.write_move_batched(list(npos), "Park", npos, guard=False)
                    continue
                bump('slide_unblock_fail')
            if not rnd:
                self._leg_round_fallbacks = getattr(self, '_leg_round_fallbacks', 0) + 1
                rnd = self._damage_round(remaining)
                bump('damage_rounds'); bump('damage_riders', len(rnd))
            bump('rounds')
            self.write_activate_instruction(rnd)
            remaining = [q for q in remaining if q not in rnd]
            if not remaining:
                return
            if (os.environ.get("ZAP_LAZY_SLIDE", "0") != "1"
                    and len(self._clean_round(remaining, {})) < len(remaining)
                    and slides <= len(new)):
                movable = [h for h in self._held if h in leg_set]
                npos = self._held_slide(movable, remaining, leg_qs, ends)
                if npos is not None:
                    slides += 1
                    bump('slide_after')
                    self.write_move_batched(list(npos), "Park", npos, guard=False)
                else:
                    bump('slide_after_fail')
        if remaining:
            self.write_activate_instruction(remaining)

    def _ghosts_at(self, members, pos):
        """Bystanders lit by holding ``members`` at positions ``pos`` (q -> xy;
        falls back to current_mapping). Same rule as _ghosts."""
        members = set(members)
        at = lambda q: pos.get(q, self.current_mapping[q])
        cols = {at(q)[0] for q in members}
        rows = {at(q)[1] for q in members}
        return [q for q, (x, y) in enumerate(self.current_mapping)
                if q not in members and x in cols and y in rows]

    def _clean_round_old(self, cands, pos):
        """Largest-first greedy set of ``cands`` that can be lifted together
        with no ghost, given what is held (positions overridden by ``pos``).

        Atoms with the fewest conflicts go first -- the usual greedy for a big
        independent set, so the pick-up needs fewer rounds (each extra round
        is an Activate and usually a carrying slide). Two atoms conflict when
        one's column meets the other's row on a bystander.
        """
        at = lambda q: pos.get(q, self.current_mapping[q])
        held = set(self._held)
        cset = set(cands)
        occ = {tuple(at(q)) for q in range(self.n_q)
               if q not in held and q not in cset}
        def clash(a, b):
            (xa, ya), (xb, yb) = at(a), at(b)
            return (xa, yb) in occ or (xb, ya) in occ
        deg = {a: sum(clash(a, b) for b in cands if b != a) for a in cands}
        order = sorted(cands, key=lambda q: deg[q])
        rnd = []
        for q in order:
            if not self._ghosts_at(held | set(rnd) | {q}, pos):
                rnd.append(q)
        return rnd

    def _pick_up_rounds_old(self, new, leg_qs, ends, reach=3):
        """Lift ``new`` (all on their traps) with no ghost pick-up (2026-10-01).

        Case B/C of the ghost audit: the chain used to lift a leg's full riders
        and then its partial riders blind; 49 of tqe's ghosts were corners
        between atoms picked up in the same leg. Same method as the pre-park:
        take the largest set that lifts nobody given what is held, then slide
        that round ONCE, by one common vector (order and spacing preserved),
        to a carrying position whose lines leave empty corners with everything
        lit at the next pick-up; repeat. A carrying slide must keep every leg
        move compatible with the ones it was compatible with, stay >= d_s from
        every other lit line, not split a tone it shares with a held atom, not
        land on an atom, and strictly reduce the lifts at the next pick-up --
        nearest first, then most progress toward ``ends``. When nobody can be
        lifted cleanly the rest go up together, as before (mid-air held lines
        are case (2), not handled here).
        """
        remaining = list(new)
        ds = max(self._min_tone_sep, 1e-9)
        for _ in range(len(new) + 1):
            if not remaining:
                return
            rnd = self._clean_round_old(remaining, {})
            if not rnd:
                self._leg_round_fallbacks = getattr(self, '_leg_round_fallbacks', 0) + 1
                self.write_activate_instruction(remaining)
                return
            self.write_activate_instruction(rnd)
            remaining = [q for q in remaining if q not in rnd]
            if not remaining:
                return
            # Carrying slide for this round.
            pos = self.current_mapping
            rset = set(rnd)
            others = [h for h in self._held if h not in rset]
            lit_x = {pos[h][0] for h in others} | {pos[q][0] for q in remaining}
            lit_y = {pos[h][1] for h in others} | {pos[q][1] for q in remaining}
            shared_x = any(pos[q][0] in lit_x for q in rnd)
            shared_y = any(pos[q][1] in lit_y for q in rnd)
            occ = {tuple(p) for i, p in enumerate(pos) if i not in rset}
            leg_other = [u for u in leg_qs if u not in rset]
            vec = lambda q, p: (p[0], ends[q][0], p[1], ends[q][1])
            # A slide is only worth its stop if more atoms can then join the
            # next round cleanly than without it.
            base_n = len(self._clean_round_old(remaining, {}))
            if base_n == len(remaining):
                continue
            best = None
            for dx in range(-reach, reach + 1):
                for dy in range(-reach, reach + 1):
                    if (dx, dy) == (0, 0):
                        continue
                    if (dx and shared_x) or (dy and shared_y):
                        continue
                    npos = {q: (pos[q][0] + dx, pos[q][1] + dy) for q in rnd}
                    bad = False
                    for q in rnd:
                        nx, ny = npos[q]
                        if (nx, ny) in occ:
                            bad = True; break
                        if dx and any(abs(nx - o) < ds or (pos[q][0] < o) != (nx < o)
                                      for o in lit_x):
                            bad = True; break
                        if dy and any(abs(ny - o) < ds or (pos[q][1] < o) != (ny < o)
                                      for o in lit_y):
                            bad = True; break
                        slide = (pos[q][0], nx, pos[q][1], ny)
                        if not all(self.compatible_2D(slide, self._stationary(h))
                                   for h in others):
                            bad = True; break
                        old_v, new_v = vec(q, pos[q]), vec(q, npos[q])
                        for u in leg_other:
                            vu = vec(u, pos[u])
                            if (self.compatible_2D(old_v, vu)
                                    and not self.compatible_2D(new_v, vu)):
                                bad = True; break
                        if bad:
                            break
                    if bad:
                        continue
                    n_next = len(self._clean_round_old(remaining, npos))
                    if n_next <= base_n:
                        continue
                    prog = sum(math.dist(pos[q], ends[q]) - math.dist(npos[q], ends[q])
                               for q in rnd)
                    key = (-n_next, abs(dx) + abs(dy), -prog)
                    if best is None or key < best[0]:
                        best = (key, npos)
            if best is not None:
                self.write_move_batched(rnd, "Park", best[1], guard=False)
        return

    def _prepark_row_shift(self, picked, wave, col_shift, end_locs, reach=4):
        """Row offset for this pre-park round's carrying position (2026-10-01).

        The fan-out slides a picked column off its traps but leaves its atoms
        on their trap ROW, and that row stays lit while the rest of the wave is
        picked up: any later atom whose column meets it on a bystander lifts
        that bystander (ising_n26 inst 4: q20's column 51 x held row 15 = q15).
        Lookahead: count, for each candidate offset dy, the corners a picked
        atom at its carrying position (x + col shift, y + dy) would form with
        every wave atom still to be picked up, and take the offset with the
        fewest bystanders on them, nearest first; 0 unless it strictly helps.
        Same slide as the column shift, so every atom still moves once.

        A shifted row must stay >= d_s from every other row lit in this wave,
        keep its order relative to them, and not land on an atom. It must also
        keep each picked atom's main move compatible with every wave-mate it
        was compatible with: moving q7 off row 15 while q16/q17 stay on it,
        all three bound for row 31, merges two row tones, and the BigMove fell
        apart into sequential Parks (ising_n26 stage 1, -0.0149).
        """
        picked = set(picked)
        if not picked:
            return 0
        pos = self.current_mapping
        members = set(wave) | set(self._held)
        later = [b for b in wave if b not in self._held and b not in picked]
        if not later:
            return 0
        occ = {tuple(p): q for q, p in enumerate(pos)}
        bystander = {p for p, q in occ.items() if q not in members}
        other_rows = {pos[q][1] for q in members if q not in picked}
        ds = max(self._min_tone_sep, 1e-9)

        def cost(dy):
            n = 0
            for a in picked:
                xa = pos[a][0] + col_shift.get(pos[a][0], 0)
                ya = pos[a][1] + dy
                for b in later:
                    xb, yb = pos[b]
                    n += ((xa, yb) in bystander) + ((xb, ya) in bystander)
            return n

        def legal(dy):
            for a in picked:
                y, ny = pos[a][1], pos[a][1] + dy
                nx = pos[a][0] + col_shift.get(pos[a][0], 0)
                if (nx, ny) in occ and occ[(nx, ny)] not in picked:
                    return False
                for r in other_rows:
                    if abs(ny - r) < ds:
                        return False
                    if r != y and (y < r) != (ny < r):
                        return False
                ex, ey = end_locs[a]
                old = (pos[a][0], ex, y, ey)
                new = (nx, ex, ny, ey)
                for b in wave:
                    if b == a or b in picked:
                        continue
                    vb = (pos[b][0], end_locs[b][0], pos[b][1], end_locs[b][1])
                    if (self.compatible_2D(old, vb)
                            and not self.compatible_2D(new, vb)):
                        return False
            return True

        best, best_cost = 0, cost(0)
        if best_cost == 0:
            return 0
        for k in range(1, reach + 1):
            for dy in (k * self.PARKING_DIST, -k * self.PARKING_DIST):
                if not legal(dy):
                    continue
                c = cost(dy)
                if c < best_cost:
                    best, best_cost = dy, c
            if best_cost == 0:
                break
        return best

    def _stationary(self, q):
        x, y = self.current_mapping[q]
        return (x, x, y, y)

    def _can_empty(self, pos, leg=None):
        """Can the AOD be emptied from held positions ``pos`` ({q: (x, y)})?

        The no-deadlock check (2026-10-31). Simulates two steps that are always
        legal when they apply: switch off a line whose held atoms all sit on
        traps (they are released there), or fly one atom that is between traps
        straight to its free final trap past the other held atoms held still.
        True if the AOD empties. Every step that moves held atoms after their
        wave was planned (the batch split, _safe_lift's slide, the tone shifts,
        the pick-up slide choice) keeps this true, so held atoms can always be
        released or flown home and never block each other for good. Measured:
        set-downs 197 -> 0 on six families at n = 300/500, 8 -> 0 on the 30
        Table III/IV circuits, 3 -> 0 on tqe; all traces legal.

        ``leg`` ({q: end}): first let these held atoms make their planned leg
        move if it is legal from ``pos`` (2026-11-01). Used by _best_slide only:
        during a pick-up the leg move comes next, and without it the check
        refused nearly every carrying slide on wide stages (row-mates heading
        to different rows cannot fly one at a time), deferring riders.
        Measured vs no leg step: random_n600 -log10 F 56.441 -> 50.990, tqe
        6.712855 -> 6.714318 (ising_n26 back to its floor), still 0 set-downs
        on tqe and the n = 300/500 set.
        """
        pos = {q: tuple(p) for q, p in pos.items()}
        occ = {tuple(p) for i, p in enumerate(self.current_mapping) if i not in pos}
        slm = self._slm_set
        if leg:
            vec = {q: (pos[q][0], e[0], pos[q][1], e[1]) for q, e in leg.items() if q in pos}
            vs = list(vec.values())
            if (all(self.compatible_2D(vs[i], vs[j]) for i in range(len(vs)) for j in range(i + 1, len(vs)))
                    and all(self.compatible_2D(v, (o[0], o[0], o[1], o[1]))
                            for v in vs for r, o in pos.items() if r not in vec)):
                for q, v in vec.items():
                    pos[q] = (v[1], v[3])
        while pos:
            on = {q for q, p in pos.items() if p in slm}
            rel = None
            for q in on:
                x, y = pos[q]
                for line in ([r for r, p in pos.items() if p[1] == y],
                             [r for r, p in pos.items() if p[0] == x]):
                    if all(r in on for r in line):
                        rel = line
                        break
                if rel:
                    break
            if rel:
                for r in rel:
                    occ.add(pos.pop(r))
                continue
            taken = occ | set(pos.values())
            for q, p in pos.items():
                if p in slm:
                    continue
                t = tuple(self.final_mapping[q])
                if t in taken:
                    continue
                v = (p[0], t[0], p[1], t[1])
                if all(self.compatible_2D(v, (o[0], o[0], o[1], o[1]))
                       for r, o in pos.items() if r != q):
                    pos[q] = t
                    break
            else:
                return False
        return True

    def _held_after(self, chosen, vec):
        """Held positions after moving ``chosen`` (their atoms are lifted)."""
        pos = {h: tuple(self.current_mapping[h]) for h in self._held}
        for q in chosen:
            pos[q] = (vec[q][1], vec[q][3])
        return pos

    def _legal_batch(self, remaining, end_locs):
        """Largest greedy batch of ``remaining`` that is legal as one move.

        Legal means: members pairwise compatible_2D; every member compatible
        with every atom the AOD holds but does not move in this batch (its
        tones are still on -- a stationary held atom is a zero-length vector);
        no member lands on an atom that stays put. Greedy in ``remaining``
        order, then pruned to a fixpoint, since removing a member makes it a
        stationary held atom the others must now clear.
        """
        rem = set(remaining)
        vec = {q: (self.current_mapping[q][0], end_locs[q][0],
                   self.current_mapping[q][1], end_locs[q][1]) for q in remaining}
        # Two greedy passes, keep the larger legal result.
        #   strict: a candidate must clear every held atom not yet in the batch
        #     as a stationary atom, so nothing admitted is ever pruned and a
        #     short move blocked only by a long one still gets in (qnn_n15
        #     stage 55: {q6, q3} rather than {q6}).
        #   prune: admit on mover-vs-mover compatibility, then prune against
        #     the atoms left out. Needed for atoms that share a tone and can
        #     only move together (each blocks the other while stationary).
        # No-deadlock check (_can_empty), unless the AOD already could not be
        # emptied; off for pick-up slides (write_move_batched(guard=False)).
        guard = getattr(self, '_guard_on', False) and self._can_empty(self._held_after((), vec))
        hazard = lambda qs: self._corner_hazard(
            {q: (vec[q][1], vec[q][3]) for q in qs})
        waiting = set()
        while True:
            strict = []
            for q in remaining:
                if q in waiting:
                    continue
                v = vec[q]
                if not all(self.compatible_2D(v, vec[b]) for b in strict):
                    continue
                chosen = set(strict) | {q}
                if not all(self.compatible_2D(v, self._stationary(h))
                           for h in self._held if h not in chosen):
                    continue
                pos = {tuple(self.current_mapping[o]) for o in range(self.n_q)
                       if o not in chosen}
                if (v[1], v[3]) in pos:
                    continue
                if guard and not self._can_empty(self._held_after(chosen, vec)):
                    continue
                strict.append(q)
            bad = hazard(strict) if strict else None
            if bad is None:
                break
            waiting.add(bad)
        pruned = self._pruned_batch(remaining, vec, rem)
        if guard and pruned and not self._can_empty(self._held_after(pruned, vec)):
            pruned = []
        # One atom that would drag waits; the rest of the batch still flies.
        cand = list(remaining)
        while pruned:
            bad = hazard(pruned)
            if bad is None:
                break
            cand.remove(bad)
            pruned = self._pruned_batch(cand, vec, set(cand))
        best = strict if len(strict) >= len(pruned) else pruned
        return [(q, vec[q]) for q in best]

    def _pruned_batch(self, remaining, vec, rem):
        batch = []
        for q in remaining:
            v = vec[q]
            if not all(self.compatible_2D(v, vec[b]) for b in batch):
                continue
            if not all(self.compatible_2D(v, self._stationary(h))
                       for h in self._held if h != q and h not in rem):
                continue
            batch.append(q)
        while batch:
            members = set(batch)
            pos = {tuple(self.current_mapping[o]): o for o in range(self.n_q)
                   if o not in members}
            keep = []
            for q in batch:
                v = vec[q]
                if (v[1], v[3]) in pos:
                    continue
                if not all(self.compatible_2D(v, self._stationary(h))
                           for h in self._held if h not in members):
                    continue
                keep.append(q)
            if len(keep) == len(batch):
                break
            batch = keep
        return batch

    def _release_blockers(self, remaining, end_locs):
        """Release held atoms that are home and would block ``remaining``.

        An atom is home when it sits on its final trap. It can be released
        by ramping down its row OR its column, whichever carries no other held
        atom (the other tone keeps holding the rest).
        """
        rem = set(remaining)
        vecs = [(self.current_mapping[q][0], end_locs[q][0],
                 self.current_mapping[q][1], end_locs[q][1]) for q in remaining]
        cand = [h for h in self._held if h not in rem
                and tuple(self.current_mapping[h]) == tuple(self.final_mapping[h])
                and tuple(self.current_mapping[h]) in self._slm_set
                and any(not self.compatible_2D(v, self._stationary(h)) for v in vecs)]
        if not cand:
            return
        drop = []
        for h in cand:
            x, y = self.current_mapping[h]
            others = [o for o in self._held if o != h]
            if (all(self.current_mapping[o][1] != y for o in others)
                    or all(self.current_mapping[o][0] != x for o in others)):
                drop.append(h)
        if drop:
            self.write_deactivate_instruction(drop)

    def process_gate(self):
        """
        Processes the gates in the current stage.
        """
        gate_1q = [q0 for q0, q1 in self.list_gate if q0 == q1]
        gate_2q = [(q0, q1) for q0, q1 in self.list_gate if q0 != q1]
        if gate_1q:
            self.write_1q_gate_instruction(gate_1q)
            if self.collect_stats:
                self.stats['gate1q_time_us'] += self.time_1q_gate
        if gate_2q:
            self.write_2q_gate_instruction(gate_2q)
            if self.collect_stats:
                self.stats['gate2q_time_us'] += self.time_2q_gate
            self.write_idle(gate_2q)

    def write_idle(self, gate_2q: list):
        """
        Writes idle instructions for qubits that are not involved in the current stage.

        Args:
            gate_2q (list): List of qubit pairs for two-qubit gates.
        """
        idle_qubits_for_2q_gates = list(set(range(self.n_q)) - set([q for q0, q1 in gate_2q for q in [q0, q1]]))
        idle_qubits = []
        if idle_qubits_for_2q_gates:
            for q in idle_qubits_for_2q_gates:
                if self.current_mapping[q] in self._ent_site_set:
                    idle_qubits.append(q)
        if idle_qubits:
            self.results_code['instructions'].append(
                {
                    'type': "Crosstalk",
                    'qs': idle_qubits,
                    'duration': [self.time_2q_gate for _ in range(len(idle_qubits))],
                    'locs': [{
                        'id': q,
                        'x': self.current_mapping[q][0],
                        'y': self.current_mapping[q][1]
                    } for q in idle_qubits]
                }
            )
            if self.collect_stats:
                self.stats['total_crosstalk_exposures'] += len(idle_qubits)

    def _feasible_axis_range(self, s_coord, batch, axis):
        """Exact feasible interval for a waypoint coordinate on one axis.

        ``compatible_2D`` is an ORDERING constraint, not a geometric one: an
        AOD row/column may not cross another, so for a move starting at
        ``s_coord`` the waypoint coordinate w must satisfy, for every committed
        move u in the batch:

            s_coord < u.start  ->  w < u.end
            s_coord > u.start  ->  w > u.end
            s_coord == u.start ->  w == u.end

        Those are half-planes, so their intersection is a single interval
        (or a fixed point). Computing it directly finds a legal waypoint
        whenever one exists, instead of sampling k points along the path and
        hoping one lands inside — which is why the old search reported
        "no waypoint found" on 100% of its failures.

        Returns ``(lo, hi, exact)`` with lo/hi EXCLUSIVE bounds, or exact
        pinned value (then lo/hi are ignored).
        """
        si, ei = (0, 1) if axis == 0 else (2, 3)
        lo, hi, exact = -math.inf, math.inf, None
        for u in batch:
            if s_coord == u[si]:
                exact = u[ei]
            elif s_coord < u[si]:
                hi = min(hi, u[ei])
            else:
                lo = max(lo, u[ei])
        return lo, hi, exact

    def _path_clear(self, start, end, occupied):
        """Same traversal as ``verify_path``, but against a supplied occupancy.

        ``verify_path`` reads ``self.current_mapping``, i.e. where atoms are
        BEFORE the wave. That is the right test for the wave's own BigMove, but
        the wrong one for the Park leg that follows it: by then every mover has
        landed on its new trap, and a leg that was clear at planning time can be
        crossing an atom that has just arrived. Both phases therefore need the
        same walk against different occupancy sets.
        """
        x1, y1 = start
        x2, y2 = end
        if (x1, y1) == (x2, y2):
            return True
        dx, dy = x2 - x1, y2 - y1
        gcd = abs(dx) if dy == 0 else abs(dy) if dx == 0 else abs(math.gcd(dx, dy))
        step_x, step_y = dx // gcd, dy // gcd
        x, y = x1 + step_x, y1 + step_y
        while (x, y) != (x2, y2):
            if (x, y) in occupied:
                return False
            x += step_x
            y += step_y
        return True

    def _best_waypoint(self, v, batch, occupied, blocked, remainders, min_frac,
                       pre_occ=None, post_occ=None, ok=None, future=None):
        """The legal waypoint that banks the most progress toward ``v``'s target.

        What a split actually buys is a SHORTER SEQUENTIAL REMAINDER: the prefix
        rides the wave's parallel BigMove (free if the wave's bottleneck already
        covers it), while the leftover ``w -> target`` leg runs in the sequential
        Park phase. So the quantity to maximise is
        ``gain = |start-target| - |w-target|``, and nothing else.

        Two consequences, both of which the previous version got wrong.

        First, the waypoint should NOT be searched along the straight line to
        the target. It is an AOD-held position, not an SLM trap, so every
        integer point in the feasible ordering box is admissible. The old code
        scanned on-path fractions first and returned the farthest that fit,
        falling back to an off-path box search only when every on-path point
        failed. When the batch pins an axis — e.g. another mover starts on the
        same row, which forces the waypoint onto that row exactly — the on-path
        scan succeeds with a token fraction and the far better off-path point is
        never considered. Measured on multiplier_n15: q4's (24,52) -> (12,58)
        trip banked 1.0 of 13.4 µm on-path, where the box allowed (13,52) and a
        7.3 µm bank.

        Second, ``_feasible_axis_range`` already encodes ``compatible_2D``
        against the batch exactly (its three cases ARE that predicate's ordering
        rules), so the feasible set is a plain axis-aligned integer box. The
        point of a box closest to the target is the target CLAMPED into it — no
        search at all in the common case. The neighbourhood walk below only
        exists to step around occupancy and the remainder constraint, which the
        box does not model.
        """
        sx, tx, sy, ty = v
        full = math.dist((sx, sy), (tx, ty))
        if full <= 0:
            return None

        lox, hix, ex = self._feasible_axis_range(sx, batch, 0)
        loy, hiy, ey = self._feasible_axis_range(sy, batch, 1)

        def axis_bounds(lo, hi, exact):
            """Inclusive integer bounds of one feasible interval (None = open)."""
            if exact is not None:
                return exact, exact
            lo_i = int(math.floor(lo)) + 1 if lo != -math.inf else None
            hi_i = int(math.ceil(hi)) - 1 if hi != math.inf else None
            return lo_i, hi_i

        xlo, xhi = axis_bounds(lox, hix, ex)
        ylo, yhi = axis_bounds(loy, hiy, ey)
        if ((xlo is not None and xhi is not None and xhi < xlo)
                or (ylo is not None and yhi is not None and yhi < ylo)):
            return None

        def clamp(c, lo, hi):
            if lo is not None and c < lo:
                return lo
            if hi is not None and c > hi:
                return hi
            return c

        def in_box(c, lo, hi):
            return (lo is None or c >= lo) and (hi is None or c <= hi)

        cx = clamp(round(tx), xlo, xhi)
        cy = clamp(round(ty), ylo, yhi)

        # Candidates: the clamp point and a small neighbourhood of it, ranked by
        # the progress each one banks. The clamp point is optimal, so anything
        # further out is only tried when it is vetoed.
        span = 4
        cands = []
        for dx in range(-span, span + 1):
            wx = cx + dx
            if not in_box(wx, xlo, xhi):
                continue
            for dy in range(-span, span + 1):
                wy = cy + dy
                if not in_box(wy, ylo, yhi):
                    continue
                w = (wx, wy)
                if w == (sx, sy) or w == (tx, ty):
                    continue
                gain = full - math.dist(w, (tx, ty))
                # A waypoint that does not shorten the sequential remainder by a
                # meaningful margin is pure cost: it lengthens the wave and adds
                # a Park leg for nothing. (The old staged branch had no gain test
                # at all and emitted 1 µm hops AWAY from the target.)
                if gain < min_frac * full or gain <= 0:
                    continue
                cands.append((-gain, w))
        if not cands:
            return None
        cands.sort()
        if os.environ.get("ZAP_WP_BOTTLENECK", "0") == "1" and batch:
            # Free-ride first (2026-10-03, experimental): a leg's Park lasts as
            # long as its longest move, so a prefix up to that length costs
            # nothing while a longer one stretches the whole leg. Rank the
            # waypoints whose prefix fits within the batch's bottleneck ahead
            # of the rest (each group still by gain).
            neck = max(math.dist((u[0], u[2]), (u[1], u[3])) for u in batch)
            fit = [c for c in cands if math.dist((sx, sy), c[1]) <= neck + 1e-9]
            if fit:
                seen_fit = {w for _g, w in fit}
                cands = fit + [c for c in cands if c[1] not in seen_fit]
        # Plan ahead for ghost pick-ups (2026-09-30). A waypoint is a held,
        # mid-air position: its row and column stay switched on while the
        # chain picks up riders, and every rider's new line crosses them. If
        # an atom sits anywhere on that row or column, some later crossing can
        # land on it and lift it. So among candidates that bank nearly the
        # best progress (>= 90% of the best gain), try the ones on lines no
        # unheld atom sits on first. Pure re-ranking: no extra moves.
        best_gain = -cands[0][0]
        if future is not None:
            # Lookahead (2026-10-01): while this atom waits at w, the chain
            # picks up the atoms in ``future`` (the still-pending riders). Each
            # pick-up lights the rider's row and column, which cross w's
            # column and row at (w.x, y_r) and (x_r, w.y); a bystander there is
            # lifted. Rank the near-best waypoints (>= 90% of the best gain) by
            # how many such corners hold a bystander, then by gain. Pure
            # re-ranking, no extra moves.
            fut = list(future)
            fut_ids = {q for q, _p in fut}
            held = self._held
            occ = {tuple(p) for i, p in enumerate(self.current_mapping)
                   if i not in held and i not in fut_ids}
            def corners(w):
                return sum(((w[0], py) in occ) + ((px, w[1]) in occ)
                           for _q, (px, py) in fut)
            near = sorted((corners(w), g, w) for g, w in cands
                          if -g >= 0.9 * best_gain)
            seen = {w for _c, _g, w in near}
            cands = [(g, w) for _c, g, w in near] + [
                c for c in cands if c[1] not in seen]
        elif os.environ.get("ZAP_WP_RANK", "0") == "1":
            # Graded (2026-10-02): rank the near-best waypoints by how many of
            # their two lines (column, row) carry an unheld atom -- 0, then 1,
            # then 2 -- then by gain. A waypoint usually shares its target's
            # column (vertical approach), so the all-or-nothing version below
            # rarely found a fully clear one and then preferred nothing.
            rows_occ, cols_occ = self._occupied_lines()
            frac = float(os.environ.get("ZAP_WP_FRAC", "0.85"))
            near = sorted(((w[0] in cols_occ) + (w[1] in rows_occ), g, w)
                          for g, w in cands if -g >= frac * best_gain)
            seen = {w for _n, _g, w in near}
            cands = [(g, w) for _n, g, w in near] + [
                c for c in cands if c[1] not in seen]
        else:
            rows_occ, cols_occ = self._occupied_lines()
            safe = [c for c in cands
                    if -c[0] >= 0.9 * best_gain
                    and c[1][1] not in rows_occ and c[1][0] not in cols_occ]
            if safe:
                seen = set(w for _g, w in safe)
                cands = safe + [c for c in cands if c[1] not in seen]

        for _neg_gain, w in cands:
            if w in occupied or w in blocked:
                continue
            if not all(self.compatible_2D((w[0], tx, w[1], ty), r)
                       for r in remainders):
                # Recorded so the stage-level pick (route_qubit) knows the
                # relaxed-banking variant could differ from this one.
                self._remainder_rejected = True
                continue
            # d_s: the PREFIX leg (start -> waypoint) is what flies in this
            # wave's BigMove alongside every batch member, so the waypoint is a
            # live AOD tone position and must clear MIN_TONE_SEP against them.
            # Only the remainder was ever checked here -- correct for the
            # ordering constraint (the prefix is a truncation of an already
            # admitted vector, so it cannot introduce a CROSSING) but not for
            # separation, which a truncation can absolutely introduce by
            # stopping the atom 1 um short of a neighbour's tone. This was the
            # last source of sub-d_s gaps on tqe: 4 cases, all chained Parks.
            if self._min_tone_sep and not all(
                    self.compatible_2D((sx, w[0], sy, w[1]), u) for u in batch):
                continue
            # Both legs must be physically flyable, not merely AOD-compatible.
            # The old on-path search got this for free: its waypoints sat on the
            # straight line to the target, so the remainder retraced a path the
            # router had already accepted. Searching the whole feasible box does
            # not — the remainder is a NEW line that can pass straight over a
            # stationary atom. Unchecked, that emitted 59 fly-overs across the
            # tqe suite (e.g. cat_n35 q7 (24,41) -> (24,52) crossing (24,42),
            # occupied by q2), where the baseline compiler emits none.
            #
            # The two legs run in different phases and so see different
            # occupancies: the prefix flies during the BigMove (atoms still at
            # their old traps), the remainder during the Park that follows it
            # (movers already landed). Testing both against the pre-move picture
            # leaves the remainder wrong, which is the residual 8 fly-overs that
            # survived the first version of this check.
            if pre_occ is not None and not self._path_clear((sx, sy), w, pre_occ):
                continue
            if post_occ is not None and not self._path_clear(w, (tx, ty), post_occ):
                continue
            # Planning-time held-atom rule: the AOD must still be able to
            # bring every held atom home from the state this waypoint creates.
            if ok is not None and not ok(w):
                continue
            return w
        return None

    def plan_waypoints(self, vectors, aod_qubits, execute_vectors, safe_vectors):
        """Waypoint decomposition: bank partial progress for deferred vectors.

        Generalises the parking-displacement mechanism (paper Fig. 5) from its
        one hard-coded geometry into a search. A vector that cannot join the
        current wave in full is not necessarily blocked along its whole path —
        often a *prefix* of it is compatible with every batch member. That
        prefix rides the wave's parallel BigMove for free; the short remainder
        then completes through the wave's existing sequential Park phase (the
        same two-segment pattern collision-parked qubits already use), with the
        atom held by the AOD throughout. The qubit therefore finishes its whole
        trip in THIS wave with the same single activate/deactivate it would
        have paid anyway, and the later wave it would have occupied disappears.

        Compatibility uses ``compatible_2D`` — the router's own ground truth —
        so an accepted prefix is exactly as safe as any baseline batch member.

        Returns {qubit: waypoint} for deferred vectors worth splitting.
        """
        if self.ablate_waypoints:
            return {}  # ABLATION: no partial progress banked (see __init__).
        min_frac = 0.02  # prefix must bank at least this fraction,
                         # else the sequential Park remainder eats the gain
        exec_set = set(execute_vectors)
        safe_set = set(safe_vectors)
        # Targets that this wave's own BigMove vacates. The Park phase runs
        # AFTER the BigMove, so a split qubit driven to such a target lands on
        # a trap its previous occupant has already left — the same chain-aware
        # reasoning ``route_qubit`` uses to admit chained vectors into a wave.
        vacated = {(u[0], u[2]) for u in execute_vectors}

        # All-or-nothing gate: a split only pays when the follow-up wave
        # disappears entirely. movement_duration is concave (sqrt), so a split
        # lengthens the splitting qubit's own move; if any deferred vector
        # still needs a later wave, banked prefixes save nothing and the
        # sequential Park remainder is pure added time. When ALL deferred
        # vectors split, the eliminated wave repays it with room to spare:
        # saved >= 2 transfers + full bottleneck, paid <= 0.71 x bottleneck.
        # Only "safe" vectors may split. A vector is unsafe when its FINAL
        # target is still occupied by another qubit that has not moved yet;
        # since the Park phase drives the split qubit all the way to that
        # target within this same wave, splitting an unsafe vector lands two
        # atoms on one trap. (Verified: allowing them produced 207 double
        # occupancies on vqc_n15 — the split must respect the router's
        # cycle-breaking rule exactly as a full move does.)
        #
        # The one safe relaxation is a CHAIN: if the occupant of the final
        # target is itself moving in this wave's BigMove, it is gone before the
        # Park phase drives the split qubit in, so the landing is clean.
        # Refusing these made a qubit chasing a trap that is being vacated right
        # now wait a whole extra wave (multiplier_n15 stage 88: q1 (28,52) ->
        # (0,52) sat out the very wave in which q11 vacated (0,52), then paid
        # two transfers and a 28 µm wave of its own).
        deferred = [(v, q) for v, q in zip(vectors, aod_qubits)
                    if v not in exec_set
                    and (v in safe_set or (v[1], v[3]) in vacated)]
        if not deferred:
            return {}

        # Greedy per-vector banking. Each deferred vector independently takes
        # the farthest waypoint that is compatible with everything already
        # committed this wave; there is no all-or-nothing requirement, because
        # a banked prefix shortens that vector's later move whether or not its
        # neighbours also split. Vectors are tried longest-first so the biggest
        # savings claim the scarce compatible geometry.
        batch = list(execute_vectors)
        accepted = {}
        accepted_remainders = []
        blocked_targets = {(u[1], u[3]) for u in batch}
        occupied = set(self.current_mapping)

        ordered = sorted(
            deferred,
            key=lambda vq: -math.dist((vq[0][0], vq[0][2]), (vq[0][1], vq[0][3])))

        # Occupancy as each leg will actually see it.
        #   pre_occ  — during the wave's BigMove: everyone still at their start,
        #              minus the atoms flying in this same wave.
        #   post_occ — during the Park that follows: movers have landed on their
        #              wave targets, non-movers have not budged.
        wave_starts = {(u[0], u[2]) for u in execute_vectors}
        pre_occ = {tuple(p) for p in self.current_mapping} - wave_starts
        post_occ = set(pre_occ) | {(u[1], u[3]) for u in execute_vectors}

        # State after the BigMove, for the finishability check: the wave's
        # full moves are home, accepted waypoint atoms sit at their waypoints.
        after = {}
        for u, qq in zip(vectors, aod_qubits):
            if u in exec_set:
                after[qq] = (u[1], u[3])
        finals = {qq: tuple(self.final_mapping[qq]) for qq in aod_qubits}

        fut_all = [(qq, tuple(self.current_mapping[qq])) for _v, qq in ordered]
        for v, q in ordered:
            sx, tx, sy, ty = v
            if math.dist((sx, sy), (tx, ty)) <= 0:
                continue

            def ok(w, q=q):
                trial = dict(after)
                trial[q] = w
                return self._finishable(trial, finals)

            # ZAP_HELD_CHAIN (2026-10-03, experiment): remainders need not
            # fly together in one leg -- _run_chain_legs chains held atoms
            # whose remainders conflict (one finishes, the other banks a
            # second waypoint). _finishable still guarantees a legal finish.
            w = self._best_waypoint(v, batch, occupied, blocked_targets,
                                    [] if self._wave_relax else accepted_remainders,
                                    min_frac, pre_occ, post_occ, ok,
                                    future=fut_all if self.ghost_lookahead else None)
            if w is None:
                continue
            accepted[q] = w
            after[q] = w
            # This qubit's own prefix lands on w, so later Park legs in the same
            # wave must treat w as occupied.
            post_occ.add(w)
            batch.append((sx, w[0], sy, w[1]))
            accepted_remainders.append((w[0], tx, w[1], ty))
            blocked_targets.add(w)
        return accepted

    def _fan_out_shifts(self, cols, parking_qs, activate_ys, other_cols=()):
        """Per-column x displacement that clears a pre-park activation collision.

        Activating rows ``activate_ys`` and columns ``cols`` energises every
        tone intersection, so a STATIONARY atom sitting at one of them would be
        picked up too. The fix is to slide the offending columns aside before
        the wave starts — one shared offset per column, never per qubit, because
        an AOD column is rigid and splitting it silently breaks a wave that was
        already validated (ising_n26 stage 7 emitted a real crossing that way).

        Three things every assignment here has to satisfy:

        * **Non-zero.** Each column in ``cols`` is one that actually collides,
          so a shift of 0 resolves nothing. The previous centred formula
          (``-len(cols) // 2 + i``) handed exactly one column a zero shift
          whenever there were two or more — that column was activated, emitted
          as a zero-distance Park, still collided, and went round the loop again
          to be activated a SECOND time. Measured on tqe: 53 of 317 shifted
          columns, each one a wasted atom transfer and a wasted Park.
        * **Order-preserving.** The shifted columns must stay in their original
          left-to-right order, or the wave's already-checked ``compatible_2D``
          relationships invert and the BigMove becomes a crossing.
        * **Landing on free space.** The displaced atoms are held mid-air by the
          AOD, but a displaced atom parked exactly on another atom's trap is a
          double occupancy in the emitted trace. Shifting blindly (the naive
          "just make every offset non-zero" version) put one there on
          ising_n26.

        So offsets are searched, nearest-first, rather than assumed: the closest
        non-zero displacement that stays right of the previous column, lands
        every atom of this column on free space, and does not simply recreate
        the collision at the new column. If nothing in range qualifies the
        column keeps the old centred offset, so this can only improve on the
        previous behaviour, never fail to produce an answer.
        """
        col_atoms = {}
        for q in parking_qs:
            col_atoms.setdefault(self.current_mapping[q][0], []).append(q)
        fanned = set(parking_qs)
        # Where every atom that is NOT being fanned currently sits. These are
        # the points a displaced atom may not land on.
        blocked = {tuple(p) for i, p in enumerate(self.current_mapping)
                   if i not in fanned}
        rows = set(activate_ys)

        shifts = {}
        prev_x = None
        self._fanout_fallback = set()
        reach = 7 if os.environ.get("ZAP_FANOUT_STRICT", "1") == "1" else 5
        for i, x in enumerate(cols):
            fallback = (-len(cols) // 2 + i) * self.PARKING_DIST
            chosen = None
            for k in range(1, reach):
                for step in (-k, k):
                    nx = x + step * self.PARKING_DIST
                    # Ordering alone is not enough: two fanned columns are two
                    # distinct AOD tones, so they must clear MIN_TONE_SEP (d_s)
                    # and not merely be strictly increasing. With d_s off this
                    # is exactly the old `nx <= prev_x` test.
                    if prev_x is not None and nx < prev_x + self._min_tone_sep:
                        continue
                    # The wave's other columns are AOD tones too: the shifted
                    # column must neither pass one nor come within d_s of it
                    # (ising_n26 stage 1: column 51 shifted to 49, 1 um from
                    # column 48, left the BigMove with no legal order at all).
                    # 2026-10-01.
                    if any(min(x, nx) <= o <= max(x, nx) or abs(nx - o) < self._min_tone_sep
                           for o in other_cols):
                        continue
                    # Would any atom of this column land on top of another?
                    if any((nx, self.current_mapping[q][1]) in blocked
                           for q in col_atoms[x]):
                        continue
                    # Does the new column still cross a stationary atom on one
                    # of the activated rows? If so it has resolved nothing.
                    if any((nx, y) in blocked for y in rows):
                        continue
                    chosen = step * self.PARKING_DIST
                    break
                if chosen is not None:
                    break
            if chosen is None:
                chosen = fallback
                self._fanout_fallback.add(x)
            shifts[x] = chosen
            prev_x = x + chosen
        return shifts

    def process_rearrangement(self, aod_qubits: list, waypoints: dict = None,
                              deferred: list = None):
        """Resolve column collisions, optional parking, then emit activate/move/deactivate for ``aod_qubits``.

        ``waypoints`` maps deferred qubits joining this wave to the intermediate
        point their prefix move ends at; they complete to ``final_mapping`` via
        the sequential Park phase, exactly like collision-parked qubits.
        """
        waypoints = waypoints or {}
        aod_qubits = list(aod_qubits) + [q for q in waypoints if q not in aod_qubits]
        activated_qs = []
        self._wave_dropped = set()
        wave_defer = os.environ.get("ZAP_WAVE_DEFER", "1") == "1"
        deferred = list(deferred or [])

        def drop_from_wave(cands):
            """Leave unlifted wave atoms whose pick-up cannot be made clean
            for the chain / a later wave (2026-10-02). A wave-mate landing on
            a dropped atom's trap (this wave's endpoint or its final trap)
            is dropped too if it is not lifted yet; if it is, the dropped
            atom is kept instead."""
            if getattr(self, '_no_wave_drop', False):
                return False   # last wave made no progress: lift, don't drop
            D = {q for q in cands if q not in self._held}
            # Atoms taken back out of D because something held lands on
            # their trap stay out: they are lifted with the wave. Without
            # this, such an atom could be re-added as a lander and taken out
            # again forever (ising_n200 / qft_n950 with ZAP's fixed array and
            # ASAP hung here, 2026-10-04). Each atom now changes state at
            # most twice, so the loop terminates.
            kept = set()
            changed = True
            while changed and D:
                changed = False
                at = {tuple(self.current_mapping[d]): d for d in D}
                for m in aod_qubits:
                    if m in D:
                        continue
                    for t in (tuple(tmp_end_locs[m]), tuple(self.final_mapping[m])):
                        d = at.get(t)
                        if d is None:
                            continue
                        if m in self._held and (os.environ.get("ZAP_DEFER_HELD", "0") == "1"
                                                or os.environ.get("ZAP_WAVE_DEFER_HELD", "0") == "1"):
                            continue
                        if m in self._held or m in kept:
                            D.discard(d)
                            kept.add(d)
                        else:
                            D.add(m)
                        changed = True
                        break
                    if changed:
                        break
            if not D:
                return False
            aod_qubits[:] = [q for q in aod_qubits if q not in D]
            for q in D:
                if q in parking_end_qubits:
                    parking_end_qubits.remove(q)
                tmp_end_locs[q] = tuple(self.final_mapping[q])
                c, f = self.current_mapping[q], self.final_mapping[q]
                deferred.append(((c[0], f[0], c[1], f[1]), q))
            self._wave_dropped |= D
            return True
        parking_end_qubits = []
        # Shallow copy is sufficient: elements are immutable (x, y) tuples and
        # only ever reassigned wholesale (e.g. tmp_end_locs[q] = w below), never
        # mutated in place, so deepcopy here was pure overhead.
        tmp_begin_locs = list(self.current_mapping)
        tmp_end_locs = list(self.final_mapping)
        for q, w in waypoints.items():
            tmp_end_locs[q] = w            # BigMove prefix stops at the waypoint
            parking_end_qubits.append(q)   # stay activated; Park finishes the trip
        while True:
            activate_xs = [self.current_mapping[q][0] for q in aod_qubits]
            activate_ys = [self.current_mapping[q][1] for q in aod_qubits]
            aod_qubits_xys = [self.current_mapping[q] for q in aod_qubits]

            activate_xys = [(x, y) for x in activate_xs for y in activate_ys]

            col_xs = []
            for x, y in activate_xys:
                if (x, y) in self.current_mapping and (x, y) not in aod_qubits_xys:
                    col_xs.append(x)
            col_xs = list(set(col_xs))

            parking_qs = [aod_qubits[i] for i in range(len(aod_qubits)) if activate_xs[i] in col_xs]

            if parking_qs:
                # Pick the blocked columns up in clean rounds (2026-10-01).
                # Lifting every blocked column at once re-creates the very
                # row x column grid that flagged them, so it lifted the same
                # bystanders (114 of 265 ghost pick-ups on tqe). Take only the
                # column groups whose pick-up, together with what the AOD
                # already holds, lifts nobody; the rest wait for the next round,
                # by which time the accepted columns have slid off their traps.
                # A lone column group with nothing held can never lift anyone,
                # so a round only comes up empty when atoms are already held;
                # then fall back to the old all-at-once pick-up.
                groups = {}
                for q in parking_qs:
                    if q not in self._held:
                        groups.setdefault(self.current_mapping[q][0], []).append(q)
                accepted = []
                pcols = {self.current_mapping[h][0] for h in self._held}
                prows = {self.current_mapping[h][1] for h in self._held}
                pocc = self._occ_nonheld()
                pmem = set(self._held)
                order = sorted(groups)
                if os.environ.get("ZAP_PREPARK_ORDER", ("1" if os.environ.get("ZAP_STRICT", "1") == "1" else "0")) == "1":
                    # Groups other wave atoms will land on go first, while the
                    # fewest lines are lit: once a wave-mate bound for their
                    # trap is held, they can neither be dropped nor lifted
                    # cleanly (ising_n300 q184 / q186). 2026-10-02.
                    landing = {tuple(tmp_end_locs[m]) for m in aod_qubits} | {
                        tuple(self.final_mapping[m]) for m in aod_qubits}
                    order = sorted(groups, key=lambda x: (
                        not any(tuple(self.current_mapping[q]) in landing for q in groups[x]), x))
                for x in order:
                    # Incremental: the group may join if it adds no new
                    # lifted corner (see _clean_round).
                    tc, tr, tm = set(pcols), set(prows), pmem | set(groups[x])
                    ok = True
                    for q in groups[x]:
                        xy = self.current_mapping[q]
                        if (not self._sep_ok(xy, tc, tr)
                                or self._new_corners(xy, tc, tr, pocc, tm)):
                            ok = False
                            break
                        tc.add(xy[0]); tr.add(xy[1])
                    if ok:
                        accepted += groups[x]
                        pcols, prows, pmem = tc, tr, tm
                if groups and not accepted and wave_defer:
                    if drop_from_wave([q for g in groups.values() for q in g]):
                        continue
                    if os.environ.get("ZAP_DEBUG_PREPARK"):
                        with open(os.environ["ZAP_DEBUG_PREPARK"], "a") as _f:
                            _f.write(json.dumps({
                                "inst": len(self.results_code['instructions']),
                                "groups": {str(x): [(q, list(self.current_mapping[q])) for q in g] for x, g in groups.items()},
                                "held": [(h, list(self.current_mapping[h])) for h in self._held],
                                "wave": [(q, list(self.current_mapping[q]), list(tmp_end_locs[q]), list(self.final_mapping[q])) for q in aod_qubits],
                            }) + "\n")
                if groups and not accepted:
                    # No group lifts cleanly: take only the least damaging
                    # one this round (it slides off, and the next round
                    # re-tests the rest). Lifting every group at once crossed
                    # every group's column with every other's rows -- 21,381
                    # lifts on ising_n1000. 2026-10-01.
                    self._prepark_fallbacks = getattr(self, '_prepark_fallbacks', 0) + 1
                    def dmg(g):
                        tc, tr, tm = set(pcols), set(prows), pmem | set(g)
                        hit = set()
                        for q in g:
                            xy = self.current_mapping[q]
                            hit |= self._new_corners(xy, tc, tr, pocc, tm)
                            tc.add(xy[0]); tr.add(xy[1])
                        return len(hit)
                    accepted = list(min(groups.values(), key=dmg))
                # Atoms already slid in an earlier round stay where they are:
                # re-sliding them every round walked them across the storage
                # grid (qaoa_n950 q282 293->291->289->287, landing on q371's
                # trap mid-way). If only already-held atoms are still blocked,
                # the pre-park has done what it can.
                if not accepted:
                    break
                if os.environ.get("ZAP_FANOUT_STRICT", "1") == "1":
                    # A column with no legal sideways offset used to get an
                    # unchecked centred one -- 1 um from another tone on the
                    # stagger grid (rule c). Leave such groups out of this
                    # round instead, before they are lifted. 2026-10-02.
                    for _t in range(len(accepted) + 1):
                        acols = sorted({self.current_mapping[q][0] for q in accepted})
                        self._fan_out_shifts(
                            acols, accepted, activate_ys,
                            other_cols={self.current_mapping[q][0] for q in aod_qubits} - set(acols))
                        fb = self._fanout_fallback
                        if not fb:
                            break
                        if wave_defer:
                            drop_from_wave([q for q in accepted
                                            if self.current_mapping[q][0] in fb])
                        accepted = [q for q in accepted
                                    if self.current_mapping[q][0] not in fb
                                    and q in aod_qubits]
                        if not accepted:
                            break
                    if not accepted:
                        break
                parking_qs = accepted
                if os.environ.get("ZAP_SAFE_LIFT", ("1" if os.environ.get("ZAP_STRICT", "1") == "1" else "0")) == "1" and (self._lift_ghosts(accepted) or self._persist()):
                    if self._safe_lift(accepted, future=[q for q in aod_qubits if q not in activated_qs and q not in accepted]):
                        for _h in self._held:
                            tmp_begin_locs[_h] = tuple(self.current_mapping[_h])
                else:
                    self.write_activate_instruction(accepted)
                activated_qs += accepted
                parking_qs = sorted(parking_qs, key=lambda x: self.current_mapping[x][0])
                # Fan the blocked columns apart, ONE offset per distinct column.
                #
                # The offsets used to be indexed per qubit, so two qubits sitting
                # in the same column were pushed to different x. An AOD column is
                # rigid — it cannot be split — and the split also silently broke
                # the wave that had already been validated: two moves with equal
                # start x and equal end x are compatible, but after the fan-out
                # their starts differ while their ends still coincide, which is
                # exactly the crossing ``compatible_2D`` forbids. (ising_n26
                # stage 7: q0 and q10 both left x=30 for x=16, were fanned to 31
                # and 32, and were emitted as a crossing.)
                cols = sorted({self.current_mapping[q][0] for q in parking_qs})
                col_shift = self._fan_out_shifts(
                    cols, parking_qs, activate_ys,
                    other_cols={self.current_mapping[q][0] for q in aod_qubits} - set(cols))
                if self.proxy_audit and self._audit is not None:
                    self._audit['prepark_rounds'] += 1
                    self._audit['prepark_cols'] += len(cols)
                    self._audit['prepark_zero_shift'] += sum(
                        1 for v in col_shift.values() if v == 0)
                row_dy = self._prepark_row_shift(accepted, aod_qubits, col_shift,
                                                 tmp_end_locs)
                for q in parking_qs:
                    tmp_begin_locs[q] = (
                        tmp_begin_locs[q][0] + col_shift[self.current_mapping[q][0]],
                        tmp_begin_locs[q][1] + (row_dy if q in accepted else 0),
                    )
                    parking_end_qubits.append(q)
                self.write_move_batched(parking_qs, "Park", tmp_begin_locs, guard=False)
            else:
                break

        if wave_defer and list(set(aod_qubits) - set(activated_qs)):
            # Check before the main lift (and before paths are re-verified): atoms whose pick-up would put
            # a lit corner on a bystander leave the wave (2026-10-02).
            rest = [q for q in aod_qubits if q not in activated_qs]
            ok = set(self._clean_subset(rest))
            bad = [q for q in rest if q not in ok]
            if bad:
                drop_from_wave(bad)

        # The path-check diversion that used to follow (ZAP's rule that an atom
        # may not fly exactly over an occupied trap: its destination was nudged
        # aside) was removed 2026-10-31. It is not an AOD rule (Sec. II-B, the
        # verifier), and it changed a move's destination after the waypoints
        # of its wave-mates had been planned against it, which caused the
        # mid-air deadlocks traced on vqc_n15 and qft_n200 (stage 1176: q137
        # nudged to (2, 24) stranded q140 and q155). tqe +0.0077 alone.

        if list(set(aod_qubits) - set(activated_qs)):
            _main = list(set(aod_qubits) - set(activated_qs))
            if os.environ.get("ZAP_SAFE_LIFT", ("1" if os.environ.get("ZAP_STRICT", "1") == "1" else "0")) == "1" and (self._lift_ghosts(_main) or self._persist()):
                if self._safe_lift(_main):
                    for _h in self._held:
                        tmp_begin_locs[_h] = tuple(self.current_mapping[_h])
            else:
                self.write_activate_instruction(_main)

        # Moves that cannot be made legal against the atoms held stay where
        # they are; the chain below takes them to their final traps.
        stuck = self.write_move_batched(aod_qubits, "BigMove", tmp_end_locs,
                                        allow_defer=True)
        parking_end_qubits.extend(stuck)

        # An atom the BigMove carried all the way home is finished even if it
        # went through the pre-park fan-out (which put it in
        # parking_end_qubits). Holding it back kept its row and column tones
        # on while the chain's next leg moved other atoms across them -- 157
        # of the 182 held-atom crossings on tqe (tools/verify_held.py) were a
        # home atom still held this way (bv_n14 inst 26: q2 home at (21,15)
        # held while q3 flew from row 14 to row 25). 2026-09-30.
        at_home = {q for q in aod_qubits
                   if tuple(self.current_mapping[q]) == tuple(self.final_mapping[q])}
        parking_end_qubits = [q for q in dict.fromkeys(parking_end_qubits)
                              if q not in at_home]
        finished = [q for q in aod_qubits if q in at_home]
        if finished:
            # Releasing an atom means ramping down its AOD row or column tone,
            # and that tone is shared with every atom on the same line. An atom
            # still held AWAY from a trap has nothing to catch it, so it must
            # keep at least one of its two tones. A finished atom whose row AND
            # column are both shared with such an atom therefore cannot be put
            # down yet — it simply stays held until the closing Park, where
            # everything is deposited together.
            # (knn_n25: q17 at (16,64) sat on the column of two mid-air atoms
            # and the row of a third; dropping it there would have dropped them.)
            # An atom over a trap that is not its own counts too: it still has
            # to move (qft_n500 q104 would have been dropped at (-1, 94), then
            # moved by the next Park). 2026-11-03.
            midair = list(set(parking_end_qubits))
            rows = {self.current_mapping[q][1] for q in midair}
            cols = {self.current_mapping[q][0] for q in midair}
            safe, stuck = [], []
            for q in finished:
                x, y = self.current_mapping[q]
                (safe if (y not in rows or x not in cols) else stuck).append(q)
            if safe:
                self.write_deactivate_instruction(safe)
            parking_end_qubits.extend(stuck)

        if parking_end_qubits:
            return self._run_chain_legs(set(parking_end_qubits), deferred)
        return set()

    def _held_leg_plan(self, movers, already_arrived):
        """Where each held mover goes this leg when they cannot all finish
        together (ZAP_HELD_CHAIN, 2026-10-03, experiment).

        Banking used to require every remainder of a wave to fit in one leg,
        and a held atom always flew straight home in the next leg. Instead:
        a compatible set of held movers finishes, and each other one banks a
        waypoint against them -- the same ``_best_waypoint`` riders use -- or,
        failing that, stays put if that is legal as a stationary held atom.
        Tried from several seeds; keeps the plan that finishes the most atoms,
        then banks the most progress. Every plan finishes at least one atom,
        so the chain still terminates. Returns ``{q: end}`` for every mover,
        or None when they are already compatible (or no plan is legal).
        """
        cm, fm = self.current_mapping, self.final_mapping
        vec = {q: (cm[q][0], fm[q][0], cm[q][1], fm[q][1]) for q in movers}
        ms = list(movers)
        if all(self.compatible_2D(vec[a], vec[b])
               for i, a in enumerate(ms) for b in ms[i + 1:]):
            return None
        order = sorted(ms, key=lambda q: -math.dist(
            (vec[q][0], vec[q][2]), (vec[q][1], vec[q][3])))
        leg_set = set(ms)
        vacated = {tuple(cm[q]) for q in ms}
        occupied = {tuple(p) for i, p in enumerate(cm) if i not in leg_set}
        pre = occupied - vacated
        finals = {q: tuple(fm[q]) for q in list(ms) + list(already_arrived)}
        best = None
        for seed in order[:8]:
            fin = [seed]
            for q in order:
                if q not in fin and all(self.compatible_2D(vec[q], vec[f]) for f in fin):
                    fin.append(q)
            ends = {f: tuple(fm[f]) for f in fin}
            batch = [vec[f] for f in fin]
            blocked = {tuple(fm[q]) for q in ms}
            prog, legal = 0.0, True
            for r in order:
                if r in ends:
                    continue
                v = vec[r]
                after = dict(ends)
                after.update({q: tuple(cm[q]) for q in already_arrived})
                after.update({q: tuple(cm[q]) for q in order if q not in ends and q != r})

                def ok(w, r=r, after=after):
                    trial = dict(after)
                    trial[r] = w
                    return self._finishable(trial, finals)

                post = pre | {(u[1], u[3]) for u in batch}
                w = self._best_waypoint(v, batch, occupied | blocked, blocked, [],
                                        0.02, pre, post, ok)
                if w is not None:
                    ends[r] = tuple(w)
                    batch.append((v[0], w[0], v[2], w[1]))
                    blocked.add(tuple(w))
                    t = (v[1], v[3])
                    prog += math.dist((v[0], v[2]), t) - math.dist(w, t)
                    continue
                st = (cm[r][0], cm[r][0], cm[r][1], cm[r][1])
                if not all(self.compatible_2D(u, st) for u in batch):
                    legal = False
                    break
                ends[r] = tuple(cm[r])
                batch.append(st)
            if not legal:
                continue
            key = (len(fin), prog)
            if best is None or key > best[0]:
                best = (key, ends)
        return None if best is None else best[1]

    def _run_chain_legs(self, held, deferred):
        """Run the wave's remaining legs as a rolling chain.

        The Park used to be a single closing move: every atom still holding an
        unfinished vector flew straight to its trap, alone. But that leg IS a
        movement vector like any other, so it can host exactly the same
        decomposition the BigMove hosts — a deferred atom may fly with it, in
        full or as a prefix, and if it takes a prefix then ITS remainder becomes
        the next leg, which can host the next atom. Instead of

            wave{A,B} -> park{B}   then   wave{C,D} -> park{D}

        the tail becomes

            wave{A,B} -> leg{B',C} -> leg{C',D} -> ...

        Each link costs no extra transfers: a rider is picked up and put down
        once, exactly as its own wave would have cost, and everything else is
        already in the air. What disappears is the later wave's travel time and,
        when a wave empties entirely, its activate/deactivate overhead too.

        Self-limiting: a leg only adds to ``partial`` via ``_best_waypoint``,
        which requires >= ``min_frac`` (2%) real progress toward the target
        to accept a split at all, so every productive leg strictly shrinks
        ``pending`` by at least one genuinely-progressed qubit. The loop's
        own ``if not partial and not full: break`` below fires the instant a
        leg accomplishes nothing, so this naturally terminates within at
        most ``len(deferred)`` productive legs — no arbitrary count needed.
        ``chain_max_legs`` (see ``__init__``) is a purely defensive ceiling
        for this reason, not a tuned parameter; RAISED 2026-09-14 after
        measuring the old default of 8 (tuned against tqe, where it never
        actually bound) was truncating real progress on `graphs`' wider
        stages — see ``__init__``'s comment for the numbers.
        """
        absorbed = set()
        pending = list(deferred or [])
        # ABLATION (see __init__): skipping the leg loop entirely falls through
        # to the unconditional single Park + Deactivate tail below, which IS
        # the pre-chain closing behaviour -- no rolling legs, no riders, no
        # absorbed deferrals. Nothing else in the wave loop changes.
        n_legs = 0 if self.ablate_chain else self.chain_max_legs
        for _leg in range(n_legs):
            movers = [q for q in held
                      if tuple(self.current_mapping[q]) != tuple(self.final_mapping[q])]
            # Qubits already at their final position on entry (e.g. carried
            # there by this wave's own BigMove) but still logically "held"
            # because they went through collision-parking. They are not
            # movers, so they must not be dropped when `held` below is
            # rebuilt from this leg's movers -- otherwise they never reach
            # the arrived/safe check and never get deactivated at all
            # (found via knn_n25: qubits 3,6,7,10,15,18,19 stayed "Activated"
            # for the rest of the circuit because exactly this happened).
            already_arrived = [q for q in held if q not in movers]
            if not movers:
                held = set(already_arrived)
                break
            leg_vecs = [(self.current_mapping[q][0], self.final_mapping[q][0],
                         self.current_mapping[q][1], self.final_mapping[q][1])
                        for q in movers]

            # Held movers that cannot all finish together (ZAP_HELD_CHAIN):
            # some finish, the rest bank a waypoint or wait this leg.
            held_ends = None
            if len(movers) > 1:
                if self._held_chain:
                    held_ends = self._held_leg_plan(movers, already_arrived)
                elif self._pick_stage and not self._held_conflict:
                    _v = [(self.current_mapping[q][0], self.final_mapping[q][0],
                           self.current_mapping[q][1], self.final_mapping[q][1])
                          for q in movers]
                    self._held_conflict = not all(
                        self.compatible_2D(a, b)
                        for i, a in enumerate(_v) for b in _v[i + 1:])
            if held_ends is not None:
                leg_vecs = [(self.current_mapping[q][0], held_ends[q][0],
                             self.current_mapping[q][1], held_ends[q][1])
                            for q in movers]

            # Riders that can finish outright on this leg.
            pend_map = {q: v for v, q in pending}
            admit = (self._admitter()
                     if os.environ.get("ZAP_LEG_MODE", "plan") == "admit" else None)
            full = self._park_riders(movers, pending, admit, ends=held_ends)
            targets = dict(enumerate(self.final_mapping))
            if held_ends is not None:
                targets.update(held_ends)
            if full:
                if not self.ghost_leg_rounds:
                    self.write_activate_instruction(full)
                for q in full:
                    a = tuple(self.current_mapping[q])
                    b = tuple(self.final_mapping[q])
                    leg_vecs.append((a[0], b[0], a[1], b[1]))
                absorbed |= set(full)
                pending = [(v, q) for v, q in pending if q not in absorbed]

            # Riders that cannot finish but can bank a prefix: they join with a
            # waypoint and their remainder becomes the next leg. This is the
            # recursive step — without it the chain stops after one link.
            partial = {}
            # SAFETY (the rule plan_waypoints learned the hard way): a split
            # atom is driven all the way to its FINAL trap by a later leg of
            # this same chain, so that trap must genuinely be free. Skipping
            # this check put 58 double occupancies on vqc_n15 alone — the
            # same failure mode plan_waypoints documents at 207.
            leg_set = set(movers) | set(full)
            vacated = {tuple(self.current_mapping[q]) for q in leg_set}
            occupied = {tuple(p) for i, p in enumerate(self.current_mapping)
                        if i not in leg_set}
            claimed = {tuple(self.final_mapping[q]) for q in leg_set}
            blocked = {(u[1], u[3]) for u in leg_vecs} | claimed
            remainders = []
            refused_partial = []
            for v, q in sorted(pending, key=lambda t: -math.dist(
                    (t[0][0], t[0][2]), (t[0][1], t[0][3]))):
                target = (v[1], v[3])
                # final trap must be free, or freed by this chain
                if target in claimed:
                    continue
                if target in occupied and target not in vacated:
                    continue
                pre = occupied - vacated
                post = (occupied - vacated) | {(u[1], u[3]) for u in leg_vecs}
                # State after this leg: movers and full riders home, earlier
                # partial riders at their waypoints, arrived atoms where they
                # are.
                after = {r: tuple(self.final_mapping[r]) for r in leg_set}
                after.update({r: tuple(self.current_mapping[r])
                              for r in already_arrived})
                if held_ends is not None:
                    after.update(held_ends)
                after.update(partial)
                finals = {r: tuple(self.final_mapping[r])
                          for r in list(after) + [q]}

                def ok(w, q=q, after=after, finals=finals):
                    trial = dict(after)
                    trial[q] = w
                    return self._finishable(trial, finals)

                fut = ([(r, tuple(self.current_mapping[r])) for _u, r in pending
                        if r != q and r not in partial]
                       if self.ghost_lookahead else None)
                w = self._best_waypoint(v, leg_vecs, occupied | blocked,
                                        blocked, remainders, 0.02,
                                        pre, post, ok, future=fut)
                if w is None:
                    continue
                if self.ghost_refuse and self._ghosts(
                        (list(full) if self.ghost_leg_rounds else [])
                        + list(partial) + [q]):
                    continue
                if admit is not None and not admit(q):
                    refused_partial.append((q, v, w, target))
                    continue
                partial[q] = w
                leg_vecs.append((v[0], w[0], v[2], w[1]))
                remainders.append((w[0], v[1], w[1], v[3]))
                blocked.add(w)
                claimed.add(target)
            if admit is not None and not full and not partial:
                # Admission must not end the chain: a leg with no rider stops
                # it, and every refused rider then waits for a whole new wave.
                # Take the best refused rider anyway (its lift is counted).
                rp = {r[0] for r in refused_partial}
                rf = [q for q in admit.refused if q not in rp]
                if rf:
                    q = rf[0]
                    admit.force(q)
                    full = [q]
                    a = tuple(self.current_mapping[q]); b = tuple(self.final_mapping[q])
                    leg_vecs.append((a[0], b[0], a[1], b[1]))
                    absorbed.add(q)
                    pending = [(vv, qq) for vv, qq in pending if qq not in absorbed]
                elif refused_partial:
                    q, v, w, target = refused_partial[0]
                    admit.force(q)
                    partial[q] = w
                    leg_vecs.append((v[0], w[0], v[2], w[1]))
                    remainders.append((w[0], v[1], w[1], v[3]))
                    blocked.add(w)
                    claimed.add(target)
            if partial:
                if not self.ghost_leg_rounds:
                    self.write_activate_instruction(list(partial))
                absorbed |= set(partial)
                pending = [(v, q) for v, q in pending if q not in absorbed]

            leg_qs = list(movers) + list(full) + list(partial)
            for q, w in partial.items():
                targets[q] = w
            if self.ghost_leg_rounds and (full or partial):
                if admit is not None:
                    self.write_activate_instruction(list(full) + list(partial))
                elif os.environ.get("ZAP_LEG_MODE", "plan") == "plan":
                    dropped = self._pick_up_planned(list(full) + list(partial),
                                                    leg_qs, targets)
                    if dropped:
                        dset = set(dropped)
                        leg_qs = [q for q in leg_qs if q not in dset]
                        absorbed -= dset
                        for q in dropped:
                            targets[q] = tuple(self.final_mapping[q])
                            pending.append((pend_map[q], q))
                        full = [q for q in full if q not in dset]
                        partial = {q: w for q, w in partial.items() if q not in dset}
                elif os.environ.get("ZAP_LEG_MODE", "plan") == "old":
                    self._pick_up_rounds_old(list(full) + list(partial), leg_qs, targets)
                else:
                    self._pick_up_rounds(list(full) + list(partial), leg_qs, targets)
            # A home atom still held keeps its row and column tones on, and
            # this leg's moves were only checked against each other. If any of
            # them would cross such an atom's tones, release it first (same
            # safety rule as below: its row or its column must carry no
            # mid-air atom). Released only when the leg actually crosses it,
            # so atoms nothing crosses still share the closing Deactivate.
            # 2026-09-30; before this, 157 of 182 held-atom crossings on tqe
            # were a home atom held through a leg that crossed it.
            if already_arrived:
                leg_vecs_all = [(self.current_mapping[q][0], targets[q][0],
                                 self.current_mapping[q][1], targets[q][1])
                                for q in leg_qs]
                midair = [q for q in list(leg_qs) + list(already_arrived)
                          if tuple(self.current_mapping[q]) != tuple(self.final_mapping[q])]
                rows = {self.current_mapping[q][1] for q in midair}
                cols = {self.current_mapping[q][0] for q in midair}
                early = []
                for q in already_arrived:
                    x, y = self.current_mapping[q]
                    if (x, y) != tuple(self.final_mapping[q]):
                        # Slid off its trap by _safe_lift (allow_home=True
                        # inside legs): releasing it here drops it between
                        # traps (qft_n300 q42 at (80, 35), final (80, 37),
                        # 2026-10-06). Keep it held; the closing legs fly it
                        # home before the final Deactivate.
                        continue
                    if all(self.compatible_2D(v, (x, x, y, y)) for v in leg_vecs_all):
                        continue
                    if y not in rows or x not in cols:
                        early.append(q)
                if early:
                    self.write_deactivate_instruction(early)
                    already_arrived = [q for q in already_arrived if q not in early]
            _before = [tuple(self.current_mapping[q]) for q in leg_qs]
            self.write_move_batched(leg_qs, "Park", targets, allow_defer=True)
            leg_moved = _before != [tuple(self.current_mapping[q]) for q in leg_qs]

            held = set(leg_qs) | set(already_arrived)
            # Release everyone who is home and can be let go safely.
            arrived = [q for q in held
                       if tuple(self.current_mapping[q]) == tuple(self.final_mapping[q])]
            stay = held - set(arrived)
            midair = stay
            rows = {self.current_mapping[q][1] for q in midair}
            cols = {self.current_mapping[q][0] for q in midair}
            safe = [q for q in arrived
                    if self.current_mapping[q][1] not in rows
                    or self.current_mapping[q][0] not in cols]
            if safe:
                self.write_deactivate_instruction(safe)
            held = held - set(safe)
            unfinished_held = held_ends is not None and any(
                tuple(held_ends[q]) != tuple(self.final_mapping[q]) for q in movers)
            if not partial and not full and not (unfinished_held and leg_moved):
                break

        set_down = []
        if held:
            # Closing legs: keep going while moves get deferred (each round an
            # atom that blocked the others may land and be released); only a
            # round with no progress at all falls back to unchecked emission.
            for _ in range(len(held) + 1):
                todo = [q for q in held
                        if tuple(self.current_mapping[q]) != tuple(self.final_mapping[q])]
                if not todo:
                    break
                left = self.write_move_batched(todo, "Park", self.final_mapping,
                                               allow_defer=True)
                if len(left) == len(todo):
                    # Deadlock among mid-air atoms: set one down on a free trap
                    # (legally), finish the rest, and bring it home afterwards.
                    q = self._set_down_elsewhere(left)
                    if q is None:
                        self.results_code['unchecked_finish'] = self.results_code.get('unchecked_finish', 0) + 1
                        self.write_move_batched(left, "Park", self.final_mapping)
                        break
                    set_down.append(q)
                    held.discard(q)
            # Rounds exhausted with atoms still between traps (random_n1000,
            # 2026-10-06: q418 at (142, 29) bound for (140, 103) was released
            # mid-air by the Deactivate below). Never release off a trap: set
            # stragglers down legally while that works, else fall back to the
            # same unchecked finish the deadlock branch uses.
            for _ in range(len(held) + 1):
                stray = [q for q in held
                         if tuple(self.current_mapping[q]) != tuple(self.final_mapping[q])
                         and tuple(self.current_mapping[q]) not in self._slm_set]
                if not stray:
                    break
                q = self._set_down_elsewhere(stray)
                if q is None:
                    self.results_code['unchecked_finish'] = self.results_code.get('unchecked_finish', 0) + 1
                    self.write_move_batched(stray, "Park", self.final_mapping)
                    break
                set_down.append(q)
                held.discard(q)
            self.write_deactivate_instruction(list(held))
        # Atoms set down to break a deadlock: the AOD is empty now, so each
        # finishes alone -- a single held atom can never cross anything.
        for q in set_down:
            tgt = tuple(self.final_mapping[q])
            if os.environ.get("ZAP_STRICT", "1") == "1" and any(
                    tuple(self.current_mapping[o]) == tgt for o in range(self.n_q) if o != q):
                # Final trap still occupied (2026-10-02): this unchecked
                # solo trip is what used to put two atoms on one trap. Leave
                # q where it was set down; the stage loop routes it again.
                self._wave_dropped = getattr(self, '_wave_dropped', set()) | {q}
                absorbed.discard(q)
                continue
            self.write_activate_instruction([q])
            self.write_move_instruction([q], "Park", self.final_mapping)
            self.write_deactivate_instruction([q])
        return absorbed

    def _set_down_elsewhere(self, left):
        """Break a mid-air deadlock by landing one atom on a free trap.

        For each stuck atom (nearest-first free traps, up to 40 tried), the
        move there must clear every other held atom as a stationary atom, its
        path must be clear, and the atom must be releasable there (its row or
        its column carries no other held atom). Returns the atom set down, or
        None.
        """
        for q in left:
            if (os.environ.get("ZAP_STRICT", "1") == "1" and q not in self._held
                    and tuple(self.current_mapping[q]) in self._slm_set):
                return q   # put back on a trap already: nothing to move (2026-10-02)
        # An unfinished atom still over a trap needs no flight: release it there
        # and it frees the lines it shares (qft_n600: q92 on its own trap kept
        # home q102's column lit, and q102's row held q87). 2026-11-04.
        for q in left:
            if q not in self._held or tuple(self.current_mapping[q]) not in self._slm_set:
                continue
            x, y = self.current_mapping[q]
            others = [o for o in self._held if o != q]
            if (all(self.current_mapping[o][1] != y for o in others)
                    or all(self.current_mapping[o][0] != x for o in others)):
                self.write_deactivate_instruction([q])
                return q
        occupied = {tuple(p) for p in self.current_mapping}
        claimed = {tuple(self.final_mapping[q]) for q in range(self.n_q)
                   if tuple(self.current_mapping[q]) != tuple(self.final_mapping[q])}
        free = [t for t in self._slm_set if t not in occupied and t not in claimed]
        for q in left:
            sx, sy = self.current_mapping[q]
            others = [o for o in self._held if o != q]
            # 400 nearest, not 40 (2026-10-02): when every near trap failed,
            # the caller fell back to an unchecked Park, the source of the
            # remaining collective-motion and crossing violations.
            for t in sorted(free, key=lambda t: (t[0] - sx) ** 2 + (t[1] - sy) ** 2)[:400]:
                v = (sx, t[0], sy, t[1])
                if not all(self.compatible_2D(v, self._stationary(o)) for o in others):
                    continue
                if not self._path_clear((sx, sy), t, occupied - {(sx, sy)}):
                    continue
                if not (all(self.current_mapping[o][1] != t[1] for o in others)
                        or all(self.current_mapping[o][0] != t[0] for o in others)):
                    continue
                self.write_move_instruction([q], "Park", {q: t})
                self.write_deactivate_instruction([q])
                self.results_code['set_downs'] = self.results_code.get('set_downs', 0) + 1
                return q
        if os.environ.get("ZAP_GROUP_SETDOWN", ("1" if os.environ.get("ZAP_STRICT", "1") == "1" else "0")) != "1":
            return None
        # Group set-down (2026-10-02). The usual deadlock: two mid-air atoms
        # share a row (or column) tone but need different rows next, so
        # neither can move alone without splitting the tone (rule a) and
        # neither can be released mid-air (qft_n300: q15 (199,48) and q27
        # (221,48)). Move q's whole tone-connected group by ONE vector --
        # legal by construction among themselves -- so that q lands on a
        # free trap whose column (or row) it holds alone; release q there.
        held = list(self._held)
        pos = self.current_mapping
        tiers = [False]
        if os.environ.get("ZAP_STRICT", "1") == "1":
            # Third tier (2026-10-03): move EVERY held atom by one vector.
            # With nothing held left stationary the move cannot cross, split
            # or merge a tone; only landings, paths and the release of q need
            # checking (graphs_500 with forced evacuation hit a deadlock no
            # tone-connected group could break, and fell through to the
            # unchecked Park in the caller).
            tiers.append(True)
        for whole in tiers:
            for q in left:
                if whole:
                    group = set(held)
                else:
                    group, todo = {q}, [q]
                    while todo:
                        a = todo.pop()
                        for o in held:
                            if o not in group and (pos[o][0] == pos[a][0]
                                                   or pos[o][1] == pos[a][1]):
                                group.add(o); todo.append(o)
                    if len(group) < 2:
                        continue
                if q not in group:
                    continue
                others = [o for o in held if o not in group]
                sx, sy = pos[q]
                for t in sorted(free, key=lambda t: (t[0] - sx) ** 2 + (t[1] - sy) ** 2)[:400]:
                    dx, dy = t[0] - sx, t[1] - sy
                    ok = True
                    for m in group:
                        mx, my = pos[m]
                        nm = (mx + dx, my + dy)
                        if m != q and nm in occupied:
                            ok = False; break
                        v = (mx, nm[0], my, nm[1])
                        if not all(self.compatible_2D(v, self._stationary(o)) for o in others):
                            ok = False; break
                        if not self._path_clear((mx, my), nm, occupied - {(mx, my)}):
                            ok = False; break
                    if not ok:
                        continue
                    rest = [o for o in held if o != q]
                    newp = {m: (pos[m][0] + dx, pos[m][1] + dy) for m in group}
                    at = lambda o: newp.get(o, pos[o])
                    if not (all(at(o)[1] != t[1] for o in rest)
                            or all(at(o)[0] != t[0] for o in rest)):
                        continue
                    self.write_move_instruction(sorted(group), "Park", newp)
                    self.write_deactivate_instruction([q])
                    self.results_code['set_downs'] = self.results_code.get('set_downs', 0) + 1
                    return q
        return None

    def _admitter(self):
        """Rider admission test for one chain leg (2026-10-01).

        Returns admit(q): True, and q's lines join the lit set, if lifting q
        together with everything held and already admitted adds no NEW lifted
        corner; False otherwise (q stays pending for a later leg). All
        admitted riders then go up in a single Activate. Unlike the earlier
        whole-grid refusal, corners that are already lit do not block anyone.
        """
        pos = self.current_mapping
        cols = {pos[h][0] for h in self._held}
        rows = {pos[h][1] for h in self._held}
        occ = self._occ_nonheld()
        members = set(self._held)
        def admit(q):
            members.add(q)
            if self._new_corners(pos[q], cols, rows, occ, members):
                members.discard(q)
                admit.refused.append(q)
                return False
            cols.add(pos[q][0]); rows.add(pos[q][1])
            return True
        def force(q):
            members.add(q)
            cols.add(pos[q][0]); rows.add(pos[q][1])
        admit.refused = []
        admit.force = force
        return admit

    def _park_riders(self, park_qs, deferred, admit=None, ends=None):
        """Deferred qubits whose FULL move can join this wave's Park move.

        A rider must be pairwise AOD-compatible with every Park vector and with
        every other rider, and its destination must be genuinely free — either
        never occupied, or vacated by this same Park (a chain, which is safe
        because the Park is one simultaneous transform). Riders complete in
        full, so they need no follow-up leg.
        """
        if not deferred:
            return []
        # ``ends``: where each held atom goes this leg (default: its final
        # trap). Under ZAP_HELD_CHAIN a held atom may stop at a waypoint, or
        # not move; its final trap stays reserved either way.
        end = (lambda q: tuple(self.final_mapping[q])) if ends is None else \
            (lambda q: tuple(ends.get(q, self.current_mapping[q])))
        park_vecs = []
        for q in park_qs:
            a, b = tuple(self.current_mapping[q]), end(q)
            if a != b:
                park_vecs.append((a[0], b[0], a[1], b[1]))
            elif ends is not None:
                park_vecs.append((a[0], a[0], a[1], a[1]))
        if not park_vecs:
            return []

        moving = {q for q in park_qs if tuple(self.current_mapping[q]) != end(q)}
        vacated = {tuple(self.current_mapping[q]) for q in moving}
        occupied = {tuple(p) for i, p in enumerate(self.current_mapping)
                    if i not in moving}
        taken = {tuple(self.final_mapping[q]) for q in park_qs} | {end(q) for q in park_qs}

        riders, rider_vecs = [], []
        for v, q in sorted(deferred, key=lambda t: -math.dist(
                (t[0][0], t[0][2]), (t[0][1], t[0][3]))):
            target = (v[1], v[3])
            if target in taken:
                continue
            if target in occupied and target not in vacated:
                continue
            if not all(self.compatible_2D(v, u) for u in park_vecs + rider_vecs):
                continue
            if not self._path_clear((v[0], v[2]), target,
                                    (occupied | taken) - vacated - {(v[0], v[2])}):
                continue
            if self.ghost_refuse and self._ghosts(riders + [q]):
                continue
            if admit is not None and not admit(q):
                continue
            riders.append(q)
            rider_vecs.append(v)
            taken.add(target)
        return riders

    def _free_park_target(self, q, anchor, axis, tmp_end_locs, aod_qubits,
                          tmp_begin_locs=None):
        """A displaced parking target near ``anchor``, offset along ``axis``.

        The collision-parking rule diverts a blocked move by a fixed
        ``+PARKING_DIST`` along one axis and takes the result unconditionally.
        That is unsafe: nothing checks whether the diverted point is already
        somebody else's destination this wave. It collides in particular with
        waypoints, which are held AOD positions rather than SLM traps and so sit
        exactly where a +1 nudge lands (observed on ising_n26: q24 diverted onto
        (13,70), the waypoint q19 had already been granted, putting two atoms on
        one point).

        Same mechanism, but the offset is searched instead of assumed: try
        +/-1, +/-2, ... multiples of PARKING_DIST and take the first point no
        other atom occupies or is heading to. Falls back to the original fixed
        nudge if nothing is free, so behaviour is unchanged where it was already
        safe.

        Occupancy is not the only thing a diversion can break. The wave's moves
        were checked pairwise with ``compatible_2D`` when the wave was chosen,
        but this function then REWRITES one of those moves' endpoints, and the
        rewritten vector was never re-checked against its wave-mates. That is
        where the router's emitted AOD-row/column crossings come from: they are
        not planning mistakes, they are validated plans invalidated at emission
        time. Measured on multiplier_n15, every crossing in the trace had a
        diverted (or waypointed) vector on at least one side of it — e.g. q1
        (0,52) -> (0,53) travelling beside q12 (28,52) -> (36,52), two atoms
        starting on row 52 and ending on different rows.

        So candidate offsets are now also screened against the rest of the wave,
        and a compatible one is preferred. ``tmp_begin_locs`` supplies the
        wave's true starting points (they differ from ``current_mapping`` for
        qubits that were pre-parked out of a column collision). When no offset
        is both free and compatible, the first merely-free one is used, so this
        never fails to produce a target.
        """
        if self.proxy_audit and self._audit is not None:
            self._audit['diversions'] += 1
        moving = set(aod_qubits)
        busy = set()
        for o in range(self.n_q):
            if o == q:
                continue
            busy.add(tuple(tmp_end_locs[o]) if o in moving
                     else tuple(self.current_mapping[o]))

        others = []
        if tmp_begin_locs is not None:
            for o in aod_qubits:
                if o == q:
                    continue
                bx, by = tmp_begin_locs[o]
                ex, ey = tmp_end_locs[o]
                if (bx, by) != (ex, ey):
                    others.append((bx, ex, by, ey))

        start = tmp_begin_locs[q] if tmp_begin_locs is not None else anchor
        first_free = None
        compat_only = None
        clear_only = None
        for step in range(1, 6):
            for sign in (1, -1):
                delta = sign * step * self.PARKING_DIST
                cand = ((anchor[0] + delta, anchor[1]) if axis == 0
                        else (anchor[0], anchor[1] + delta))
                if cand in busy:
                    continue
                # d_s: a diverted target that lands within MIN_TONE_SEP of a
                # wave-mate's tone is not merely suboptimal, it is unbuildable.
                # Screened BEFORE first_free so the "merely free" fallback can
                # never hand back an illegal tone either (that fallback was the
                # last source of sub-d_s gaps after the predicate itself was
                # fixed: 4 cases across tqe, all chained Park moves).
                if self._min_tone_sep and not self._tone_ok(cand, others, start):
                    continue
                if first_free is None:
                    first_free = cand
                v = (start[0], cand[0], start[1], cand[1])
                if not all(self.compatible_2D(v, u) for u in others):
                    continue
                if compat_only is None:
                    compat_only = cand
                # The whole point of this diversion is to route AROUND an
                # occupied trap, so the replacement path must itself be clear.
                # Anchoring at the destination makes this check load-bearing in
                # a way origin anchoring never did: the old target was one step
                # from the start, so its path was trivially short, whereas the
                # new one spans the entire journey and can cross other atoms.
                # Skipping it emitted 59 moves that fly an atom straight over an
                # occupied trap across the tqe suite.
                if self.verify_path(start, cand)[0]:
                    # Planning-time held-atom rule: prefer a diversion from
                    # which every atom of the wave can still be brought home.
                    trial = {o: tuple(tmp_end_locs[o]) for o in aod_qubits}
                    trial[q] = cand
                    finals = {o: tuple(self.final_mapping[o]) for o in aod_qubits}
                    if self._finishable(trial, finals):
                        return cand
                    if clear_only is None:
                        clear_only = cand
        # Prefer a wave-compatible target over a merely-free one, and a free one
        # over the blind fixed nudge; a path that is still blocked is caught by
        # the re-verification loop in process_rearrangement.
        if clear_only is not None:
            return clear_only
        if compat_only is not None:
            return compat_only
        if first_free is not None:
            return first_free
        return ((anchor[0] + self.PARKING_DIST, anchor[1]) if axis == 0
                else (anchor[0], anchor[1] + self.PARKING_DIST))

    def _tone_ok(self, cand, others, start):
        """Does landing at ``cand`` keep every tone gap >= MIN_TONE_SEP (d_s)?

        ``compatible_2D`` already applies d_s to the pairs it is given, but the
        diversion fallbacks in ``_free_park_target`` return candidates that were
        never put through it. This is the same test, applied to one candidate
        against the wave's other movers.
        """
        ds = self._min_tone_sep
        if not ds:
            return True
        v = (start[0], cand[0], start[1], cand[1])
        for u in others:
            if not self.compatible_2D(v, u):
                return False
        return True

    def verify_path(self, start, end, ignore=None):
        """
        Verifies the path between two points.

        Args:
            start (tuple): Starting point.
            end (tuple): Ending point.
            ignore (set): Positions to treat as clear even if occupied in
                ``self.current_mapping`` — used to exclude a wave-mate's
                pre-move position when that wave-mate is compatibly leaving
                the same wave, since it will have vacated the site by the
                time this qubit passes through it.

        Returns:
            bool: True if the path is clear, False otherwise.
        """
        ignore = ignore or ()
        x1, y1 = start
        x2, y2 = end

        if x1 == x2:
            for y in range(min(y1, y2) + 1, max(y1, y2)):
                if (x1, y) in self.current_mapping and (x1, y) not in ignore:
                    return False, "Vertical"

        elif y1 == y2:
            for x in range(min(x1, x2) + 1, max(x1, x2)):
                if (x, y1) in self.current_mapping and (x, y1) not in ignore:
                    return False, "Horizontal"

        else:
            dx = x2 - x1
            dy = y2 - y1
            gcd = abs(dx) if dy == 0 else abs(dy) if dx == 0 else abs(math.gcd(dx, dy))
            step_x = dx // gcd
            step_y = dy // gcd

            x, y = x1 + step_x, y1 + step_y
            while (x, y) != (x2, y2):
                if (x, y) in self.current_mapping and (x, y) not in ignore:
                    return False, "Diagonal"
                x += step_x
                y += step_y
        return True, "Clear"

    def find_chain_aware_independent_set(self, vectors, vacated=None):
        """The wave-selection algorithm: pick the largest safe, compatible batch.

        ABLATION (``ablate_chain_aware_mis``, default off): fall back to a
        degree-sorted greedy MIS over ``compatible_2D`` that admits only
        vectors whose target is unoccupied outright -- no dependency
        ordering, no chained admission. This is the pre-2026-09-13 "strict"
        candidate in isolation, and is the component this gate measures.

        SHIPPED 2026-09-13, replacing a two-candidate scheme
        (``find_maximal_independent_set(vectors, vacated=starts)`` — "chain",
        allowing a vector to land on a trap another selected vector is
        vacating, pruned to a fixpoint — compared against the same function
        restricted to vectors whose target wasn't anyone's current start at
        all, keeping whichever came out bigger). That scheme is deleted, not
        just superseded — recover it from git history if ever needed again.
        Measured reason for replacing it: chain lost to the simpler
        candidate on 216/799 tqe stages, every loss chain=0 vs strict>=1 —
        not a close call. Root cause was the fixpoint prune, not greedy-order
        luck: an admit-then-cascade-prune step can wipe out an entire
        dependency chain (if the vacater of A's target gets pruned, A must
        be pruned too, which can un-vacate what a third vector depended on,
        and so on), while the simpler candidate's non-cascading rule never
        had anything to cascade.

        FIRST ATTEMPT at a replacement (resolve each vector's feasibility
        from the dependency graph alone, independently of which vectors the
        MIS actually selects) was WRONG and had a real bug: a vector could
        be marked "feasible" because its dependency COULD be admitted
        somewhere, while the greedy MIS then admitted the vector but
        rejected its dependency for an unrelated compatible_2D conflict —
        landing an atom on a trap that never actually vacated. Measured:
        double-occupancy on 11/14 tqe circuits (verify_traces.py), one real
        AOD crossing on vqc_n15 (verify_crossings.py). Fixed by coupling
        admission to the dependency INSIDE the same greedy pass instead of
        resolving it beforehand: a vector is only added to the growing
        independent set at the moment its dependency (if any) is confirmed
        already in that same set, so nothing is ever admitted on the
        strength of a dependency that doesn't itself end up chosen.

        Measured on tqe (final tie-break version, both physical gates
        passing): +0.002693 total (6.950821 -> 6.953515), 8/14 circuits
        better, 0 regressed, 6 unchanged. Not independently re-verified on
        graphs before shipping — if graphs later shows a trade, that is new
        information to surface, not evidence this measurement was wrong.

        Algorithm: process vectors in dependency order — depth from a
        dependency-free (or already-admitted) vector, shallowest first, so a
        vector is only even considered once whatever it depends on has had
        its own chance to be admitted. Within a depth tier, order is by
        (a) longest downstream chain first — admitting the root of a long
        chain unlocks the most follow-on admissions, whereas admitting an
        unrelated depth-0 vector first can occupy a compatible_2D "slot" a
        chain root needed for no gain — then (b) fewest compatible_2D
        conflicts (degree-sorted, the same tie-break the deleted two-
        candidate scheme used internally), then (c) original index for
        full determinism. A vector is admitted iff
        it is compatible_2D with everything already admitted, AND its
        dependency (if any) is already in the admitted set — a real-time
        check against the actual admitted set, not a precomputed
        feasibility flag (the gap the first attempt got wrong).

        This is a greedy heuristic, not a proven maximum independent set —
        a vector admitted early can still block a larger combination that
        would have existed had a different tied vector gone first. The
        chain-length/degree tie-break narrows how often that happens but
        does not eliminate it; true optimality would need backtracking or
        branch-and-bound, which was deliberately not attempted here (this
        codebase's own history is that "more faithful" cost/search
        upgrades routinely measure worse or cost more than they're worth —
        see placer.md's `order_scoring`/MIS-matching ablations).

        Cycles (true swaps, A -> B's trap while B -> A's trap) are handled
        for free: ``compatible_2D`` already forbids a swap outright (it is a
        path inversion on every axis), so a cyclic pair can never both be
        admitted regardless of processing order — the same physical
        constraint that already rules out double-occupancy.
        """
        if self.ablate_chain_aware_mis:
            return self._ablation_strict_mis(vectors)
        occupied_by_other = set(self.current_mapping)
        vacated = set(vacated or ())

        # dep[i] = index of the vector (in `vectors`) currently occupying
        # vector i's target, or None if that target is free / not one of
        # this batch's own starts (i.e. genuinely infeasible this wave).
        start_index = {}
        for i, v in enumerate(vectors):
            start_index.setdefault((v[0], v[2]), i)

        dep = [None] * len(vectors)
        infeasible = [False] * len(vectors)
        for i, v in enumerate(vectors):
            target = (v[1], v[3])
            if target in start_index:
                j = start_index[target]
                if j != i:
                    dep[i] = j
            elif target in occupied_by_other and target not in vacated:
                infeasible[i] = True

        # Depth = distance to a dependency-free node, with cycle detection
        # (a dependency chain that loops back on itself gets depth None —
        # infeasible as a chain; compatible_2D would reject the underlying
        # swap anyway).
        depth = [None] * len(vectors)

        def compute_depth(i, trail):
            if depth[i] is not None:
                return depth[i]
            if infeasible[i] or i in trail:
                depth[i] = None
                return None
            j = dep[i]
            if j is None:
                depth[i] = 0
                return 0
            d = compute_depth(j, trail | {i})
            depth[i] = None if d is None else d + 1
            return depth[i]

        for i in range(len(vectors)):
            compute_depth(i, frozenset())

        # chain_len[i] = how many vectors (transitively) depend on i being
        # admitted. Admitting a vector with a long downstream chain first
        # unlocks the most follow-on admissions; admitting an isolated
        # depth-0 vector first (by arbitrary index luck) can instead occupy
        # a compatible_2D "slot" a chain root needed, silently truncating an
        # otherwise-admittable chain for no gain.
        chain_len = [0] * len(vectors)
        for i in range(len(vectors)):
            j = dep[i]
            while j is not None:
                chain_len[j] += 1
                j = dep[j]

        compatible = self.compatible_2D
        degree = [0] * len(vectors)
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                if not compatible(vectors[i], vectors[j]):
                    degree[i] += 1
                    degree[j] += 1

        # Within a depth tier: longest downstream chain first, then fewest
        # compatible_2D conflicts, then original index for full determinism.
        order = sorted(
            (i for i in range(len(vectors)) if depth[i] is not None),
            key=lambda i: (depth[i], -chain_len[i], degree[i], i))

        admitted = []          # list of vector tuples, in admission order
        admitted_idx = set()   # indices into `vectors`
        for i in order:
            if dep[i] is not None and dep[i] not in admitted_idx:
                continue
            v = vectors[i]
            if all(compatible(v, u) for u in admitted):
                admitted.append(v)
                admitted_idx.add(i)

        if not admitted:
            raise ValueError("No compatible vectors found.")
        return admitted

    def _ablation_strict_mis(self, vectors):
        """Degree-greedy MIS admitting only vectors with an unoccupied target.

        Measurement path for ``ablate_chain_aware_mis`` -- never used by the
        shipped compiler. Mirrors the contract of the function it replaces
        (returns a non-empty list of vectors, raises ValueError if none
        survive) so the wave loop is otherwise untouched.
        """
        occupied = {tuple(loc) for loc in self.current_mapping}
        starts = {(v[0], v[2]) for v in vectors}
        cand = [i for i, v in enumerate(vectors)
                if (v[1], v[3]) not in occupied or (v[1], v[3]) in starts]
        # Restrict to targets that are genuinely free right now (no chaining).
        cand = [i for i in cand if (vectors[i][1], vectors[i][3]) not in occupied]
        if not cand:
            cand = list(range(len(vectors)))
        conflict = {i: set() for i in cand}
        for a in range(len(cand)):
            for b in range(a + 1, len(cand)):
                i, j = cand[a], cand[b]
                if not self.compatible_2D(vectors[i], vectors[j]):
                    conflict[i].add(j); conflict[j].add(i)
        chosen, blocked = [], set()
        for node in sorted(conflict, key=lambda x: (len(conflict[x]), x)):
            if node not in blocked:
                chosen.append(node); blocked.update(conflict[node])
        if not chosen:
            raise ValueError("no vector could be selected")
        return [vectors[i] for i in chosen]

    def _mis_compat_only(self, vectors):
        """Degree-sorted greedy MIS over compatible_2D alone, no occupancy filter.

        Proxy-audit only, a separate small implementation (not a call into
        ``find_chain_aware_independent_set``). The placer's model packs a
        stage's vectors by ``compatible_2D`` alone; comparing this function's
        size against the wave actually executed isolates how often occupancy
        — not AOD compatibility — was the binding constraint. Returns indices
        into ``vectors``.
        """
        n = len(vectors)
        conflict = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if not self.compatible_2D(vectors[i], vectors[j]):
                    conflict[i].add(j)
                    conflict[j].add(i)
        chosen, visited = [], set()
        for node in sorted(conflict, key=lambda x: len(conflict[x])):
            if node not in visited:
                chosen.append(node)
                visited.update(conflict[node])
                visited.add(node)
        return chosen

    def _crossing_lower_bound(self, vectors):
        """Dilworth lower bound: size of the largest mutually-crossing vector set.

        Every pair in such a set violates ``compatible_2D``, so no two of them
        can share a wave and the stage provably needs at least this many waves,
        whatever the router does. That is a maximum clique in the conflict
        graph; stages here hold at most a few dozen vectors, so exact
        Bron-Kerbosch is affordable, with a size guard in case a much larger
        architecture ever runs this mode.
        """
        n = len(vectors)
        if n <= 1:
            return n
        if n > 40:
            return 0            # not attempted; reported as unavailable
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if not self.compatible_2D(vectors[i], vectors[j]):
                    g.add_edge(i, j)
        return max(len(c) for c in nx.find_cliques(g))

    # Stages above this many vectors are not solved exactly. The colouring
    # search is exponential in the worst case; 16 keeps the whole tqe audit
    # under a second while still covering the large majority of moving stages.
    EXACT_WAVE_MAX_VECTORS = 16

    def _exact_min_waves(self, vectors):
        """Exact minimum number of waves ``vectors`` can be packed into.

        A wave is a set of vectors that are *pairwise* ``compatible_2D`` — that
        is what both ``batch_moves`` and the router's own batcher require — so a
        wave is an independent set in the conflict graph and the minimum wave
        count is exactly that graph's chromatic number. ``_crossing_lower_bound``
        returns the clique number, which lower-bounds it; the two coincide only
        if the graph is perfect. On a single axis the conflict relation is a
        permutation graph (perfect), but in 2D the graph is the *union* of the
        x- and y-conflict graphs and a union of perfect graphs need not be
        perfect, so the bound is valid but not obviously tight. This measures
        the slack instead of assuming it away.

        Returns 0 when not attempted (too many vectors), matching the
        "unavailable" convention of ``_crossing_lower_bound``.
        """
        n = len(vectors)
        if n <= 1:
            return n
        if n > self.EXACT_WAVE_MAX_VECTORS:
            return 0
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if not self.compatible_2D(vectors[i], vectors[j]):
                    adj[i].add(j)
                    adj[j].add(i)
        # Most-constrained-first ordering makes the backtracking search fail
        # fast on infeasible k.
        order = sorted(range(n), key=lambda v: -len(adj[v]))

        def colourable(k):
            colour = {}

            def place(idx, used):
                if idx == n:
                    return True
                v = order[idx]
                taken = {colour[u] for u in adj[v] if u in colour}
                # Symmetry breaking: colours are interchangeable, so vertex idx
                # never needs a colour above the number already in use.
                for c in range(min(used + 1, k)):
                    if c in taken:
                        continue
                    colour[v] = c
                    if place(idx + 1, max(used, c + 1)):
                        return True
                    del colour[v]
                return False

            return place(0, 0)

        k = self._crossing_lower_bound(vectors) or 1
        while not colourable(k):
            k += 1
        return k

    def _exact_min_wave_time(self, vectors):
        """Exact minimum total move TIME for ``vectors``, over all wave splits.

        Wave *count* is not the quantity the simulator charges for: each emitted
        move instruction costs its own bottleneck duration, serially, and
        ``movement_duration`` is concave, so a stage's cost is the sum of its
        waves' longest legs. Minimising that is not the same problem as
        minimising the number of waves (an extra wave holding only short moves
        can be cheaper than packing them behind a long one), so it gets its own
        exact solve.

        Sorting descending makes the first vector placed in a wave that wave's
        bottleneck, so the objective is just the sum of the wave-openers, which
        branch-and-bound prunes on directly. Returns 0.0 when not attempted.
        """
        n = len(vectors)
        if n == 0:
            return 0.0
        if n > self.EXACT_WAVE_MAX_VECTORS:
            return 0.0
        idx = sorted(range(n), key=lambda i: -transport._duration(vectors[i]))
        durs = [transport._duration(vectors[i]) for i in idx]
        vs = [vectors[i] for i in idx]
        compat = [[self.compatible_2D(a, b) for b in vs] for a in vs]
        best = [sum(durs)]           # every vector in its own wave
        waves = []                   # list of member-index lists

        def search(k, cost):
            if cost >= best[0]:
                return
            if k == n:
                best[0] = cost
                return
            for w in waves:
                if all(compat[k][m] for m in w):
                    w.append(k)
                    search(k + 1, cost)
                    w.pop()
            waves.append([k])
            search(k + 1, cost + durs[k])
            waves.pop()

        search(0, 0.0)
        return best[0]

    def _record_proxy_audit(self, placer, n_before):
        """One per-stage audit row: predicted vs actual, plus mechanism counters."""
        fc = placer.proxy_stage_forecast()
        emitted = self.results_code['instructions'][n_before:]
        moves = [i for i in emitted if i['type'] in ('BigMove', 'Park')]
        transfers = [i for i in emitted if i['type'] in ('Activate', 'Deactivate')]
        vecs = self._audit['stage_vectors']
        row = dict(fc)
        row.update({
            'stage': self.index_list_gate,
            'actual_waves': len(moves),
            'actual_move_time': sum(max(i['duration']) for i in moves if i['duration']),
            'actual_transfer_time': sum(max(i['duration']) for i in transfers
                                        if i['duration']),
            'actual_duration': (sum(max(i['duration']) for i in moves if i['duration'])
                                + sum(max(i['duration']) for i in transfers
                                      if i['duration'])),
            'actual_n_tr': sum(len(i['qs']) for i in transfers),
            'n_bigmove': sum(1 for i in moves if i['type'] == 'BigMove'),
            'n_park': sum(1 for i in moves if i['type'] == 'Park'),
            'mis_waves': self._mis_waves(vecs),
            'lower_bound': self._crossing_lower_bound(vecs),
            'exact_waves': self._exact_min_waves(vecs),
            'exact_move_time': self._exact_min_wave_time(vecs),
            'bigmove_time': sum(max(i['duration']) for i in moves
                                if i['type'] == 'BigMove' and i['duration']),
            'park_time': sum(max(i['duration']) for i in moves
                             if i['type'] == 'Park' and i['duration']),
            'wave_iters': self._audit['iters'],
            'waypoints': self._audit['waypoints'],
            'riders': self._audit['riders'],
            'occ_limited_waves': self._audit['occ_limited'],
            'diversions': self._audit['diversions'],
            'prepark_rounds': self._audit['prepark_rounds'],
            'prepark_cols': self._audit['prepark_cols'],
            'prepark_zero_shift': self._audit['prepark_zero_shift'],
            'unchecked_emit': self._audit['unchecked_emit'],
            'diversion_rounds_exhausted': self._audit['diversion_rounds_exhausted'],
        })
        self.audit_stages.append(row)

    def _mis_waves(self, vectors):
        """Waves the ROUTER'S batching algorithm needs for ``vectors`` alone.

        Iterated degree-sorted MIS with no occupancy, waypoint, collision or
        Park-chain effects — i.e. the placer's prediction recomputed with the
        router's packing rule instead of first-fit. The gap between this and
        ``pred_waves`` is pure batching-algorithm mismatch; the gap between
        this and ``actual_waves`` is everything else the router does.
        """
        remaining = list(vectors)
        waves = 0
        while remaining:
            picked = set(self._mis_compat_only(remaining))
            if not picked:
                break
            waves += 1
            remaining = [v for i, v in enumerate(remaining) if i not in picked]
        return waves

    def route_qubit(self):
        """Greedy maximal compatible move batches between scheduled gate layers."""
        placer = Placer(
            self.slm_sites,
            self.results_code,
            self.list_full_gates,
            self.qubit_mapping,
            self.architecture
        )
        self.current_mapping = list(placer.current_mapping)
        self._line_atoms = None
        self._held = set()          # atoms currently in the AOD
        self.results_code['held_crossings'] = 0
        self.write_init_instruction()

        for index_list_gate, list_gate in enumerate(
                tqdm(self.list_full_gates, desc="\t\tProcessing")):
            self.index_list_gate = index_list_gate
            self.list_gate = list_gate

            self.final_mapping = placer.placing(self.index_list_gate)
            n_before = len(self.results_code['instructions'])
            if self._pick_stage:
                self._route_stage_pick()
            else:
                self._route_stage_moves()
            if self.proxy_audit:
                self._record_proxy_audit(placer, n_before)

            if self.collect_stats:
                self.stats['waves_per_stage'].append(self._stage_wave_count)
                self._stage_wave_count = 0

            self.process_gate()

    def _stage_cost(self, instructions):
        """This stage's routing cost in nats, exactly as the Simulator charges
        it: decoherence (n_q x wall time) / T2,
        plus one atom-transfer loss per lifted or released atom."""
        T, transfers = 0.0, 0
        for ins in instructions:
            d = ins.get('duration') or [0]
            T += max(d)
            if ins['type'] in ("Activate", "Deactivate"):
                transfers += len(ins['qs'])
        f_t = self.architecture.get('operation_fidelity', {}).get('atom_transfer', 0.999)
        t2 = self.architecture.get('qubit_spec', {}).get('T2', 1.5e6)
        return self.n_q * T / t2 - transfers * math.log(f_t)

    def _router_state(self):
        """Everything routing a stage mutates, for _route_stage_pick."""
        import copy
        # Deep: routing mutates nested state in place (current_mapping's
        # entries, _occ_lines' sets); a shallow copy shares them and the
        # restore silently keeps a discarded variant's positions.
        st = {k: (copy.deepcopy(v) if isinstance(v, (list, set, dict)) else v)
              for k, v in self.__dict__.items()
              if k != 'results_code' and k not in self._STATIC_ATTRS}
        # 'stages' (MBs on wide circuits) is the scheduler's output, read by
        # the placer and never written while routing: share it, don't copy.
        rc = {k: copy.deepcopy(v) for k, v in self.results_code.items()
              if k not in ('instructions', 'stages')}
        return st, rc, len(self.results_code['instructions'])

    # Set before routing starts and never mutated by it (checked by diffing
    # the router's __dict__ around _route_stage_moves on qft_n100).
    _STATIC_ATTRS = frozenset({'list_full_gates', '_slm_set', '_ent_site_set',
                               'slm_sites', 'architecture'})

    def _restore_state(self, snap, instructions=None):
        import copy
        st, rc, n = snap
        # Copy again on the way in: a snapshot is restored more than once
        # (before each variant), so handing the router the snapshot's own
        # objects let one variant's routing mutate the start state the next
        # variant was restored from.
        rc = copy.deepcopy(rc)
        self.__dict__.update({k: (copy.deepcopy(v) if isinstance(v, (list, set, dict)) else v)
                              for k, v in st.items()})
        for k in [k for k in self.__dict__ if k not in st and k != 'results_code'
                  and k not in self._STATIC_ATTRS]:
            del self.__dict__[k]
        ins = self.results_code['instructions']
        del ins[n:]
        if instructions:
            ins.extend(instructions)
        for k in [k for k in self.results_code if k not in ('instructions', 'stages')]:
            del self.results_code[k]
        self.results_code.update(rc)
        self._line_atoms = None

    def _route_stage_pick(self):
        """Route this stage under up to three policies and keep the cheapest
        (ZAP_HELD_CHAIN=pick, 2026-10-03):

          off   the shipped router;
          leg   held atoms whose remainders conflict chain inside a leg (one
                finishes, the other banks a waypoint: _held_leg_plan);
          wave  leg, plus wave-level banking of prefixes whose remainders
                conflict (Fig. 4b).

        Every stage ends in the same state under all three -- every atom on
        the placer's final mapping, the AOD empty -- so stages are independent
        and keeping the cheapest per stage, by the Simulator's own cost
        (``_stage_cost``), cannot lose fidelity to any single policy. A policy
        is only tried when the previous run hit the case it changes (a leg
        with conflicting held atoms, a waypoint rejected for its remainder),
        so most stages are routed once.
        """
        start = self._router_state()
        n0 = start[2]
        best = None
        for name, chain, relax in (("off", False, False), ("leg", True, False),
                                   ("wave", True, True)):
            if best is not None:
                if name == "leg" and not self._held_conflict:
                    continue
                if name == "wave" and not self._remainder_rejected:
                    continue
                self._restore_state(start)
            self._held_chain, self._wave_relax = chain, relax
            if name == "off":
                self._held_conflict = False
                self._remainder_rejected = False
            flags = (self._held_conflict, self._remainder_rejected)
            self._route_stage_moves()
            hc = self._held_conflict or flags[0]
            rr = self._remainder_rejected or flags[1]
            ins = list(self.results_code['instructions'][n0:])
            cost = self._stage_cost(ins)
            force = os.environ.get("ZAP_PICK_FORCE")   # test hook: keep this variant
            if best is None or (cost < best[0] if not force else name == force):
                best = (cost, name, self._router_state(), ins)
            self._held_conflict, self._remainder_rejected = hc, rr
        self._restore_state(best[2])
        ins = self.results_code['instructions']
        del ins[n0:]
        ins.extend(best[3])
        self._held_chain = self._wave_relax = False
        st = self.results_code.setdefault('ghost_stats', {})
        st['pick_' + best[1]] = st.get('pick_' + best[1], 0) + 1

    def _route_stage_moves(self):
        """Emit every wave needed to reach ``self.final_mapping`` this stage."""
        vectors, aod_qubits = [], []
        for q in range(self.n_q):
            if self.current_mapping[q] != self.final_mapping[q]:
                aod_qubits.append(q)
                vectors.append(
                    (self.current_mapping[q][0], self.final_mapping[q][0],
                     self.current_mapping[q][1], self.final_mapping[q][1]))
        if self.proxy_audit:
            self._audit = {'stage_vectors': list(vectors), 'iters': 0,
                           'waypoints': 0, 'riders': 0, 'occ_limited': 0,
                           'diversions': 0, 'prepark_rounds': 0,
                           'prepark_cols': 0, 'prepark_zero_shift': 0,
                           'unchecked_emit': 0,
                           'diversion_rounds_exhausted': 0}
        while vectors:
            # See find_chain_aware_independent_set's docstring for the full
            # algorithm and the measurement that replaced the old
            # chain-vs-strict comparison (removed 2026-09-13; recover from
            # git history if needed) with this single deterministic pass.
            start_positions = set((v[0], v[2]) for v in vectors)
            safe_vectors = [v for v in vectors if (v[1], v[3]) not in start_positions]
            execute_vectors = self.find_chain_aware_independent_set(
                vectors, vacated=start_positions)
            if self.proxy_audit:
                self._audit['iters'] += 1
                # Category (a): was this wave bounded by AOD compatibility
                # (what the proxy models) or by trap occupancy (what it does
                # not)? The proxy packs the stage's vectors purely by
                # compatible_2D, so any wave the router had to make smaller
                # than the compatibility-only independent set was shrunk by
                # something the proxy cannot see.
                if len(execute_vectors) < len(self._mis_compat_only(vectors)):
                    self._audit['occ_limited'] += 1
            execute_aod_qubits = []
            for execute_vector in execute_vectors:
                execute_idx = vectors.index(execute_vector)
                execute_aod_qubits.append(aod_qubits[execute_idx])

            # Waypoint decomposition: deferred vectors bank their compatible
            # prefix in this wave and finish via the Park phase, instead of
            # waiting whole for a later wave.
            waypoints = self.plan_waypoints(
                vectors, aod_qubits, execute_vectors, safe_vectors)

            deferred = [(v, q) for v, q in zip(vectors, aod_qubits)
                        if q not in set(execute_aod_qubits) | set(waypoints)]
            riders = self.process_rearrangement(
                execute_aod_qubits, waypoints, deferred) or set()
            if self.proxy_audit:
                self._audit['waypoints'] += len(waypoints)
                self._audit['riders'] += len(riders)

            done = set(execute_aod_qubits) | set(waypoints) | riders
            done -= getattr(self, '_wave_dropped', set()) - set(riders)
            # Progress guarantee (2026-10-02): a wave that dropped every atom
            # it was given would be planned again unchanged, forever
            # (ising_n300 hung). The next wave then may not drop anything.
            # Safety net (2026-10-30): an atom counts as done only if it sits on
            # its final trap. A waypoint finished by the Park phase could be
            # left short, on another atom's target (ising_n200 after a foreign
            # layout at stage 61: atom 86 left on q154's target at stage 77);
            # dropped from the list, it was never moved again and the stage
            # deadlocked ("No compatible vectors found"). Such an atom is kept,
            # its vector restarted from where it stands.
            stranded = {q for q in done
                        if tuple(self.current_mapping[q]) != tuple(self.final_mapping[q])}
            if stranded:
                done -= stranded
                st = self.results_code.setdefault('ghost_stats', {})
                st['stranded'] = st.get('stranded', 0) + len(stranded)
            self._no_wave_drop = not done
            remaining = [(v, q) for v, q in zip(vectors, aod_qubits) if q not in done]
            vectors = [(self.current_mapping[q][0], self.final_mapping[q][0],
                        self.current_mapping[q][1], self.final_mapping[q][1])
                       if q in stranded else v for v, q in remaining]
            aod_qubits = [q for _, q in remaining]

