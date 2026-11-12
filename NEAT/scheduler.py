import os
class Scheduler:
    """Build per-stage gate groups from a flat gate list ``g_q`` (indices into ``g_q``)."""

    def __init__(self,
                 g_q: list,
                 results_code: dict,
                 ):
        """
        Args:
            g_q: Flat list of gates as ``(q0, q1)`` with ``q0==q1`` for 1-qubit gates.
            results_code: Mutated in place; ``stages`` is filled by ``save_results``.
        """
        self.g_q = g_q
        self.results_code = results_code
        self.results_code['stages'] = {
            "num_stage": 0,
            "stage": {q: {} for q in range(self.results_code['n_q'])},
            "qs_status": {}
        }
        self.list_scheduling = []

    # REMOVED 2026-09-13: asap_joint, asap_separate, and mid_separate.
    # NEAT only ever ships alap_separate (NEAT/run.py's SCHEDULING
    # picks it for NEAT specifically); the other three were reachable
    # only via an explicit --scheduling_strategy CLI override, kept as
    # documented, MEASURED comparison points, not accidental leftovers:
    #   - asap_separate: 6.8567/6.8606 vs alap_separate's 6.9050 under the old
    #     5-config autotune grid — won no circuit outright. This is why
    #     alap_separate was chosen as the default in the first place.
    #   - asap_joint: character-for-character identical to baseline
    #     baselines/zap/scheduler.py's own asap_joint (confirmed by direct
    #     diff); never separately measured against alap_separate under the
    #     fixed-config regime, since asap_separate already lost decisively.
    #   - mid_separate(frac): interpolates each 2q gate's layer between its
    #     ASAP and ALAP position. frac=1.0 reproduces alap_separate bit-for-
    #     bit (sanity-checked). Swept frac in {0.6, 0.7, 0.8, 0.9} against tqe
    #     (E1, 2026-09-02): all four measured SUBSTANTIALLY worse than
    #     alap_separate's 6.911217 floor at the time — 0.6: 6.821140 (-0.090),
    #     0.7: 6.818024 (-0.093), 0.8: 6.823799 (-0.087), 0.9: 6.838976
    #     (-0.072) — 5-7 circuits regressed at every frac, 0 improved
    #     anywhere; an order of magnitude above the ~0.0022 noise floor, so a
    #     real effect: frac does not interpolate toward a better minimum
    #     between ASAP and ALAP on this suite.
    # Recover any of the three from git history if a similar scheduling
    # comparison is needed again.

    def _asap_separate_ablation(self):
        """ASAP 2q layering -- the BASELINE zap scheduler, for ablation only.

        Ported verbatim from ``baselines/zap/scheduler.py`` so the
        "scheduler block" ablation replaces NEAT's ALAP schedule with
        the baseline's ASAP one rather than with something invented for the
        experiment. Reached only via ZAP_ABLATE_SCHEDULER=1; never from any
        shipped configuration. ALAP measured 6.9050 vs ASAP 6.8567/6.8606 on
        the old autotune grid, which is the gap this ablation re-measures.
        """
        list_qubit_stage = [0 for _ in range(self.results_code['n_q'])]
        two_qubit_gates, _single = [], []
        for i, gate in enumerate(self.g_q):
            (_single if gate[0] == gate[1] else two_qubit_gates).append((i, gate))
        for i, gate in two_qubit_gates:
            stage = max(list_qubit_stage[gate[0]], list_qubit_stage[gate[1]])
            if stage >= len(self.list_scheduling):
                self.list_scheduling.append([])
            self.list_scheduling[stage].append(i)
            list_qubit_stage[gate[0]] = list_qubit_stage[gate[1]] = stage + 1
        self._place_1q_gates()
        self.save_results()

    def alap_separate(self):
        """Schedule 2q gates as late as possible, then place 1q gates around them.

        Walks the 2q gates in reverse to compute a reverse-time layering
        (same recurrence an ASAP forward sweep uses, run backwards), then
        flips it to forward time order. The total number of 2q layers is
        bounded by the circuit's critical path either way (an ASAP-style
        sweep would produce the same count, though this file no longer
        implements one to compare against directly — see this method's
        removal note above), but each qubit's 2q gates are pushed toward the
        end of the circuit. This shifts where the idle windows fall, which
        can change decoherence / handover fidelity. 1q gates are placed by
        the shared `_place_1q_gates()` once the 2q layers are fixed.
        """
        if os.environ.get("ZAP_ABLATE_SCHEDULER"):
            return self._asap_separate_ablation()
        # 1q gates are placed later by _place_1q_gates, which reads self.g_q
        # directly — no need to collect them here, only the 2q gates that
        # this method actually lays out.
        two_qubit_gates = [(i, gate) for i, gate in enumerate(self.g_q)
                           if gate[0] != gate[1]]

        # --- Schedule 2q gates ALAP -------------------------------------------
        # Walk the 2q gates in reverse. ``rev_stage[q]`` is the depth (counted
        # from the END of the circuit) at which qubit q is next used. Each gate
        # takes the max depth of its two qubits, giving a reversed layering that
        # we flip at the end so layer 0 is the earliest in time.
        rev_layers = []  # list of lists of gate indices, indexed from the end
        rev_stage = [0 for _ in range(self.results_code['n_q'])]
        for i, gate in reversed(two_qubit_gates):
            stage = max(rev_stage[gate[0]], rev_stage[gate[1]])
            if stage >= len(rev_layers):
                rev_layers.append([])
            rev_layers[stage].append(i)
            stage += 1
            rev_stage[gate[0]] = stage
            rev_stage[gate[1]] = stage

        # Flip to forward time order; reverse each layer's contents so that
        # gates land in their original relative order within a layer.
        self.list_scheduling = [list(reversed(layer)) for layer in reversed(rev_layers)]

        self._place_1q_gates()
        self.save_results()

    def _place_1q_gates(self):
        """Insert the 1q gates around already-fixed 2q layers, in place.

        Only ``alap_separate`` calls this now — ``asap_separate`` and
        ``mid_separate``, which used to share it, were removed 2026-09-13
        (see the comment above ``alap_separate``).

        Default since 2026-10-07: ``stab_late`` (fewest 1q stages, see
        ``_place_1q_gates_window``). The code below is the previous rule
        (each 1q gate right after its qubit's previous 2q gate), kept as
        ``ZAP_1Q_PLACE=early``. Measured 2026-10-07 (experiments/data/
        1q_placement): tqe 6.786142 -> 6.792155 (9 better, knn_n25 -0.0027
        worse, accepted as a trade); 18 scalability circuits at n~100/300/
        500, summed -log10 F -0.99% (7 better / 2 worse by ~0.01%); graphs,
        ising and random identical (no 1q slack). A 1q stage costs 52 us
        whatever it holds and moves no atom, so its count is the only cost.
        """
        mode = os.environ.get("ZAP_1Q_PLACE", "stab_late")
        if mode != "early":
            return self._place_1q_gates_window(mode)
        # F3 speedup (2026-09-02): gate_stage[i] gives gate index i's CURRENT
        # stage, kept live instead of rebuilding a flattened index list (and
        # rescanning it twice more via next(...)) on every gate of the O(G)
        # main loop below — that was O(G^2) total. "Current" is the operative
        # word: the original next(...) scan reads self.list_scheduling fresh
        # every call, so it is never stale — it always finds i's true
        # present position, including after an insert() has shifted later
        # stages. gate_stage has to be re-based the same way, at the same
        # point insert() does the shifting, or it silently drifts.
        #
        # An earlier version of this fix built gate_stage ONCE before the
        # loop and never touched it again; that measured -0.028 on tqe
        # (qft_n10 alone -0.0037) because later iterations' appends/inserts
        # were invisible to it. This version updates it at both mutation
        # sites below (list append and list insert, the latter incrementing
        # every already-recorded stage >= the insertion point) and is
        # verified bit-identical to the original O(G^2) scan on tqe.
        #
        # This is NOT the same thing as F3's separately tracked staleness
        # bug: that one is about prev_2q_stage's *stored* values (a qubit's
        # last-seen 2q stage) not being re-based after an insert — a
        # different dict with different semantics. Fixing gate_stage's
        # liveness here does not touch that bug either way.
        # Speedup (2026-09-03): the two dict rebuilds below were the compiler's
        # remaining O(G^2) term — every insert walked all of gate_stage to add 1
        # to the entries at or past the insertion point. On qft_n150 that is 594
        # inserts over a dict growing to 100k entries: 29.0M entry-rebuilds, of
        # which only 6.6M (23%) actually changed value. Worse, inserts arrive at
        # increasing stages, so the late ones shift 1-6 entries while still
        # copying a 100k dict.
        #
        # gate_stage is now derived from stage_gates (stage index -> the gate
        # indices recorded there) rather than stored per gate, so an insert
        # shifts only the stages at or past the cut instead of every gate. It is
        # still fully LIVE — updated at the same two mutation sites, so the
        # -0.028 trap described below (a gate_stage built once and never
        # updated) does not apply: nothing here is cached across mutations.
        # Verified bit-identical: the exact list_scheduling for every circuit of
        # tqe, tqe2 and qft_n150 hashes the same as before this change.
        stage_gates = [list(sublist) for sublist in self.list_scheduling]
        gate_stage = {}
        for stage_idx, sublist in enumerate(stage_gates):
            for idx in sublist:
                gate_stage[idx] = stage_idx

        prev_2q_stage = {}
        for i, gate in enumerate(self.g_q):
            if i in gate_stage:
                prev_2q_stage[gate[0]] = gate_stage[i]
                prev_2q_stage[gate[1]] = gate_stage[i]
            else:
                stage = max(prev_2q_stage.get(gate[0], -1), prev_2q_stage.get(gate[1], -1)) + 1

                while stage >= len(self.list_scheduling):
                    self.list_scheduling.append([])
                    stage_gates.append([])

                existing_two_qubit_gates = any(
                    self.g_q[j][0] != self.g_q[j][1] for j in self.list_scheduling[stage]
                )

                if existing_two_qubit_gates:
                    self.list_scheduling.insert(stage, [i])
                    # Re-base only the gates actually at or past the cut, found
                    # through stage_gates instead of by scanning every entry.
                    for s in range(stage, len(stage_gates)):
                        for idx in stage_gates[s]:
                            gate_stage[idx] = s + 1
                    stage_gates.insert(stage, [i])
                    gate_stage[i] = stage
                    # F3 correctness fix (2026-09-02): prev_2q_stage stores
                    # ABSOLUTE stage indices (see the assignment above), so it
                    # needs the same +1 re-basing gate_stage just got, or a
                    # qubit's recorded "last 2q stage" silently drifts one
                    # short of its true (post-insert) position .
                    # Self-heals for any qubit
                    # with a LATER 2q gate (its entry gets overwritten with a
                    # fresh, correct value from gate_stage); only a qubit
                    # whose last 2q gate precedes this insert keeps the stale
                    # value, placing its next 1q gate one stage too early.
                    prev_2q_stage = {q: (s + 1 if s >= stage else s)
                                     for q, s in prev_2q_stage.items()}
                else:
                    self.list_scheduling[stage].append(i)
                    stage_gates[stage].append(i)
                    gate_stage[i] = stage

    def _place_1q_gates_window(self, mode):
        """1q placement on fixed 2q layers (ZAP_1Q_PLACE; default stab_late).

        Slot s (0..K) is the point just before 2q layer s (K = after the
        last). A 1q gate on q may go in any slot of [a+1, b], where a is q's
        previous 2q layer (-1 if none) and b its next (K if none). Every
        nonempty slot becomes one 1q stage, so the stage count is the number
        of distinct slots used.
          early_generic : slot a+1 (reproduces ZAP_1Q_PLACE=early exactly)
          late          : slot b
          stab_late     : fewest slots (greedy by right end), each gate in
                          the latest chosen slot of its window
          stab_early    : fewest slots (greedy by left end), each gate in
                          the earliest chosen slot of its window
        """
        layers = self.list_scheduling
        K = len(layers)
        layer_of = {}
        for s, layer in enumerate(layers):
            for i in layer:
                layer_of[i] = s
        n_q = self.results_code['n_q']
        nxt = [K] * n_q
        win = {}
        for i in reversed(range(len(self.g_q))):
            q0, q1 = self.g_q[i]
            if q0 != q1:
                nxt[q0] = nxt[q1] = layer_of[i]
            else:
                win[i] = [None, nxt[q0]]
        prev = [-1] * n_q
        for i, (q0, q1) in enumerate(self.g_q):
            if q0 != q1:
                prev[q0] = prev[q1] = layer_of[i]
            else:
                win[i][0] = prev[q0] + 1

        if mode == "early_generic":
            slot = {i: lo for i, (lo, hi) in win.items()}
        elif mode == "late":
            slot = {i: hi for i, (lo, hi) in win.items()}
        elif mode in ("stab_late", "stab_early"):
            pts = []
            if mode == "stab_late":
                for lo, hi in sorted(win.values(), key=lambda w: w[1]):
                    if not pts or pts[-1] < lo:
                        pts.append(hi)
            else:
                for lo, hi in sorted(win.values(), key=lambda w: -w[0]):
                    if not pts or pts[-1] > hi:
                        pts.append(lo)
                pts.reverse()
            import bisect
            slot = {}
            for i, (lo, hi) in win.items():
                if mode == "stab_late":
                    slot[i] = pts[bisect.bisect_right(pts, hi) - 1]
                else:
                    slot[i] = pts[bisect.bisect_left(pts, lo)]
        else:
            raise ValueError(f"unknown ZAP_1Q_PLACE {mode!r}")

        by_slot = {}
        for i in sorted(slot):
            by_slot.setdefault(slot[i], []).append(i)
        out = []
        for s in range(K + 1):
            if s in by_slot:
                out.append(by_slot[s])
            if s < K:
                out.append(list(layers[s]))
        self.list_scheduling = out

    def save_results(self):
        """Write ``list_scheduling`` into ``results_code['stages']`` for the placer/router."""
        stage_dict = {}
        for stage, gates in enumerate(self.list_scheduling):
            if all(self.g_q[gate][0] == self.g_q[gate][1] for gate in gates):
                stage_type = "1qGate"
            elif all(self.g_q[gate][0] != self.g_q[gate][1] for gate in gates):
                stage_type = "2qGate"
            else:
                stage_type = "mGate"
            stage_dict[stage] = {
                'type': stage_type,
                'idx': gates,
                'gates': [self.g_q[gate] for gate in gates]
            }

        # Every entry references one of three shared, read-only status dicts
        # rather than owning a fresh {"stage", "status"} dict (2026-10-11):
        # n_q * n_stages dicts (8M on qft_n1000) are gone, ~10% less compile
        # CPU on qft_n700, output bit-identical. (Not the cause of the QFT
        # compile-time spikes: that was the QASM 3 parse tree's cyclic
        # garbage being collected inside the timed window -- see
        # tools/regen/worker.py Timer.) Readers only ever use ['status']
        # (incl. zap's placer under ZAP_ABLATE_PLACER); the dropped "stage"
        # key duplicated the list index and was never read. Never mutate an
        # entry: it is shared by every (qubit, stage) with that status.
        status = {s: {"status": s} for s in (None, "1qGate", "2qGate")}
        qs_status = {q: [status[None]] * len(self.list_scheduling) for q in range(self.results_code['n_q'])}
        for stage, gates in enumerate(self.list_scheduling):
            for gate in gates:
                q0, q1 = self.g_q[gate]
                if q0 == q1:
                    qs_status[q0][stage] = status["1qGate"]
                else:
                    qs_status[q0][stage] = status["2qGate"]
                    qs_status[q1][stage] = status["2qGate"]

        self.results_code['stages']['qs_status'] = qs_status
        self.results_code['stages']['num_stage'] = len(self.list_scheduling)
        self.results_code['stages']['stage'] = stage_dict
