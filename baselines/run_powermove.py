"""Compile a benchmark suite with PowerMove and re-score it with ZAP's model.

NOTE: this standalone suite runner scores with the compiler's own durations, so its
numbers differ from the paper's by a few percent. The paper compiles through
tools/regen/worker.py (which imports this module) and rescores every trace with
tools/regen/rescore_all.py.

Run from the repository root:

    python3 baselines/run_powermove.py [setting]      # default: qasmbench

Writes ``results/powermove/<setting>/log/<circuit>.json`` in ZAP's log schema for
every circuit PowerMove manages to compile, and prints an explicit failure line
for every one it does not.

Read this before trusting a number it prints
--------------------------------------------
PowerMove is research code with no README, no pinned dependencies and no generic
entry point; the public interface is ``mvqc(cz_blocks, Row, n, storage_flag)``,
reconstructed from ``test_scripts/``. Two consequences:

1. **It crashes on many circuits, including its own benchmarks.** ``mvqc.py:336``
   raises ``ValueError: list.remove(x): x not in list``. This reproduces on
   ``qft/qft_n18.qasm`` shipped in their own repo, so it is not an artefact of
   feeding it the tqe suite — most likely their block-construction assumes a
   gate layout that a newer qiskit's ``optimization_level=2`` no longer emits.
   ``storage_flag=True`` — the zoned mode that is the paper's contribution —
   fails more often than the no-storage fallback, so a suite-wide average would
   silently be a mix of two different algorithms. This script therefore records
   the mode per circuit and never averages across modes.

2. **Its fidelity model omits terms and uses its own constants**, so its native
   number is not comparable with ZAP's or ZAC's. Specifically: ``cir_fidelity_1q_gate``
   is initialised to 1 and never updated (single-qubit gate error is simply not
   charged), atom transfers are charged one factor per moved atom where ZAP and
   ZAC both charge two (pickup and drop), and coherence uses the linear
   ``1 - t/T2``. It also computes its 2Q-gate and transfer fidelity products with
   its own hardcoded constants (``Fidelity_2Q_Gate = 0.995``, ``Fidelity_Atom_Transfer
   = 0.999`` in ``mvqc.py``) rather than reading ``baselines/zap/architectures/default.json`` —
   these currently equal ZAP's own values, but nothing keeps them in sync. This
   module recomputes 1Q-gate, 2Q-gate, crosstalk, and transfer fidelity entirely
   from raw counts (``mvqc.py`` now also exports ``LAST_N_CROSSTALK`` and
   ``LAST_N_TRANSFERS`` alongside the busy-time export) against ZAP's own
   architecture constants, so none of PowerMove's own fidelity constants ever
   reach the reported number.

3. **Its decoherence meant something different**, and this is now corrected at
   the source rather than caveated. ZAP charges every microsecond a qubit is not
   busy (``total_duration - busy_time``); mvqc accumulated only the movement
   time a qubit spent outside storage. That is a strictly smaller quantity, and
   it grows with OTHER atoms' movement rather than with the schedule's length,
   so PowerMove could report less decoherence on a schedule that ran 63% longer
   (ghz_n30). ``mvqc.py`` now also exports per-qubit BUSY time, and
   ``score_powermove`` rebuilds the term ZAP's way, with the wall clock
   including the 1q-gate layers mvqc charges no time for (see
   ``count_1q_layers``).

What remains uncontrolled: the ARCHITECTURE. PowerMove builds its own grid sized
to the circuit (``Row = ceil(sqrt(n))``, X_SEP 19 / Y_SEP 15), so it is not
running on ``baselines/zap/architectures/default.json`` at all. Fidelity differences here are
tool-on-its-own-hardware, not algorithm-on-equal-hardware.
"""

import contextlib
import io
import json
import math
import os
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 2026-10-13 : PowerMove's router/placer/scheduler from the
# original copy (baselines/powermove/mvqc.py) and its gate blocking from
# upstream (baselines/powermove/Construct_Circuit_upstream.py).
#  * The original get_cz_blocks grouped CZ gates across non-diagonal 1q gates,
#    i.e. executed a different circuit on most non-CZ-only benchmarks
#    (2026-10-11); upstream's front-layer blocking respects the dependencies,
#    and visits the front layer in DAG node order (repo fix, 2026-10-13) so
#    PowerMove is reproducible.
#  * Upstream's mvqc differs from the original only in "CZ throttling", which
#    enumerates every simple cycle of each stage's move-dependency graph:
#    10^3 to >2.6*10^6 cycles on same-sized stages, hours per circuit and
#    unbounded memory at n >= 850 (random_n950 5482 s; graphs_550_4 > 3 h).
#    The original mvqc has no throttling and is used as is.
PM_DIR = os.path.join(ROOT, "baselines", "powermove")
PM_BLOCKS = os.path.join(ROOT, "baselines", "powermove", "Construct_Circuit_upstream.py")


def _load_pm():
    sys.path.insert(0, PM_DIR)
    os.chdir(PM_DIR)
    from qiskit import QuantumCircuit, transpile
    import importlib.util
    spec = importlib.util.spec_from_file_location("pm_upstream_blocks", PM_BLOCKS)
    blocks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(blocks)
    get_cz_blocks = blocks.get_cz_blocks
    import mvqc as PM
    return QuantumCircuit, transpile, get_cz_blocks, PM


def count_1q_layers(circuit):
    """Number of single-qubit layers, using PowerMove's own layering rule.

    ``get_cz_blocks`` splits the DAG into alternating CZ and 1q blocks and then
    throws the 1q ones away, because mvqc charges no time for them. ZAP does
    charge them — one ``time_1q_gate`` per LAYER, executed in parallel across
    qubits — and that time is a large part of its wall clock (qft_n10: ~35
    layers x 52 us). Leaving it out of PowerMove's timeline would hand it a
    shorter circuit for free, so the same decomposition is replayed here purely
    to count the discarded layers.
    """
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
            types.append(cur); blocks.append(set())
        added = 0
        for op in list(front):
            if kind(op) != cur:
                continue
            args = [q._index for q in op.qargs]
            if cur != 'CZ' and args[0] in blocks[-1]:
                continue
            added += 1
            blocks[-1].add(tuple(args) if len(args) > 1 else args[0])
            succ = list(dag.successors(op))
            dag.remove_op_node(op); front.discard(op)
            for s in succ:
                if not isinstance(s, DAGOutNode) and not list(dag.op_predecessors(s)):
                    front.add(s)
        if not added:
            cur = kind(next(iter(front)))
            types.append(cur); blocks.append(set())
    return sum(1 for t in types if t != 'CZ')


def score_powermove(out, busy_times, n_1q, n_1q_layers, n_2q_stages, n_2q,
                     n_crosstalk, n_transfers, arch):
    """Apply ZAP's fidelity formulas to one PowerMove run.

    ``out`` is mvqc's return tuple; ``busy_times`` is the per-qubit busy time it
    exports via ``LAST_QUBIT_BUSY_TIME``. ``n_2q``/``n_crosstalk``/``n_transfers``
    are raw counts (LAST_N_CROSSTALK / LAST_N_TRANSFERS, plus the externally
    computed circuit gate count) — deliberately NOT mvqc's own native fidelity
    products, which are computed with mvqc's own hardcoded Fidelity_2Q_Gate/
    Fidelity_Atom_Transfer constants (0.995/0.999 in ``baselines/powermove/
    mvqc.py``). Those constants currently equal ZAP's own, but nothing ties
    them together — recomputing from counts instead means this stays correct
    even if ``baselines/zap/architectures/default.json``'s fidelity constants ever change.
    """
    fid = arch["operation_fidelity"]
    dur = arch["operation_duration"]
    f_1q = fid["single_qubit_gate"]
    f_2q = fid["two_qubit_gate"]
    f_idle = fid.get("two_qubit_gate_for_idle", 1 - (1 - f_2q) / 2)
    f_tr = fid["atom_transfer"]
    t2 = arch["qubit_spec"]["T2"]

    (transfer_dur, move_dur, _f_total, _f1q_native,
     _f2q_native, _fidle_native, _f_transfer_native, _f_coh_native) = out[:8]

    # 1q gates: never charged by mvqc, so compute from the circuit.
    c_1q = f_1q ** n_1q
    c_2q = f_2q ** n_2q
    c_idle = f_idle ** max(0, n_crosstalk)
    # Transfers: mvqc charges one factor per moved atom; ZAP charges pickup AND
    # drop. Doubling the count converts one convention to the other exactly.
    c_tr = f_tr ** (2 * n_transfers)

    # Decoherence, ZAP's definition: every microsecond a qubit is not busy.
    # mvqc's native cir_qubit_idle_time counts only movement time accrued while
    # a qubit sits outside storage, which is a strictly smaller quantity and
    # made PowerMove look better the longer its schedule ran — on ghz_n30 it
    # reported LESS decoherence than ZAP while taking 63% longer. Rebuilding it
    # from busy time removes that asymmetry.
    total = (transfer_dur + move_dur
             + n_2q_stages * dur["2qGate"]
             + n_1q_layers * dur["1qGate"])
    c_coh = math.exp(-len(busy_times) * total / t2)   # 2026-10-10: no busy-time credit

    return {
        "n_1q_gate": n_1q,
        "n_1q_layers": n_1q_layers,
        "total_duration": total,
        "move_transfer_duration": transfer_dur + move_dur,
        "total_fidelity": c_1q * c_2q * c_idle * c_tr * c_coh,
        "fidelity_1q_gate": c_1q,
        "fidelity_2q_gate": c_2q,
        "fidelity_idle": c_idle,
        "fidelity_handover": c_tr,
        "fidelity_decoherence": c_coh,
    }


def main():
    from run_zac import circuits, qasm2_path
    setting = sys.argv[1] if len(sys.argv) > 1 else "qasmbench"
    arch = json.load(open(os.path.join(ROOT, "baselines", "zap", "architectures", "default.json")))
    benchmarks = circuits(setting)

    QuantumCircuit, transpile, get_cz_blocks, PM = _load_pm()
    # PowerMove's own pickup/drop time is 8 us (MUS_PER_FRM, an animation-frame
    # constant); ZAP, ZAC and QMAP all use the architecture's 15 us. It only
    # feeds timing, never routing decisions, so set it to the shared value.
    PM.MUS_PER_FRM = arch["operation_duration"]["atom_transfer"]
    log_dir = os.path.join(ROOT, "results", "powermove", setting, "log")
    os.makedirs(log_dir, exist_ok=True)

    ok, failed = {}, {}
    for b in benchmarks:
        name = os.path.splitext(os.path.basename(b))[0]
        path = qasm2_path(b)
        circ = QuantumCircuit.from_qasm_file(path)
        # No second transpile (removed 2026-10-04): _bench_qasm2 already holds
        # the circuit at ZAP's level 3 in this basis, and re-transpiling it
        # gave sat_n11 one 1q gate fewer than every other compiler routed.
        tc = circ
        n_q = tc.num_qubits
        row = math.ceil(math.sqrt(n_q))
        n_1q = sum(1 for i in tc.data if i.operation.num_qubits == 1
                   and i.operation.name not in ("measure", "barrier"))
        n_2q = sum(1 for i in tc.data if i.operation.num_qubits == 2)

        n_1q_layers = count_1q_layers(tc)
        record = None
        # Prefer the zoned (storage) mode — that is the paper's contribution.
        # Fall back to no-storage only so a crash still yields information, and
        # tag which mode produced the number.
        for storage in (True, False):
            try:
                t0 = time.time()
                with contextlib.redirect_stdout(io.StringIO()):
                    out = PM.mvqc(get_cz_blocks(tc), row, n_q, storage)
                wall = time.time() - t0
                busy = list(PM.LAST_QUBIT_BUSY_TIME)
                n2s = PM.LAST_N_2Q_STAGES
                n_2q_pm = PM.LAST_N_2Q_GATES
                n_crosstalk = PM.LAST_N_CROSSTALK
                n_transfers = PM.LAST_N_TRANSFERS
                if n_2q_pm != n_2q:
                    # get_cz_blocks' own DAG decomposition doesn't always
                    # preserve every CZ from the qiskit-transpiled circuit
                    # 1:1 (seen on sat_n11/qram_n20) — score off PowerMove's
                    # own processed count, not the circuit's, so the exponent
                    # matches what it actually routed.
                    print(f"[PM] {name:18s} WARNING: n_2q processed {n_2q_pm} "
                          f"!= circuit's {n_2q}", flush=True)
                rec = score_powermove(out, busy, n_1q, n_1q_layers, n2s, n_2q_pm,
                                       n_crosstalk, n_transfers, arch)
                rec.update(algorithm=name, n_qubits=n_q, n_2q_gate=n_2q_pm,
                           stage=out[8], mode="storage" if storage else "no_storage",
                           compilation_time=round(wall, 4),
                           timestamp=datetime.now().isoformat())
                record = rec
                break
            except Exception as e:
                failed.setdefault(name, []).append(
                    f"storage={storage}: {type(e).__name__}: {str(e)[:60]}")

        if record is None:
            print(f"[PM] {name:18s} FAILED both modes: {failed[name]}", flush=True)
            continue

        p = os.path.join(log_dir, f"{name}.json")
        data = json.load(open(p)) if os.path.exists(p) else []
        data.append(record)
        json.dump(data, open(p, "w"), indent=2)
        ok[name] = record
        print(f"[PM] {name:18s} F={record['total_fidelity']:.6f} "
              f"mode={record['mode']}", flush=True)

    print(f"\n[PM] compiled {len(ok)}/{len(benchmarks)}; "
          f"storage mode on {sum(1 for r in ok.values() if r['mode']=='storage')}")
    if failed:
        print("[PM] failures:")
        for k, v in failed.items():
            print(f"      {k}: {v}")


if __name__ == "__main__":
    main()
