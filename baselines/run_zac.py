"""Compile a benchmark suite with ZAC and log it in ZAP's schema.

NOTE: this standalone suite runner scores with the compiler's own durations, so its
numbers differ from the paper's by a few percent. The paper compiles through
tools/regen/worker.py (which imports this module) and rescores every trace with
tools/regen/rescore_all.py.

Run from the repository root:

    python3 baselines/run_zac.py [setting]        # default: qasmbench

Produces ``results/zac/<setting>/log/<circuit>.json`` with exactly the fields
``run.py``'s comparison reads, so ZAC becomes a third panel alongside zap and
NEAT without any special-casing downstream.

Three things are worth knowing about what this comparison does and does not
control for:

* **Metric: identical.** ZAC's own simulator omits the idle-crosstalk term and
  uses a linear coherence factor. ``score`` (below) re-scores ZAC's emitted
  ZAIR with ZAP's formulas instead, so all three rows are the same number. It
  agrees with ZAC's simulator to 15 digits on every term the two models share.
* **Circuit: identical — but only because it is now forced.** ``prepare_qasm2``
  decomposes each benchmark ONCE, at the optimization level ZAP itself would
  have used for that circuit width, and ZAC runs with ``resyn: False`` so it
  routes exactly that gate list. Verify with the gate-count check: ZAP's and
  ZAC's logged ``n_1q_gate``/``n_2q_gate`` must match per circuit.

  This was not always true. ZAC's pipeline transpiles at ``optimization_level=3``
  unconditionally while ZAP's level drops with circuit width, so before
  2026-09-02 the two routed different circuits wherever those disagreed —
  harmless on tqe / tqe2 / graphs (all <= 64 qubits, where both use the same
  level; 128 circuits verified identical) but severe on the scalability sweeps,
  where ZAC saw 6.7x fewer two-qubit gates at qft_n250. Any ZAC scalability
  number produced before that date is not comparable and must be regenerated.
* **Architecture: NOT controlled.** Each compiler runs on its own default
  hardware spec — ZAP on ``baselines/zap/architectures/default.json`` (80 storage traps, 6 µm
  pitch), ZAC on ``hardware_spec/full_architecture_tall.json`` (100x100 storage, 3 µm
  pitch, a much larger entanglement zone). This is a comparison of the tools as
  their authors ship them, not an isolation of the routing algorithm. Do not
  read a fidelity difference here as purely a compiler difference.
"""

import json
import math
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZAC = os.path.join(ROOT, "baselines", "zac")
sys.path.insert(0, ROOT)
from benchmark.suites import circuits, transpiled   # noqa: E402
# ZAC's own headline configuration: the last entry of their
# exp_setting/comparison_technique.json (SA placement + dynamic + reuse).
ZAC_SETTING = {
    "arch_spec": "hardware_spec/full_architecture_tall.json",   # the paper's ZAC array
    "dependency": True,
    "routing_strategy": "maximalis_sort",
    "scheduling": "asap",
    "trivial_placement": False,
    "dynamic_placement": True,
    "use_window": True,
    "window_size": 1000,
    "reuse": True,
    "use_verifier": True,
    # prepare_qasm2 has already decomposed the circuit at ZAP's own
    # optimization level; re-transpiling here (ZAC's default, always level 3)
    # would hand ZAC a different, easier circuit than ZAP routes.
    "resyn": False,
}


# ---------------------------------------------------------------------------
# Re-scoring ZAC's ZAIR trace with ZAP's fidelity model
# ---------------------------------------------------------------------------
#
# ZAC ships its own simulator, but it does not compute the same number ZAP's
# does, so its published fidelity cannot be put in the same column as ZAP's.
# Two concrete divergences (ZAC's zac/simulator/simulator.py, not included here):
#
# * the idle-crosstalk term is commented out (line 77), so a qubit parked in
#   the entanglement zone during someone else's Rydberg pulse is charged
#   nothing;
# * coherence is the linear ``1 - t/T2`` rather than ZAP's ``exp(-t/T2)``.
#
# Rather than translate ZAIR into ZAP's instruction schema — which would
# invent structure that isn't there and hide conversion bugs inside a number
# nobody can check — ``score`` below reads ZAIR directly and applies ZAP's
# formulas to it. Every term is the ZAP simulator's, term for term.
#
# What is deliberately NOT normalised: each compiler runs on its own default
# architecture and its own transpilation, so gate counts and trap geometry
# differ between rows. That is a comparison of tools-as-shipped, not a
# controlled experiment isolating the routing algorithm. Gate counts are
# reported alongside the fidelity so the confound stays visible.

def score(code_path, arch, ent_array_ids):
    """Apply ZAP's fidelity model to one ZAIR ``*_code.json``.

    Args:
        code_path: ZAIR file written by ZAC.
        arch: ZAP-style architecture dict (fidelity/duration/T2 knobs).
        ent_array_ids: set of SLM array ids that form the entanglement zone;
            needed to decide who is exposed to the Rydberg beam.

    Returns a dict with the same keys ZAP's ``log_results`` records.
    """
    fid = arch["operation_fidelity"]
    dur = arch["operation_duration"]
    f_2q = fid["two_qubit_gate"]
    f_1q = fid["single_qubit_gate"]
    f_tr = fid["atom_transfer"]
    f_idle = 1 - (1 - f_2q) / 2          # ZAP's derived idle-crosstalk factor
    t2 = arch["qubit_spec"]["T2"]
    t_2q = dur["2qGate"]
    t_1q = dur["1qGate"]

    insts = json.load(open(code_path))["instructions"]
    assert insts[0]["type"] == "init"
    n_q = len(insts[0]["init_locs"])

    busy = [0.0] * n_q
    n_1q = n_2q = n_transfer_events = n_crosstalk = 0
    total = 0.0
    in_zone = set()

    for inst in insts:
        t = inst["type"]

        # 1q timing, PARALLEL model (restored 2026-10-04 for the fairness
        # audit). ZAC's own router times a batch of co-scheduled 1q gates as
        # time_1qGate * len(batch) ("due to sequential execution",
        # baselines/zac/zac/router/router.py:306,602); ZAP, PowerMove's rescore
        # and QMAP's rescore all charge one t_1q per layer. ZAC was kept on its
        # sequential end_time from 2026-09-10; changed 2026-10-04 to score every compiler under one convention.
        # Measured: random_n10's nine all-qubit layers cost 520 us each under
        # the sequential model vs 52 us here. A 1qGate batch that hits one
        # qubit twice still costs t_1q per repeat (that qubit is sequential).
        # Every other instruction keeps ZAC's own span; ZAC's instructions are
        # contiguous (each begin_time == previous end_time), so a running sum
        # of spans is the makespan.
        if t == "1qGate":
            per_q = {}
            for g in inst["gates"]:
                per_q[g["q"]] = per_q.get(g["q"], 0) + 1
            span = t_1q * max(per_q.values(), default=0)
        elif "begin_time" in inst and "end_time" in inst:
            span = inst["end_time"] - inst["begin_time"]
        else:
            span = 0.0
        total += span

        if t == "init":
            continue

        if t == "1qGate":
            for g in inst["gates"]:
                n_1q += 1
                busy[g["q"]] += t_1q

        elif t == "rydberg":
            gates = inst["gates"]
            n_2q += len(gates)
            # ZAP charges the gate duration to the participants AND to every
            # idle atom sitting in the zone (its "Crosstalk" instruction), and
            # charges the idle ones one f_idle factor each.
            idle = len(in_zone) - 2 * len(gates)
            n_crosstalk += max(0, idle)
            for q in in_zone:
                busy[q] += t_2q

        elif t == "rearrangeJob":
            aod = inst["aod_qubits"]
            # One pickup and one drop per moved atom, exactly as ZAP's
            # Activate/Deactivate pair.
            n_transfer_events += len(aod)
            # The atom is held by the AOD for the whole job, so the whole span
            # (transfers + moves) is busy time — the analogue of ZAP charging
            # Activate + BigMove + Park + Deactivate.
            span = inst["end_time"] - inst["begin_time"]
            for q in aod:
                busy[q] += span
            # Track zone occupancy for the crosstalk term.
            for q_idx, array_id, *_ in inst["end_locs"]:
                if array_id in ent_array_ids:
                    in_zone.add(q_idx)
                else:
                    in_zone.discard(q_idx)
        else:
            raise ValueError(f"unknown ZAIR instruction type {t!r}")

    # f_coh = prod(exp(-deficit/t2)) = exp(-sum(deficit)/t2) -- computed as a
    # SUM first (never underflows: it is bounded above by n_q * total/t2, a
    # perfectly ordinary float) and only exponentiated once at the end. The
    # old code multiplied n_q individually-exponentiated per-qubit factors
    # together; each exp(-deficit/t2) is already representable for any single
    # qubit, but their PRODUCT underflows float's ~5e-324 floor once total
    # infidelity crosses roughly exp(-745) -- observed at qft_n450+ (2026-09),
    # where fidelity_decoherence silently became exactly 0.0, poisoning
    # total_fidelity to 0.0 too and dropping those sizes from every
    # scalability curve as "underflowed", identical to the failure
    # Zap._sig()'s docstring describes for ZAP's own logs. Summing the
    # exponents first defers the underflow to the same point ZAP's OWN
    # fidelity_decoherence hits (both are fundamentally exp() of a float), but
    # buys back a a wide extra range because ZAP was already paying this
    # exact cost via _log_losses reading run.py's per-cause fields, not
    # ZAC's -- see the log_fidelity_decoherence field below, which sidesteps
    # the exponentiation entirely for run.py's plots.
    coh_exponent = n_q * total / t2   # 2026-10-10: no busy-time credit
    f_coh = math.exp(-coh_exponent) if coh_exponent < 700 else 0.0

    c_1q = f_1q ** n_1q
    c_2q = f_2q ** n_2q
    c_idle = f_idle ** n_crosstalk
    c_tr = f_tr ** (2 * n_transfer_events)

    return {
        "n_qubits": n_q,
        "n_1q_gate": n_1q,
        "n_2q_gate": n_2q,
        "total_duration": total,
        "total_fidelity": c_1q * c_2q * c_idle * c_tr * f_coh,
        "fidelity_1q_gate": c_1q,
        "fidelity_2q_gate": c_2q,
        "fidelity_idle": c_idle,
        "fidelity_handover": c_tr,
        "fidelity_decoherence": f_coh,
        # Log-space decoherence loss (nats), ALWAYS finite and never
        # underflows regardless of circuit size -- run.py's _log_losses
        # (which is what every scalability curve actually plots; see its own
        # docstring on why raw fidelity underflows) prefers this field over
        # -ln(fidelity_decoherence) whenever the latter would be undefined
        # (fidelity_decoherence == 0.0). No other cause needs this: 1q/2q/idle
        # factors are f**n_gates for small per-gate f and bounded n_gates in
        # these suites, so they stay representable far longer than the
        # decoherence PRODUCT over up to n_q busy-time terms does.
        "log_fidelity_decoherence": coh_exponent,
    }


def entanglement_array_ids(zac_arch_path):
    """SLM array ids belonging to the entanglement zone of a ZAC hardware spec."""
    spec = json.load(open(zac_arch_path))
    ids = set()
    for zone in spec.get("entanglement_zones", []):
        for slm in zone["slms"]:
            ids.add(slm["id"])
    return ids


def zap_opt_level(n_qubits):
    """The transpiler optimization level ``Zap.set_program`` would use.

    Mirrors ``NEAT/neat.py`` exactly, which is the whole point: ZAC is
    fed the circuit ZAP routes, so this must not drift from it.

    Since 2026-09-03 that is level 3 for every width. It used to step down with
    circuit size (3/2/1/0 at 24/64/128/more), which is what made this helper
    necessary — ZAC's pipeline hardcodes level 3, so the two compilers routed
    materially different circuits wherever the ladder disagreed. Now they agree
    everywhere by construction, and this function exists to keep it that way if
    ZAP's choice ever changes again.
    """
    return 3


def qasm2_path(b):
    """Path of benchmark ``b``'s transpiled QASM2 (benchmark/transpiled/), building it first if absent."""
    return prepare_qasm2([b])[0]


def prepare_qasm2(benchmarks):
    """Write the EXACT circuit ZAP routes, pre-decomposed, as QASM 2 for ZAC.

    Two things are being controlled here, and the second one was a real bug.

    * ZAC parses only QASM 2, so v3 inputs are converted.
    * ZAC's pipeline transpiles its input with ``optimization_level=3``
      unconditionally (baselines/zac/zac/zac.py), while ZAP picks the level from
      the circuit width (``zap_opt_level`` above). On every circuit at or below
      64 qubits the two happen to agree, which is why the tqe / tqe2 / graphs
      comparisons are unaffected — but on the scalability sweeps they diverge
      hard. Measured on scalability/qft before this was fixed: ZAP routed
      62,250 two-qubit gates at n=250 against ZAC's 9,331, a 6.7x easier
      circuit for ZAC, purely from preprocessing. Plotted naively that reads as
      ZAC winning by orders of magnitude at scale.

    So the decomposition is done HERE, once, at ZAP's level, and ZAC is run
    with ``resyn: False`` (see ZAC_SETTING) so it routes this circuit verbatim
    instead of re-transpiling it. Both compilers then see an identical gate
    list and the comparison is about routing, which is the point.
    """
    from qiskit import QuantumCircuit, qasm2, transpile  # noqa: F401
    from qiskit_qasm3_import import parse

    out = []
    for b in benchmarks:
        dst = transpiled(b)
        out.append(dst)
        if os.path.exists(dst):
            continue
        print(f"[qasm2] transpiling {b} -> {dst}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        txt = open(os.path.join(ROOT, "benchmark", b)).read()
        circ = (QuantumCircuit.from_qasm_str(txt) if "OPENQASM 2" in txt
                else parse(txt))
        # Same trailing-swap strip Zap.set_program does, so both compilers
        # receive an identical circuit.
        while circ.data and circ.data[-1].operation.name == "swap":
            circ.data.pop()
        circ = transpile(circ, basis_gates=["cz", "id", "u2", "u1", "u3"],
                         optimization_level=zap_opt_level(circ.num_qubits),
                         seed_transpiler=0)
        qasm2.dump(circ, dst)
    return out


def compile_suite(qasm_paths, setting):
    """Compile each circuit with ZAC in this process, as experiments/end_to_end/worker.py
    does; returns each circuit's result directory (results/zac/<setting>/zac_output/)."""
    sys.path.insert(0, ZAC)
    from zac.ds.architecture import Architecture
    from zac.zac import ZAC as Compiler
    out = os.path.join(ROOT, "results", "zac", setting, "zac_output") + "/"
    for d in ("code", "time"):
        os.makedirs(out + d, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(ZAC)                      # ZAC_SETTING["arch_spec"] is relative to ZAC's folder
    try:
        spec = json.load(open(ZAC_SETTING["arch_spec"]))
        arch = Architecture(spec)
        arch.preprocessing()
        for i, qasm_path in enumerate(qasm_paths):
            name = os.path.splitext(os.path.basename(qasm_path))[0]
            print(f"[ZAC] ({i + 1}/{len(qasm_paths)}) compiling {name}", flush=True)
            z = Compiler()
            z.parse_setting(dict(ZAC_SETTING, dir=out, name=name))
            z.set_architecture_spec_path(ZAC_SETTING["arch_spec"])
            z.set_architecture(arch)
            z.set_program(qasm_path)
            z.solve(save_file=True)
    finally:
        os.chdir(cwd)
    return [out] * len(qasm_paths)


def _sig(value, digits=8):
    """Round to ``digits`` SIGNIFICANT figures, not decimal places.

    Mirrors ``NEAT/neat.py``'s ``_sig``, and for the same reason:
    ``round(x, 8)`` writes any fidelity below 5e-9 as exactly 0.0. That is
    harmless on tqe but destroys the scalability suites — ZAC's adder_n118
    lands at F ~ 1e-19 and qft_n100 far below that, and every one of them was
    being logged as a flat 0.0, which the scaling curves then had to drop as
    "underflowed" when the numbers were perfectly representable. Significant
    figures keep full precision on O(1) values and stay meaningful down to
    denormals.
    """
    return float(f"{value:.{digits}g}")


def write_logs(result_dir, benchmarks, setting):
    """Score each ZAIR trace with ZAP's model and write ZAP-schema logs."""
    arch = json.load(open(os.path.join(ROOT, "baselines", "zap", "architectures", "default.json")))
    ent = entanglement_array_ids(os.path.join(ZAC, ZAC_SETTING["arch_spec"]))
    log_dir = os.path.join(ROOT, "results", "zac", setting, "log")
    os.makedirs(log_dir, exist_ok=True)

    runtimes = {}
    for b in benchmarks:
        name = os.path.splitext(os.path.basename(b))[0]
        stem = os.path.splitext(os.path.basename(b))[0]   # ZAC names outputs after its input
        code = os.path.join(result_dir, "code", f"{stem}_code.json")
        if not os.path.exists(code):
            print(f"[ZAC] MISSING {name} (compilation produced no trace)")
            continue
        rec = score(code, arch, ent)
        rec["algorithm"] = name
        rec["stage"] = None
        # NOTE: ZAIR's top-level "runtime" is the compiled CIRCUIT's execution
        # time in µs, not the compiler's wall clock (it equals total_duration).
        # Compiler wall clock lives in the sibling time/ directory.
        time_file = os.path.join(result_dir, "time", f"{stem}_time.json")
        wall = 0.0
        if os.path.exists(time_file):
            wall = json.load(open(time_file)).get("total", 0.0)
        rec["compilation_time"] = round(wall, 4)
        rec["timestamp"] = datetime.now().isoformat()
        for k in ("total_fidelity", "fidelity_1q_gate", "fidelity_2q_gate",
                  "fidelity_idle", "fidelity_handover", "fidelity_decoherence"):
            rec[k] = _sig(rec[k])
        rec["total_duration"] = round(rec["total_duration"], 6)
        path = os.path.join(log_dir, f"{name}.json")
        data = json.load(open(path)) if os.path.exists(path) else []
        data.append(rec)
        json.dump(data, open(path, "w"), indent=2)
        runtimes[name] = rec["total_fidelity"]
        print(f"[ZAC] {name:18s} F={rec['total_fidelity']:.6f}")
    return runtimes


def main():
    """``run_zac.py [setting] [--rescore] [--only name,name]``

    ``--rescore`` skips compilation and re-scores the ZAIR traces already in
    results/zac/<setting>/zac_output/code (use after a change to ``score``).
    ``--only`` restricts either mode to the named circuits.
    """
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    setting = args[0] if args else "qasmbench"
    rescore = "--rescore" in sys.argv
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    benchmarks = circuits(setting)
    if only is not None:
        benchmarks = [b for b in benchmarks
                      if os.path.splitext(os.path.basename(b))[0] in only]
    fids = {}
    if rescore:
        result_dir = os.path.join(ROOT, "results", "zac", setting, "zac_output") + "/"
        fids.update(write_logs(result_dir, benchmarks, setting))
    else:
        qasm_paths = prepare_qasm2(benchmarks)
        result_dirs = compile_suite(qasm_paths, setting)
        # write_logs right after each circuit's compile, so a kill partway
        # through leaves every already-scored circuit's log on disk.
        for b, result_dir in zip(benchmarks, result_dirs):
            fids.update(write_logs(result_dir, [b], setting))
    print(f"[ZAC] total fidelity over {len(fids)} circuits: {sum(fids.values()):.4f}")


if __name__ == "__main__":
    main()
