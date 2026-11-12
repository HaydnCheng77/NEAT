"""Compile ONE circuit with ONE compiler, in this (fresh) process.

    python3 experiments/end_to_end/worker.py <compiler> <suite> <circuit> <job_json_out>

compiler: zap | neat | zac | mqt | powermove

Writes, all under results/<compiler>/<suite>/:
  events/<circuit>.events.json.gz   shared event trace (experiments/end_to_end/events.py)
  native/<circuit>.*.gz             the compiler's own output (zap/NEAT: code/)
  log/<circuit>.json                one record appended, ZAP log schema
and finally <job_json_out> (written atomically: its presence marks the job
done). Fidelity in every record comes from experiments/end_to_end/rescore.py on the
events, so all five compilers are scored by the same code; AOD legality from
experiments/end_to_end/legality.py.

Compile time = CPU seconds (time.process_time) of the compiler's own work
only -- no transpiling, file I/O, scoring or verification -- measured the same
way for every compiler, plus wall seconds over the same region:
  zap / NEAT   Scheduler + Router (placement happens inside the router)
  ZAC          scheduling() through route_qubit()
  QMAP         RoutingAwareCompiler.compile()
  PowerMove    get_cz_blocks() + mvqc()
"""
import contextlib
import gc
import gzip
import io
import json
import math
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "baselines"))
import events as E      # noqa: E402
import legality as L    # noqa: E402
import rescore as RS    # noqa: E402

ZAC_ARCH = "hardware_spec/full_architecture_tall.json"   # ZAC + QMAP


class Timer:
    """Accumulates CPU and wall time over (possibly several) timed regions.

    The first start() runs one full gc.collect() before reading the clocks
    (2026-10-11), so cyclic garbage left by the untimed setup is freed outside
    the timed region. Without it, loading an OpenQASM 3 benchmark (qft, adder)
    leaves its ANTLR parse tree -- 4.6M objects on qft_n400, ~7.5 s to free --
    for the cyclic GC, and whether that collection fired inside the timed
    region depended on allocation counts: NEAT's qft curve jumped 2-3x at
    n=250/700/750/1000 (and n=400/550/800 after an unrelated allocation
    change). Same rule for every compiler; collections triggered by the
    compiler's own garbage still count."""
    def __init__(self):
        self.cpu = self.wall = 0.0
        self._c = self._w = None
        self._collected = False
    def start(self):
        if not self._collected:
            gc.collect()
            self._collected = True
        self._c, self._w = time.process_time(), time.perf_counter()
    def stop(self):
        self.cpu += time.process_time() - self._c
        self.wall += time.perf_counter() - self._w


def calibrate():
    """Fixed single-thread workload (2026-10-06). ~0.14 s on an M4 performance
    core, ~0.52 s on an efficiency core: CPU time is only comparable across
    jobs that ran on the same kind of core, and wall/CPU cannot tell (both
    inflate together), so the driver re-runs a job whose probe is slow."""
    t = time.perf_counter()
    s = 0
    for i in range(3_000_000):
        s += i * i
    return time.perf_counter() - t


def _atomic_json(path, obj, gz=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with (gzip.open(tmp, "wt") if gz else open(tmp, "w")) as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _atomic_text_gz(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        f.write(text)
    os.replace(tmp, path)


def _append_log(path, rec):
    data = json.load(open(path)) if os.path.exists(path) else []
    data.append(rec)
    _atomic_json(path, data[-50:])


def _bench_path(suite, circuit):
    return f"{suite}/{circuit}.qasm"


# ------------------------------------------------------------------ zap / NEAT
def run_zap(compiler, suite, circuit, timer):
    os.chdir(ROOT)
    import NEAT.run as NR
    import run_zap as ZR
    if compiler == "zap":
        import zap.zap as ZM
        from zap.zap import Zap as Cls
    else:
        import NEAT.neat as ZM
        from NEAT.neat import Neat as Cls
    for nm in ("Scheduler", "Router"):
        orig = getattr(ZM, nm)
        def wrapped(*a, _orig=orig, **k):
            timer.start()
            try:
                return _orig(*a, **k)
            finally:
                timer.stop()
        setattr(ZM, nm, wrapped)
    bench = _bench_path(suite, circuit)
    arch = NR.architecture_for(bench, verbose=False) if compiler == "neat" else ZR.architecture_for(suite)
    z = Cls(benchmark=bench, architecture=arch, initial_mapping=[],
            output_dir=f"{compiler}/{suite}",
            scheduling_strategy=NR.SCHEDULING if compiler == "neat" else ZR.SCHEDULING,
            placement_strategy="baseline", routing_strategy="baseline")
    with contextlib.redirect_stdout(io.StringIO()):
        z.solve(True, False)
    rc = z.results_code
    traps = [tuple(p) for p in z.stg_slm_sites] + [tuple(p) for p in z.ent_slm_sites]
    ev = E.from_zap(rc, traps)
    log_path = f"results/{compiler}/{suite}/log/{circuit}.json"
    own = json.load(open(log_path))[-1]
    # set_downs / unchecked_finish: mid-air deadlocks the router had to break (2026-10-31).
    extra = ({"own_verifier": rc.get("verification", {}).get("counts"),
              "set_downs": rc.get("set_downs", 0), "unchecked_finish": rc.get("unchecked_finish", 0)}
             if compiler == "neat" else {})
    return ev, None, own, extra


# ------------------------------------------------------------------ ZAC
def run_zac(suite, circuit, timer, native_path):
    import run_zac as rz
    zdir = rz.ZAC
    os.chdir(zdir)
    sys.path.insert(0, zdir)
    from zac.ds.architecture import Architecture
    from zac.zac import ZAC
    spec = json.load(open(ZAC_ARCH))
    arch = Architecture(spec)
    arch.preprocessing()
    out_dir = os.path.join(ROOT, "results", "end_to_end", "zac_tmp", f"{suite.replace('/', '_')}__{circuit}") + "/"
    for d in ("code", "time"):
        os.makedirs(out_dir + d, exist_ok=True)
    setting = dict(rz.ZAC_SETTING, arch_spec=ZAC_ARCH, dir=out_dir, name=circuit)
    z = ZAC()
    z.parse_setting(setting)
    z.set_architecture_spec_path(ZAC_ARCH)
    z.set_architecture(arch)
    z.set_program(rz.qasm2_path(_bench_path(suite, circuit)))
    o_sched, o_route = ZAC.scheduling, ZAC.route_qubit
    def sched(self, *a, **k):
        timer.start()
        return o_sched(self, *a, **k)
    def route(self, *a, **k):
        try:
            return o_route(self, *a, **k)
        finally:
            timer.stop()
    ZAC.scheduling, ZAC.route_qubit = sched, route
    with contextlib.redirect_stdout(io.StringIO()):
        z.solve(save_file=True)
    os.chdir(ROOT)
    code_text = open(z.code_filename).read()
    _atomic_text_gz(native_path + ".zair.json.gz", code_text)
    import shutil
    shutil.rmtree(out_dir, ignore_errors=True)
    ent = {s["id"] for zz in spec["entanglement_zones"] for s in zz["slms"]}
    return E.from_zair(json.loads(code_text), spec, ent), None, {}, {}


# ------------------------------------------------------------------ QMAP
def run_qmap(suite, circuit, timer, native_path):
    os.chdir(ROOT)
    import run_mqt as qz
    import run_zac as rz
    from mqt.qmap.na.zoned import RoutingAwareCompiler, ZonedNeutralAtomArchitecture
    from mqt.core import load
    from qiskit import QuantumCircuit
    tall = os.path.join(rz.ZAC, ZAC_ARCH)
    qarch = qz._qmap_arch_from_zac(tall)
    arch = ZonedNeutralAtomArchitecture.from_json_string(json.dumps(qarch))
    ranges = [tuple(map(tuple, b)) for b in qarch["rydberg_range"]]
    def in_zone(p):
        return any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for (x0, y0), (x1, y1) in ranges)
    c = QuantumCircuit.from_qasm_file(rz.qasm2_path(_bench_path(suite, circuit)))
    c.remove_final_measurements(inplace=True)
    # Search Smarter defaults: IDS + relaxed. QMAP_WINDOW_SHARE (2026-10-06):
    # QMAP's own remedy for "No path from start to any goal found" is a larger
    # window_share (default 0.8); used only as a recorded fallback for
    # circuits that fail on the defaults.
    kw = {}
    if os.environ.get("QMAP_WINDOW_SHARE"):
        kw["window_share"] = float(os.environ["QMAP_WINDOW_SHARE"])
    comp = RoutingAwareCompiler(arch, log_level="E", **kw)
    qc = load(c)
    timer.start()
    code = comp.compile(qc)
    timer.stop()
    _atomic_text_gz(native_path + ".naviz.gz", code)
    _, traps = E.zac_sites(json.load(open(tall)))
    extra = {"qmap_window_share": kw["window_share"]} if kw else {}
    return E.from_naviz(code, c.num_qubits, traps, in_zone), None, {}, extra


# ------------------------------------------------------------------ PowerMove
def run_powermove(suite, circuit, timer, native_path):
    import run_powermove as rp
    import run_zac as rz
    QC, _transpile, get_cz_blocks, PM = rp._load_pm()       # chdir's into PowerMove
    PM.MUS_PER_FRM = E.T_TR
    tc = QC.from_qasm_file(rz.qasm2_path(_bench_path(suite, circuit)))
    n = tc.num_qubits
    row = math.ceil(math.sqrt(n))
    mode, err = None, []
    for storage in (True, False):
        try:
            timer.start()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    PM.mvqc(get_cz_blocks(tc), row, n, storage)
            finally:
                timer.stop()
            mode = "storage" if storage else "no_storage"
            break
        except Exception as e:      # recorded; the run falls back once
            err.append(f"storage={storage}: {type(e).__name__}: {str(e)[:120]}")
    os.chdir(ROOT)
    if mode is None:
        raise RuntimeError("; ".join(err))
    trace = list(PM.LAST_TRACE)
    _atomic_json(native_path + ".mvqc_trace.json.gz", trace, gz=True)
    ev = E.from_powermove(trace, E.powermove_1q_layers(tc), row)
    return ev, mode, {}, {"fallback_errors": err} if err else {}


def main():
    compiler, suite, circuit, job_out = sys.argv[1:5]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    base = os.path.join(ROOT, "results", compiler, suite)
    native = os.path.join(base, "native", circuit)
    timer = Timer()
    t_job = time.perf_counter()
    load_before = os.getloadavg()
    calib_before = sorted(calibrate() for _ in range(3))[1]
    if compiler in ("zap", "neat"):
        ev, mode, own, extra = run_zap(compiler, suite, circuit, timer)
    elif compiler == "zac":
        ev, mode, own, extra = run_zac(suite, circuit, timer, native)
    elif compiler == "mqt":
        ev, mode, own, extra = run_qmap(suite, circuit, timer, native)
    elif compiler == "powermove":
        ev, mode, own, extra = run_powermove(suite, circuit, timer, native)
    else:
        raise SystemExit(f"unknown compiler {compiler}")
    calib_after = calibrate()
    load_after = os.getloadavg()
    os.chdir(ROOT)
    ev["meta"] = {"compiler": compiler, "suite": suite, "circuit": circuit}
    _atomic_json(os.path.join(base, "events", f"{circuit}.events.json.gz"), ev, gz=True)
    sc = RS.score(ev)
    leg = L.check(ev)
    rec = dict(own)
    rec.update({k: sc[k] for k in ("n_qubits", "n_1q_gate", "n_2q_gate", "total_duration", "total_fidelity",
                                   "fidelity_1q_gate", "fidelity_2q_gate", "fidelity_idle",
                                   "fidelity_handover", "fidelity_decoherence",
                                   "log_fidelity_decoherence")})
    rec.update({"algorithm": circuit, "neg_log10_fidelity": sc["neg_log10_fidelity"],
                "n_transfers": sc["n_transfers"], "n_crosstalk": sc["n_crosstalk"],
                "compilation_time": round(timer.cpu, 4), "compile_cpu_s": timer.cpu,
                "compile_wall_s": timer.wall, "load_before": load_before, "load_after": load_after,
                "aod_legal": leg["legal"], "aod_violations": leg["counts"], "aod_first": leg["first"],
                "calib_before_s": calib_before, "calib_after_s": calib_after,
                "mode": mode, "regen": "2026-10-06", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
    rec.update(extra)
    if compiler in ("zap", "neat"):
        log_path = os.path.join(base, "log", f"{circuit}.json")
        data = json.load(open(log_path))
        data[-1] = rec                      # replace the record solve() just wrote
        _atomic_json(log_path, data)
    else:
        _append_log(os.path.join(base, "log", f"{circuit}.json"), rec)
    _atomic_json(job_out, {"ok": True, "compiler": compiler, "suite": suite, "circuit": circuit,
                           "cpu": timer.cpu, "wall": timer.wall, "job_wall": time.perf_counter() - t_job,
                           "load_before": load_before, "load_after": load_after,
                           "calib_before": calib_before, "calib_after": calib_after,
                           "neg_log10_fidelity": sc["neg_log10_fidelity"], "aod_violations": leg["violations"],
                           "n_1q": sc["n_1q_gate"], "n_2q": sc["n_2q_gate"], "mode": mode})


if __name__ == "__main__":
    main()
