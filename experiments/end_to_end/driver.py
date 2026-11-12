"""Runs every compiler on every circuit of the paper (Table I and Fig. 6), resumably.

    nohup python3 experiments/end_to_end/driver.py > results/end_to_end/driver.out 2>&1 &

Five compilers x 110 circuits (the paper's 10 QASMBench circuits and four
scalability families at 25 sizes each), strictly one
compile at a time, each in a fresh process (experiments/end_to_end/worker.py), fastest
first. Kill it at any time and start it again: every finished job has a
marker in results/end_to_end/jobs/, so a restart only redoes the job that was
running. A lock file stops two drivers running at once.

Timing fairness: compile time is the worker's CPU-seconds measurement. Before
each job the driver waits (up to 20 min) for the 1-minute load average to drop
below LOAD_MAX; after it, a job whose wall time exceeds its CPU time by more
than 15% (contention), whose calibration probe ran slow (efficiency core:
CPU time inflates with wall time, so the ratio cannot see it), or that ran
under high load is re-queued, up to 3
attempts, and the cleanest attempt's timing kept. Results are deterministic,
so a rerun changes only the timing. After every job, a neighbour check (once
the whole suite is done) re-queues compile times more than 2x off the trend
of the neighbouring sizes.

Progress: results/end_to_end/progress.log (one line per job) and status.json.
"""
import fcntl
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from benchmark.suites import circuits   # noqa: E402
REGEN = os.path.join(ROOT, "results", "end_to_end")
JOBS = os.path.join(REGEN, "jobs")
COMPILERS = ["zap", "neat", "zac", "mqt", "powermove"]
SUITES = ["qasmbench", "scalability/graphs", "scalability/qft", "scalability/adder", "scalability/random"]
FIXED_SUITES = ("qasmbench",)     # one circuit per family, not size sweeps
TIMEOUT = 3 * 3600
LOAD_MAX = 4.0
RATIO_MAX = 1.15
CALIB_REF = 0.142     # s, worker.calibrate() on an M4 performance core (efficiency core: ~0.52)
CALIB_MAX = 1.4
MAX_ATTEMPTS = 3


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(REGEN, "progress.log"), "a") as f:
        f.write(line + "\n")


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def n_of(circuit):
    m = re.search(r"_n?(\d+)", circuit)
    return int(m.group(1)) if m else 0


def all_jobs():
    jobs = []
    for s in SUITES:
        for b in circuits(s):
            c = os.path.splitext(os.path.basename(b))[0]
            for comp in COMPILERS:
                jobs.append((comp, s, c))
    return jobs


def key(job):
    return "__".join([job[0], job[1].replace("/", "_"), job[2]])


# ---------------------------------------------------------------- preflight
def preflight():
    marker = os.path.join(REGEN, "preflight.done")
    if os.path.exists(marker):
        return
    # 1) estimates from the pre-run logs (for fastest-first order); kept from
    #    a first, interrupted preflight -- the logs are gone after step 2
    est = {}
    if os.path.exists(os.path.join(REGEN, "estimates.json")):
        all_jobs_ = []
    else:
        all_jobs_ = all_jobs()
    for comp, s, c in all_jobs_:
        p = f"results/{comp}/{s}/log/{c}.json"
        try:
            t = json.load(open(p))[-1].get("compilation_time")
        except Exception:
            t = None
        if not t:
            n = n_of(c)
            t = {"powermove": 30 + 0.8 * n, "zac": 10 + 0.1 * n}.get(comp, 0.05 * n)
        est[key((comp, s, c))] = float(t)
    if all_jobs_:
        atomic_json(os.path.join(REGEN, "estimates.json"), est)
    # 2) start every suite from empty results
    for comp in COMPILERS:
        for s in SUITES:
            for sub in ("log", "code", "events", "native"):
                d = f"results/{comp}/{s}/{sub}"
                if os.path.isdir(d):
                    subprocess.run(["rm", "-rf", d], check=True)
    # 3) make sure every circuit's transpiled file exists (benchmark/transpiled/
    #    ships with the repo; only missing files are built)
    sys.path.insert(0, os.path.join(ROOT, "baselines"))
    import run_zac as rz
    benches = sorted({b for s in SUITES for b in circuits(s)})
    rz.prepare_qasm2(benches)
    open(marker, "w").write(time.strftime("%Y-%m-%dT%H:%M:%S"))
    log(f"preflight: done ({len(benches)} circuits)")


# ---------------------------------------------------------------- running
def wait_quiet():
    # No time cap (2026-10-07): when another process saturates
    # the machine, wait for it rather than record noisy timings.
    t0 = time.time()
    while os.getloadavg()[0] > LOAD_MAX:
        if time.time() - t0 > 600 and int(time.time() - t0) % 1800 < 30:
            log(f"waiting for a quiet machine (load {os.getloadavg()[0]:.1f})")
        time.sleep(30)
    return os.getloadavg()


def run_job(job, attempt):
    comp, s, c = job
    out = os.path.join(JOBS, key(job) + f".a{attempt}.json")
    load0 = wait_quiet()
    logf = os.path.join(REGEN, "worker_logs", key(job) + f".a{attempt}.log")
    os.makedirs(os.path.dirname(logf), exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    t0 = time.time()
    try:
        with open(logf, "w") as lf:
            p = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "end_to_end", "worker.py"), comp, s, c, out],
                               stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=TIMEOUT)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    if rc == 0 and os.path.exists(out):
        r = json.load(open(out))
        r["attempt"] = attempt
        r["load_wait_start"] = load0
        return r
    tail = open(logf).read()[-1500:] if os.path.exists(logf) else ""
    return {"ok": False, "compiler": comp, "suite": s, "circuit": c, "attempt": attempt,
            "error": "timeout (3 h)" if rc == "timeout" else f"rc={rc}: {tail.strip().splitlines()[-1] if tail.strip() else ''}",
            "job_wall": time.time() - t0}


def suspect(r):
    if not r.get("ok"):
        return False
    cpu, wall = r["cpu"], r["wall"]
    if cpu > 1.0 and wall / cpu > RATIO_MAX:
        return f"wall/cpu {wall / cpu:.2f}"
    cal = max(r.get("calib_before", 0), r.get("calib_after", 0))
    if cal > CALIB_MAX * CALIB_REF:
        return f"slow core (probe {cal:.2f}s vs {CALIB_REF}s)"
    if max(r["load_before"][0], r["load_after"][0]) > LOAD_MAX + 1:
        return f"load {max(r['load_before'][0], r['load_after'][0]):.1f}"
    return False


def final_path(job):
    return os.path.join(JOBS, key(job) + ".json")


def fix_log_timing(job, best):
    """All attempts are deterministic: point the log record at the cleanest timing."""
    comp, s, c = job
    p = f"results/{comp}/{s}/log/{c}.json"
    try:
        data = json.load(open(p))
        data[-1].update({"compilation_time": round(best["cpu"], 4), "compile_cpu_s": best["cpu"],
                         "compile_wall_s": best["wall"], "timing_attempt": best["attempt"]})
        atomic_json(p, data)
    except Exception as e:
        log(f"  warning: could not update timing in {p}: {e}")


def do_job(job, reason=""):
    comp, s, c = job
    attempts = []
    for a in range(1, MAX_ATTEMPTS + 1):
        r = run_job(job, a)
        attempts.append(r)
        if not r.get("ok"):
            log(f"FAIL {comp:12s} {s:20s} {c:16s} attempt {a}: {r['error']}")
            if r["error"].startswith("timeout") or a >= 2:
                break
            continue
        why = suspect(r)
        log(f"done {comp:12s} {s:20s} {c:16s} cpu {r['cpu']:9.2f}s wall/cpu "
            f"{(r['wall'] / r['cpu']) if r['cpu'] > 0 else 1:.2f} -log10F {r['neg_log10_fidelity']:10.3f} "
            f"viol {r['aod_violations']}{' mode=' + r['mode'] if r.get('mode') else ''}"
            f"{' SUSPECT ' + why if why else ''}{' (' + reason + ')' if reason else ''}")
        if not why:
            break
    ok = [r for r in attempts if r.get("ok")]
    if ok:
        best = min(ok, key=lambda r: (bool(suspect(r)), r["wall"] / max(r["cpu"], 1e-9)))
        best["attempts"] = len(attempts)
        best["timing_clean"] = not suspect(best)
        fix_log_timing(job, best)
        atomic_json(final_path(job), best)
    else:
        atomic_json(final_path(job), {"ok": False, "compiler": comp, "suite": s, "circuit": c,
                                      "error": attempts[-1]["error"], "attempts": len(attempts)})


def qmap_family_failed(job, done):
    """User rule (2026-10-06): once QMAP fails on a circuit (error or 3 h
    timeout), every larger circuit of that family is skipped -- its search
    blows up past a size (adder n748, ising n850, random n550)."""
    comp, s, c = job
    if comp not in ("mqt", "zap") or s in FIXED_SUITES:   # zap added 2026-10-06
        return None
    for r in done.values():
        if (r.get("compiler") == comp and r.get("suite") == s and not r.get("ok")
                and not r.get("skipped") and n_of(r["circuit"]) < n_of(c)):
            return (f"skipped by rule: {comp} failed on smaller {r['circuit']} "
                    f"({r.get('error', '')[:60]})")
    return None


def neighbour_outliers(done):
    """(job, why) whose CPU time is > 2x off both neighbouring sizes' trend."""
    out = []
    by = {}
    for k, r in done.items():
        if r.get("ok") and r["suite"] not in FIXED_SUITES:
            by.setdefault((r["compiler"], r["suite"]), []).append(r)
    for (comp, s), rs in by.items():
        rs.sort(key=lambda r: n_of(r["circuit"]))
        for i in range(1, len(rs) - 1):
            a, b, cur = rs[i - 1]["cpu"], rs[i + 1]["cpu"], rs[i]["cpu"]
            if cur < 1.0 or rs[i].get("neighbour_checked"):
                continue
            lo, hi = min(a, b), max(a, b)
            if cur > 2 * hi or cur < 0.5 * lo:
                out.append(((comp, s, rs[i]["circuit"]), f"cpu {cur:.1f}s vs neighbours {a:.1f}/{b:.1f}s"))
    return out


def status(done, total, current, t_start):
    ok = sum(1 for r in done.values() if r.get("ok"))
    fail = sum(1 for r in done.values() if not r.get("ok"))
    atomic_json(os.path.join(REGEN, "status.json"), {
        "done": len(done), "ok": ok, "failed": fail, "total": total, "current": current,
        "elapsed_h": (time.time() - t_start) / 3600,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S")})


def load_done():
    done = {}
    for p in glob.glob(os.path.join(JOBS, "*.json")):
        if re.search(r"\.a\d+\.json$", p):
            continue
        r = json.load(open(p))
        done[os.path.basename(p)[:-5]] = r
    return done


def main():
    os.makedirs(JOBS, exist_ok=True)
    lock = open(os.path.join(REGEN, "driver.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another driver is running")
    log(f"driver start (pid {os.getpid()})")
    preflight()
    ep = os.path.join(REGEN, "estimates.json")          # only orders jobs fastest-first
    est = json.load(open(ep)) if os.path.exists(ep) else {}
    jobs = sorted(all_jobs(), key=lambda j: (est.get(key(j), 1e9), j))
    t_start = time.time()
    done = load_done()
    for job in jobs:
        if key(job) in done:
            continue
        why = qmap_family_failed(job, done)
        if why:
            atomic_json(final_path(job), {"ok": False, "skipped": True, "compiler": job[0], "suite": job[1],
                                          "circuit": job[2], "error": why})
            log(f"SKIP {job[0]:12s} {job[1]:20s} {job[2]:16s} {why}")
            done = load_done()
            continue
        status(done, len(jobs), key(job), t_start)
        do_job(job)
        done = load_done()
    # neighbour check, once everything has a first result
    for job, why in neighbour_outliers(done):
        log(f"neighbour outlier {key(job)}: {why} -> rerun")
        do_job(job, reason="neighbour outlier: " + why)
        p = final_path(job)
        r = json.load(open(p))
        r["neighbour_checked"] = True
        atomic_json(p, r)
    done = load_done()
    status(done, len(jobs), None, t_start)
    log(f"ALL DONE: {sum(1 for r in done.values() if r.get('ok'))}/{len(jobs)} ok")


if __name__ == "__main__":
    main()
