"""Fairness audit of the regeneration results; exits non-zero on any failure.

    python3 experiments/end_to_end/audit.py

Checks, over every compiler x circuit:
  1. every log's last record was scored by rescore_all under ONE model string
  2. identical 1q/2q gate counts across the five compilers (same input circuit)
  3. every job finished ok with a clean timing and a calibration probe
  4. every record carries the shared legality checker's counts
"""
import glob, json, os, re, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from benchmark.suites import circuits   # noqa: E402
COMP = ["zap", "neat", "zac", "mqt", "powermove"]
SUITES = ["qasmbench", "scalability/graphs", "scalability/qft", "scalability/adder", "scalability/random"]
bad, models, gates, n = [], set(), {}, 0
for s in SUITES:
    for b in circuits(s):
        c = os.path.splitext(os.path.basename(b))[0]
        for comp in COMP:
            n += 1
            key = f"{comp}__{s.replace('/', '_')}__{c}"
            jp = os.path.join(ROOT, "results", "end_to_end", "jobs", key + ".json")
            lp = os.path.join(ROOT, "results", comp, s, "log", f"{c}.json")
            if not (os.path.exists(jp) and os.path.exists(lp)):
                bad.append(f"missing {key}"); continue
            j, r = json.load(open(jp)), json.load(open(lp))[-1]
            if not j.get("ok"): bad.append(f"failed {key}")
            if j.get("timing_clean") is False: bad.append(f"unclean timing {key}")
            if not j.get("calib_before"): bad.append(f"no probe {key}")
            models.add(r.get("model"))
            if not isinstance(r.get("aod_violations"), dict) or "ghost" not in r["aod_violations"]:
                bad.append(f"no shared-checker counts {key}")
            gates.setdefault((s, c), set()).add((r.get("n_1q_gate"), r.get("n_2q_gate")))
if len(models) != 1 or None in models:
    bad.append(f"{len(models)} scoring models in use: {models}")
bad += [f"gate counts differ {k}: {v}" for k, v in gates.items() if len(v) > 1]
print(f"{n} results audited; model: {models}")
print("\n".join(bad[:40]) if bad else "ALL CHECKS PASS")
sys.exit(1 if bad else 0)
