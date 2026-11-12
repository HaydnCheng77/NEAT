"""Re-score every finished job from its event trace under the current model.

    python3 experiments/end_to_end/rescore_all.py [compiler ...]     # default: all five

Uses rescore.score(..., recompute=True): every duration is re-derived from
the geometry and rescore.PARAMS (one timing rule for all compilers), so a
change of a physical constant needs no recompilation for compilers whose
decisions do not depend on it. Updates, per job:
  results/end_to_end/jobs/<job>.json          fidelity fields + "model"
  results/<compiler>/<suite>/log/<c>.json one appended record (readers use [-1])
Compile-time fields are kept from the compile that produced the trace.
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rescore as RS  # noqa: E402

MODEL = (f"T_TR={RS.PARAMS['T_TR']} us, T_1Q={RS.PARAMS['T_1Q']} us, "
         f"T_2Q={RS.PARAMS['T_2Q']} us, T2={RS.PARAMS['T2']} us, dec=n_q*T/T2 (recompute)")
FIELDS = ("n_1q_gate", "n_2q_gate", "total_duration", "total_fidelity", "fidelity_1q_gate",
          "fidelity_2q_gate", "fidelity_idle", "fidelity_handover", "fidelity_decoherence",
          "log_fidelity_decoherence", "neg_log10_fidelity", "n_transfers", "n_crosstalk")


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main():
    comps = sys.argv[1:] or ["zap", "neat", "zac", "mqt", "powermove"]
    n = 0
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "end_to_end", "jobs", "*.json"))):
        if re.search(r"\.a\d+\.json$", p):
            continue
        job = json.load(open(p))
        if not job.get("ok") or job["compiler"] not in comps:
            continue
        comp, suite, c = job["compiler"], job["suite"], job["circuit"]
        ev = os.path.join(ROOT, "results", comp, suite, "events", f"{c}.events.json.gz")
        sc = RS.score(RS.load(ev), recompute=True)
        job["neg_log10_fidelity"] = sc["neg_log10_fidelity"]
        job["model"] = MODEL
        atomic_json(p, job)
        lp = os.path.join(ROOT, "results", comp, suite, "log", f"{c}.json")
        data = json.load(open(lp))
        rec = dict(data[-1])
        rec.update({k: sc[k] for k in FIELDS})
        rec["model"] = MODEL
        data.append(rec)
        atomic_json(lp, data[-50:])
        n += 1
        if n % 50 == 0:
            print(n, "rescored", flush=True)
    print(f"{n} jobs rescored under: {MODEL}")


if __name__ == "__main__":
    main()
