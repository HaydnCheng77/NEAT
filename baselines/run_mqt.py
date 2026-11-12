"""Compile a benchmark suite with MQT QMAP's zoned RoutingAwareCompiler and log
it in ZAP's schema.

NOTE: this standalone suite runner scores with the compiler's own durations, so its
numbers differ from the paper's by a few percent. The paper compiles through
tools/regen/worker.py (which imports this module) and rescores every trace with
tools/regen/rescore_all.py.

Run from the repository root:

    python3 baselines/run_mqt.py [setting]      # default: qasmbench

Writes ``results/mqt/<setting>/log/<circuit>.json`` with the same
fields ``run.py``'s comparison reads (mirroring ``baselines/run_zac.py``), so
QMAP's routing-aware zoned compiler becomes another panel alongside zap,
NEAT, ZAC and PowerMove without any special-casing downstream.

What this does and does not control for, following ``run_zac.py``'s pattern:

* **Metric: identical.** ``score()`` below is ZAP's own fidelity model
  (ported verbatim from ``run_zac.py``'s ``score()``), applied to a busy-time/
  transfer/crosstalk trace reconstructed from QMAP's own ``.naviz`` output.
  QMAP's Python API exposes no fidelity number at all (only per-stage
  compiler wall-clock via ``compiler.stats()``), so there is no "native"
  number to disagree with here the way there was for ZAC/PowerMove.
* **Circuit: identical to what zap/NEAT/ZAC/PowerMove route.** Reads
  the same pre-transpiled QASM2 in ``benchmark/transpiled`` (already
  decomposed at ZAP's own optimization level for that circuit width -- see
  ``run_zac.py``'s ``prepare_qasm2``), and strips only the trailing
  measurements (QMAP's compiler does not accept ``measure`` as an operation
  type; every other compiler in this comparison also never routes
  measurements).
* **Timing: RECONSTRUCTED, not measured.** Unlike ZAC's ZAIR, ``.naviz`` has
  no begin_time/end_time on any instruction -- it is a plain ordered
  instruction stream. This script derives a wall-clock the same way ZAP's own
  simulator does: single-qubit gates and CZ pulses cost their configured
  duration; a load/move.../store transfer sequence costs
  ``2 * atom_transfer_duration`` (pickup + drop) plus ZAP's own
  ``movement_duration(d) = 200*sqrt(d/110)`` for the total Euclidean distance
  travelled across the move chain, using ZAP's OWN transport model (not
  QMAP's -- QMAP's compiler does not expose a movement-time constant at all,
  it only reasons about which sites are reachable). This makes the timing
  axis comparable to zap/NEAT in mechanism, not merely in units.
* **Architecture: ZAC's, not zap's.** QMAP ships no architecture of its own
  (it takes one as JSON), so it runs on ZAC's
  ``hardware_spec/full_architecture_tall.json``, translated into QMAP's schema at
  import (``_qmap_arch_from_zac``) -- the QMAP and ZAC rows share hardware
  exactly; neither shares zap/NEAT's.
* **Compiler config: the paper's own defaults.** ``RoutingAwareCompiler``'s
  constructor defaults are ``placement_method=PlacementMethod.ids,
  routing_method=RoutingMethod.relaxed`` -- i.e. the "Search Smarter, Not
  Harder" configuration, not the earlier A*/strict-routing one. No arguments
  are overridden here.
"""

import json
import math
import os
import re
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ZAC's hardware_spec/full_architecture_tall.json, translated into QMAP's schema
# at import time so the two stay identical (2026-10-04). Until then this was a
# hand-written copy of QMAP's docs example, described here as "geometrically
# identical" to ZAC's but not: 20 storage rows instead of 100 (zone 10 um from
# storage instead of 10 um past a 297 um-tall storage block) and ZAC's
# pre-2026-09-10 entanglement zone (c=20, 140 Rydberg pairs, the cap that
# split every layer above n=280 on random/adder) instead of ZAC's current
# c=80 (560 pairs).
ZAC_ARCH_PATH = os.path.join(ROOT, "baselines", "zac", "hardware_spec",
                             "full_architecture_tall.json")     # the paper's ZAC array


def _qmap_arch_from_zac(path):
    zac = json.load(open(path))

    def zone(z):
        return {
            "zone_id": z["zone_id"],
            "slms": [{"id": s["id"], "site_separation": s["site_seperation"],
                      "r": s["r"], "c": s["c"], "location": s["location"]}
                     for s in z["slms"]],
            "offset": z["offset"],
            "dimension": z.get("dimension", z.get("dimenstion")),
        }

    d, f = zac["operation_duration"], zac["operation_fidelity"]
    return {
        "name": "ZAC full_architecture (translated)",
        "operation_duration": {"rydberg_gate": d["rydberg"],
                               "single_qubit_gate": d["1qGate"],
                               "atom_transfer": d["atom_transfer"]},
        "operation_fidelity": {"rydberg_gate": f["two_qubit_gate"],
                               "single_qubit_gate": f["single_qubit_gate"],
                               "atom_transfer": f["atom_transfer"]},
        "qubit_spec": {"T": zac["qubit_spec"]["T"]},
        "storage_zones": [zone(z) for z in zac["storage_zones"]],
        "entanglement_zones": [zone(z) for z in zac["entanglement_zones"]],
        "aods": [{"id": a["id"], "site_separation": a["site_seperation"],
                  "r": a["r"], "c": a["c"]} for a in zac["aods"]],
        "rydberg_range": zac["rydberg_range"],
    }


_QMAP_ARCH = _qmap_arch_from_zac(ZAC_ARCH_PATH)
QMAP_ARCH_JSON = json.dumps(_QMAP_ARCH)

# ZAP-schema constants for score(), read from the same architecture.
ARCH = {
    "operation_duration": {"2qGate": _QMAP_ARCH["operation_duration"]["rydberg_gate"],
                           "1qGate": _QMAP_ARCH["operation_duration"]["single_qubit_gate"],
                           "atom_transfer": _QMAP_ARCH["operation_duration"]["atom_transfer"]},
    "operation_fidelity": {"two_qubit_gate": _QMAP_ARCH["operation_fidelity"]["rydberg_gate"],
                           "single_qubit_gate": _QMAP_ARCH["operation_fidelity"]["single_qubit_gate"],
                           "atom_transfer": _QMAP_ARCH["operation_fidelity"]["atom_transfer"]},
    "qubit_spec": {"T2": _QMAP_ARCH["qubit_spec"]["T"]},
}

# A point is "in the entanglement zone" if it falls in any rydberg_range box.
RYDBERG_RANGE = [tuple(map(tuple, box)) for box in _QMAP_ARCH["rydberg_range"]]


def _in_zone(xy):
    x, y = xy
    for (x0, y0), (x1, y1) in RYDBERG_RANGE:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def movement_duration(d):
    """ZAP's own transport model (NEAT/transport.py): 200*sqrt(d/110) us."""
    return 200.0 * math.sqrt(d / 110.0) if d > 0 else 0.0


_COORD_RE = re.compile(r"\(\s*([\-0-9.]+)\s*,\s*([\-0-9.]+)\s*\)\s*(\w+)")
_ATOM_TOK_RE = re.compile(r"^\w+$")


def _parse_block(lines, i):
    """Parse a bracketed ``[ ... ]`` block starting at line i (the ``[``
    itself may be on the same line as the opening keyword, already stripped).
    Returns (list of raw item strings, next index).
    """
    items = []
    i += 1
    while lines[i].strip() != "]":
        items.append(lines[i].strip())
        i += 1
    return items, i + 1


def parse_naviz(code, n_q):
    """Reconstruct a ZAP-style trace (busy time, gate/transfer/crosstalk
    counts) from one QMAP ``.naviz`` program.

    Returns the same dict shape ``score()`` (below) expects from ZAC's ZAIR,
    computed directly rather than via an intermediate instruction list, since
    ``.naviz`` carries no timestamps to replay against.
    """
    dur = ARCH["operation_duration"]
    t_1q = dur["1qGate"]
    t_2q = dur["2qGate"]
    t_transfer = dur["atom_transfer"]

    busy = [0.0] * n_q
    pos = {}          # atom name -> (x, y), current known position
    in_zone = set()   # atom names currently inside the Rydberg range
    airborne = set()  # atom names currently picked up by the AOD
    layer_1q = None   # atoms in the open 1q layer, None if no layer is open
    total = 0.0
    n_1q = 0
    n_2q = 0
    n_transfer_events = 0
    n_crosstalk = 0
    name_to_idx = {}

    lines = code.split("\n")
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i].strip()
        if not raw:
            i += 1
            continue
        if raw.startswith("atom ("):
            m = _COORD_RE.match(raw[len("atom "):])
            x, y, name = float(m.group(1)), float(m.group(2)), m.group(3)
            pos[name] = (x, y)
            name_to_idx[name] = len(name_to_idx)
            if _in_zone((x, y)):
                in_zone.add(name)
            i += 1
            continue

        line = raw[2:].strip() if raw.startswith("@+") else raw
        parts = line.split(None, 1)
        kw = parts[0] if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

        if kw in ("u", "rz"):
            # u theta phi lambda atom_name / rz theta atom_name.
            # 1q timing, PARALLEL per layer (2026-10-04). QMAP's ASAPScheduler
            # does group 1q gates into layers between CZ layers, but its
            # CodeGenerator writes each gate as its own single-atom "@+" line,
            # and .naviz carries no timestamps -- so this used to charge every
            # line its own 52 us, serialising the layer (adder_n199: 1,938
            # gates = 101 ms, vs 26 ms for its 497 layers). A layer is now a
            # run of consecutive 1q lines, closed by any other instruction or
            # by an atom repeating (that atom's gates are sequential). On
            # adder_n199 this reproduces QMAP's own layer count exactly (497
            # = 496 CZ layers + 1). Same convention as ZAP and the other
            # rescorers.
            name = rest.split()[-1]
            busy[name_to_idx[name]] += t_1q
            if layer_1q is None or name in layer_1q:
                total += t_1q
                layer_1q = set()
            layer_1q.add(name)
            n_1q += 1
            i += 1
            continue
        layer_1q = None
        if kw == "cz":
            # Global CZ pulse over the entanglement zone: every atom
            # currently in_zone is either a gate participant (paired) or an
            # idle crosstalk exposure (ZAP/ZAC convention: idle = occupants
            # not part of a pair). QMAP guarantees valid Rydberg pairs by
            # construction (its own rydberg_range / blockade-radius
            # placement), so the occupant count is always even; no need to
            # resolve actual pairing to score gate count or crosstalk.
            n_gates = len(in_zone) // 2
            n_2q += n_gates
            idle = len(in_zone) - 2 * n_gates
            n_crosstalk += max(0, idle)
            for name in in_zone:
                busy[name_to_idx[name]] += t_2q
            total += t_2q
            i += 1
        elif kw == "load":
            # Atoms become "airborne" (picked up by the AOD) here, and are
            # charged one pickup cost immediately -- mirrors ZAP's Activate.
            # Tracked per-atom globally rather than as matched
            # (load, move*, store) triples: QMAP's AOD can merge
            # separately-loaded atoms into a shared move batch mid-flight
            # and release (store) them at different times, in a different
            # group than the one that loaded them (verified directly: a
            # `load atom2` group's atom can appear inside a LATER, different
            # `load [...]` group's `move` blocks, and be `store`d by yet
            # another group's `store`).
            if rest.strip() == "[":
                names, j = _parse_block(lines, i)
            elif rest == "":
                names, j = _parse_block(lines, i)
            else:
                names = [rest.strip()]
                j = i + 1
            for name in names:
                airborne.add(name)
                busy[name_to_idx[name]] += t_transfer
            total += t_transfer
            i = j
        elif kw == "move":
            if rest.strip() == "[" or rest == "":
                items, j = _parse_block(lines, i)
            else:
                items = [rest.strip()]
                j = i + 1
            step_max_dur = 0.0
            for item in items:
                mm = _COORD_RE.match(item)
                x, y, name = float(mm.group(1)), float(mm.group(2)), mm.group(3)
                ox, oy = pos.get(name, (x, y))
                d = math.hypot(x - ox, y - oy)
                pos[name] = (x, y)
                dur_d = movement_duration(d)
                if name in airborne:
                    busy[name_to_idx[name]] += dur_d
                step_max_dur = max(step_max_dur, dur_d)
            # All atoms in one move block travel in parallel under the same
            # AOD; the block's own contribution to the makespan is set by
            # its slowest (longest-displacement) atom, not the sum.
            total += step_max_dur
            i = j
        elif kw == "store":
            # Drop cost -- mirrors ZAP's Deactivate. Closes this atom's
            # transfer event (one pickup + one drop = ZAP/ZAC's convention
            # of 2 atom_transfer factors per moved atom, see score()).
            if rest.strip() == "[":
                names, j = _parse_block(lines, i)
            elif rest == "":
                names, j = _parse_block(lines, i)
            else:
                names = [rest.strip()]
                j = i + 1
            for name in names:
                busy[name_to_idx[name]] += t_transfer
                airborne.discard(name)
                n_transfer_events += 1
                if _in_zone(pos.get(name, (0, 0))):
                    in_zone.add(name)
                else:
                    in_zone.discard(name)
            total += t_transfer
            i = j
        else:
            # cz zone name token, or a stray ']' / unrecognized line -- skip.
            i += 1

    coh_exponent = n_q * total / ARCH["qubit_spec"]["T2"]   # 2026-10-10: no busy-time credit
    f_coh = math.exp(-coh_exponent) if coh_exponent < 700 else 0.0

    fid = ARCH["operation_fidelity"]
    f_2q = fid["two_qubit_gate"]
    f_1q = fid["single_qubit_gate"]
    f_tr = fid["atom_transfer"]
    f_idle = 1 - (1 - f_2q) / 2

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
        "log_fidelity_decoherence": coh_exponent,
    }


def _sig(value, digits=8):
    """Round to significant figures, not decimal places (see run_zac.py)."""
    return float(f"{value:.{digits}g}")


def compile_and_score(name, qasm_path, compiler_factory):
    from mqt.core import load
    from qiskit import QuantumCircuit

    circ = QuantumCircuit.from_qasm_file(qasm_path)
    circ.remove_final_measurements(inplace=True)
    n_q = circ.num_qubits
    compiler = compiler_factory()
    t0 = time.time()
    qcomp = load(circ)
    code = compiler.compile(qcomp)
    wall = time.time() - t0
    rec = parse_naviz(code, n_q)
    rec["algorithm"] = name
    rec["stage"] = None
    rec["compilation_time"] = round(wall, 4)
    rec["timestamp"] = datetime.now().isoformat()
    for k in ("total_fidelity", "fidelity_1q_gate", "fidelity_2q_gate",
              "fidelity_idle", "fidelity_handover", "fidelity_decoherence"):
        rec[k] = _sig(rec[k])
    rec["total_duration"] = round(rec["total_duration"], 6)
    return rec


def main():
    from run_zac import circuits, qasm2_path
    from mqt.qmap.na.zoned import RoutingAwareCompiler, ZonedNeutralAtomArchitecture

    setting = sys.argv[1] if len(sys.argv) > 1 else "qasmbench"
    benchmarks = circuits(setting)

    arch = ZonedNeutralAtomArchitecture.from_json_string(QMAP_ARCH_JSON)
    log_dir = os.path.join(ROOT, "results", "mqt", setting, "log")
    os.makedirs(log_dir, exist_ok=True)

    fids = {}
    for b in benchmarks:
        name = os.path.splitext(os.path.basename(b))[0]
        qasm_path = qasm2_path(b)
        try:
            rec = compile_and_score(name, qasm_path,
                                    lambda: RoutingAwareCompiler(arch))
        except Exception as e:
            print(f"[QMAP] {name:18s} FAILED: {type(e).__name__}: {str(e)[:150]}")
            continue
        path = os.path.join(log_dir, f"{name}.json")
        data = json.load(open(path)) if os.path.exists(path) else []
        data.append(rec)
        json.dump(data, open(path, "w"), indent=2)
        fids[name] = rec["total_fidelity"]
        print(f"[QMAP] {name:18s} F={rec['total_fidelity']:.6f}  "
              f"({rec['compilation_time']:.3f}s)")

    print(f"[QMAP] total fidelity over {len(fids)} circuits: {sum(fids.values()):.6f}")


if __name__ == "__main__":
    main()
