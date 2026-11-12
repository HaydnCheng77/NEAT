# NEAT

NEAT is a compiler for zoned neutral-atom quantum computers. It schedules a
circuit into stages, places atoms with transport-informed placement, and moves
them with partial-progress routing on a per-circuit generated array. Every
trace it emits is checked for AOD legality.

This repository holds the compiler, the baselines it is compared with, the
benchmark circuits, and everything needed to rebuild the paper's evaluation.

## Layout

| folder | contents |
|---|---|
| `NEAT/` | the compiler, one module per stage: `run.py` (command line), `neat.py` (entry point, class `Neat`; circuit loading), `architecture.py` (per-circuit array generator), `scheduler.py` (preprocessing), `placer.py`, `router.py`, `transport.py` (AOD move model), `simulator.py` (fidelity model), `verifier.py` (AOD legality), `animator.py` |
| `baselines/` | the compilers NEAT is compared with (see [Baselines](#baselines)): `zap/`, `zac/`, `powermove/` (MQT comes from pip), and a driver for each: `run_zap.py`, `run_zac.py`, `run_mqt.py`, `run_powermove.py` |
| `benchmark/` | the circuits, one folder per suite: `qasmbench/` (10 QASMBench circuits) and `scalability/{graphs,random,adder,qft}/` (25 sizes each, n = 10..1000, each family with its `generate.py`); `transpiled/` (every circuit after the shared Qiskit transpile, see below); `suites.py` (lists a suite's circuits) |
| `experiments/` | one folder per result, each with its scripts (and `data/`): `end_to_end/` compiles every circuit with every compiler, scores all traces under one fidelity model and checks their AOD legality (Table I, Fig. 6); `placer/` compares five placers under NEAT's router (Table II); `router/` compares routers on the same placements (Table III). All three use the same 110 circuits |
| `results/` | the paper's compiled results: `results/<compiler>/<suite>/log/<circuit>.json` (compilers `neat`, `zap`, `zac`, `mqt`, `powermove`; fidelity, duration, gate counts, compile time, AOD legality) and `results/end_to_end/jobs/` (one record per compile: status, CPU time, calibration). The instruction traces (`code/`, `events/`, `native/`) are not in git (5 GB); compiling regenerates them |

## Requirements

- Python with the packages in `requirements.txt` (`pip install -r requirements.txt`).
  Tested with Python 3.14 and exactly these versions; the compiled circuits
  depend on the Qiskit version.
- `zsh`, for `experiments/placer/run_table.sh` and `experiments/router/run_table.sh`.
- `ffmpeg`, only for `--animation`.

## Usage

```bash
python3 NEAT/run.py qasmbench               # compile a suite with NEAT -> results/neat/qasmbench/
python3 NEAT/run.py scalability/graphs      # any suite: a folder under benchmark/
python3 baselines/run_zap.py qasmbench      # the baseline ZAP -> results/zap/qasmbench/
```

Each compile is simulated (fidelity under the paper's model) and checked by
NEAT's AOD verifier (`NEAT/verifier.py`); both results are in the log,
`results/<compiler>/<suite>/log/<circuit>.json` (`total_fidelity`,
`total_duration`, `aod_legal`, ...).

Any compiler on one circuit, scored exactly as in the paper (compiler
`neat`, `zap`, `zac`, `mqt` or `powermove`):

```bash
python3 experiments/end_to_end/worker.py zac qasmbench bv_n14 results/end_to_end/jobs/zac__qasmbench__bv_n14.json
python3 experiments/end_to_end/rescore_all.py zac    # -> results/zac/qasmbench/log/bv_n14.json
```

`baselines/run_zac.py`, `run_powermove.py` and `run_mqt.py` also compile whole
suites on their own, but score with each compiler's own durations, so their
numbers differ from the paper's by a few percent; the paper uses
`experiments/end_to_end/` throughout.

Every compiler routes the same circuit: transpiled at Qiskit level 3 with a
fixed seed. NEAT and ZAP do this when they load a circuit; ZAC, MQT and
PowerMove read the result from `benchmark/transpiled/<suite>/<circuit>.qasm`
(QASM 2). These files are the exact inputs of the paper; `prepare_qasm2` in
`baselines/run_zac.py` rebuilds a missing one (identical under the pinned
Qiskit). After editing a circuit, delete its transpiled file.

## Baselines

- **ZAP** (`baselines/zap/`, arXiv:2411.14037,
  [BAQIS-Quantum/neutral-atom-compilation](https://github.com/BAQIS-Quantum/neutral-atom-compilation);
  MIT, see `baselines/zap/LICENSE`): the compiler NEAT is forked
  from, kept as the baseline and changed only where a fair comparison needs
  it: correct qubit indices on multi-register circuits, the same level-3
  transpile as every other compiler, the paper's fidelity model (in log space,
  so large circuits do not underflow) and a faster occupancy test with
  identical output. It runs on fixed arrays, `baselines/zap/architectures/`
  (`default.json` for QASMBench, `scale_to_500_fixed.json` for the
  scalability suites; `CONFIG` in `baselines/run_zap.py`); NEAT generates its
  own array per circuit.
- **ZAC** (`baselines/zac/`, HPCA 2025): BSD 3-Clause, see
  `baselines/zac/LICENSE`. Run with `resyn: False` so it routes the shared
  transpiled circuit unchanged.
- **PowerMove** (`baselines/powermove/`, ASPLOS 2025): `mvqc.py`, `placer/` and
  `scheduler/` are the original release (without the later "CZ throttling",
  which enumerates every simple cycle of a stage's move-dependency graph and
  does not finish at n >= 850); `Construct_Circuit_upstream.py` is the later
  release's gate blocking, which respects gate dependencies.
- **MQT** (`mqt.qmap` 3.8.0 from pip): the routing-aware zoned compiler.

Table II compares five placers under NEAT's router, and Table III compares
routers on the same placements. `experiments/placer/run_stage_candidates.py`
adapts the ZAC and PowerMove placers to NEAT's interface; the MQT placer and
router are ported from mqt-qmap 3.8.0 in `experiments/placer/run_five_placers.py` and
`experiments/router/run_router_compare.py` (see their docstrings).

## License

NEAT is released under the MIT License (`LICENSE`). The baselines keep their
own licenses.

## Reproducing the evaluation

| paper item | data | how to regenerate |
|---|---|---|
| Table I (QASMBench), Fig. 6 (scalability) | `results/<compiler>/<suite>/log/<circuit>.json` | `experiments/end_to_end/` (below) |
| Table II (placers and objectives) | `experiments/placer/data/` | `zsh experiments/placer/run_table.sh` |
| Table III (routers) | `experiments/router/data/{waypoints,no_waypoints}/` | `zsh experiments/router/run_table.sh` |

```bash
python3 experiments/end_to_end/driver.py         # 5 compilers x 110 circuits, one at a time, resumable
python3 experiments/end_to_end/rescore_all.py    # score every trace under the paper's fidelity model
python3 experiments/end_to_end/audit.py          # gate-count parity, timing and legality checks
```

- On its first run the driver clears the paper suites' results in `results/`,
  including the logs shipped with this repository.
  Kill and restart it at any time; it resumes from `results/end_to_end/`.
- The full run takes many hours: PowerMove and ZAC are slow on the larger
  circuits. Compile times are CPU seconds normalised by a calibration benchmark
  run before and after each job (`experiments/end_to_end/worker.py`), so they match the
  paper's only approximately on other hardware; fidelities and durations are
  deterministic.
