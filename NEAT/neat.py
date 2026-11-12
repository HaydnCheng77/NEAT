import ast
import json
import math
import operator
import os
import re
import time
import warnings
from datetime import datetime
from qiskit import transpile, QuantumCircuit
from qiskit.qasm3 import QASM3ImporterError, loads_experimental
from qiskit_qasm3_import import parse

from NEAT.scheduler import Scheduler
from NEAT.router import Router
from NEAT.simulator import Simulator
from NEAT.animator import Animator
from NEAT.verifier import Verifier


# --- OpenQASM 3 loading without the ANTLR parse tree (2026-10-11) -------------
# ``qiskit_qasm3_import.parse`` builds an ANTLR parse tree that becomes cyclic
# garbage the moment it returns: 28M objects on qft_n1000 (126 s to parse, 49 s
# for the cyclic GC to free). Whenever that collection fired inside the timed
# compile it was charged to the compiler (see tools/regen/worker.py Timer).
# Qiskit's native Rust importer (``qiskit.qasm3.loads_experimental``) leaves no
# Python garbage and parses qft_n1000 in 23 s, but it cannot evaluate constant
# gate arguments such as ``pi/1024``. ``load_qasm3`` therefore folds every
# parenthesised, purely constant argument list (numbers, ``pi``, + - * /) into
# float literals -- the same float operations the ANTLR importer performs --
# and hands the result to the native importer. Verified to build the identical
# circuit (gate names, float-identical parameters, qubit and clbit indices) on
# every OpenQASM 3 benchmark; anything the native importer rejects falls back to
# the ANTLR importer unchanged. baselines/zap/zap.py has the same code.
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv}


def _const(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in ("pi", "π"):
        return math.pi
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _const(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_const(node.left), _const(node.right))
    raise ValueError("not a constant expression")


def _fold(m):
    try:
        args = [_const(ast.parse(a.strip(), mode="eval").body) for a in m.group(1).split(",")]
    except (ValueError, SyntaxError):
        return m.group(0)
    return "(" + ", ".join(repr(v) for v in args) + ")"


def load_qasm3(qasm_str):
    """QuantumCircuit from OpenQASM 3 source; native importer, ANTLR fallback."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # ExperimentalWarning
            return loads_experimental(re.sub(r"\(([^()]*)\)", _fold, qasm_str))
    except QASM3ImporterError:
        return parse(qasm_str)


# The compiler runs ONE fixed configuration, compiled once per circuit — Zap
# always compiles self.scheduling_strategy / self.architecture exactly once.
# `pair_tiebreak` and `initial_mapping_parallel_lookahead` used to be listed
# here as the fixed config's values; both are gone from the config surface
# entirely as of 2026-09-13 (`pair_tiebreak` hardcoded into `_place_gate`,
# `initial_mapping_parallel_lookahead` deleted outright — see the removal
# notes in `Placer.__init__` and `_place_gate`), so there is nothing left to
# list here. The remaining fallback defaults still live in
# `Placer.__init__`'s `routing_cfg.get(...)` calls.
#
# This set also chose cost_quantum=0.0 and future_travel_weight=0.0. Both
# knobs, and the `_quantise` / `_future_travel_cost` code they gated, were
# REMOVED 2026-09-05: at 0.0 each was provably inert (`_quantise` returned its
# argument unchanged for any quantum <= 0; the travel term was weight * min(),
# identically zero), so deleting them measured bit-identical — tqe 6.906136
# with 14/14 circuits unchanged, tqe2 6.732359, both physical gates passing.
# See the two `REMOVED 2026-09-05` comments in Placer.__init__ for what the
# travel term modelled, should it ever be worth reviving.
#
# ``storage_window`` (the hot/cold/windowed storage-site split it gated was
# collapsed into one unconditional rank 2026-09-11 — see
# `_assign_storage_site_ranked`'s docstring) and ``idle_cost_alpha`` (hardcoded
# to 1.0, its shipped default — not a behavior change) were both REMOVED from
# the placer 2026-09-12/13 as knobs nothing shipped ever varied meaningfully;
# `pair_candidate_limit` stays at its placer.py default, 4. This scores
# 6.906136 on tqe (the number predates the 2026-09-09 through 2026-09-13
# placer/architecture changes recorded in git history). The sweep tool that originally chose this
# config, `tools/sweep_configs.py`, was itself deleted 2026-09-13 once every
# knob it swept was either removed or made unconditional — recover it from
# git history if a similar sweep is needed again.
#
# The config this grid selected also included gate_order_greedy=True with
# gate_order_greedy_max_k=12; the greedy gate order was REMOVED 2026-09-03
# (-0.005081 on tqe, accepted) — see the block comment in Placer.__init__ for
# the five-suite measurement behind that. The rest of the grid's choices stand.
#
# CAVEAT: there is no per-circuit fallback. This set was chosen by search over
# tqe plus a 10-circuit held-out check, and on that held-out set it was the
# WORST of six candidates on one circuit (random_n60) despite winning the
# total. The placer is chaotically sensitive circuit-to-circuit; a future
# circuit that looks like random_n60 will compile worse than a per-circuit
# search would have gotten it, with no signal that it happened.


def _sig(value, digits=8):
    """Round to ``digits`` SIGNIFICANT figures, not decimal places.

    ``round(x, 8)`` writes any fidelity below 1e-8 as exactly 0.0. On the tqe
    suite nothing goes near that, but a 300-qubit random circuit lands at
    F ~ 1e-14 and the whole record — every per-cause factor as well as the
    total — was being logged as zero, so the scalability plots had to drop
    those points as "underflowed" when the numbers were in fact perfectly
    representable. Significant figures keep the same precision on O(1) values
    and stay meaningful all the way down to denormals.
    """
    return float(f"{value:.{digits}g}")


class Neat:
    """Compile a circuit benchmark onto a zoned neutral-atom architecture."""

    # How many past runs each results/<dir>/log/<circuit>.json keeps. See
    # write_log: the file is rewritten in full on every compile, so this bounds
    # a cost that used to grow without limit. Consumers only ever read the last
    # entry; the rest is history for eyeballing.
    LOG_HISTORY = 20

    def __init__(
            self,
            benchmark: str,
            architecture: dict,
            initial_mapping: list,
            output_dir: str = "",
            scheduling_strategy: str = "alap_separate",
            placement_strategy: str = "baseline",
            routing_strategy: str = "baseline",
    ):
        """
        Args:
            benchmark: Path under ``benchmark/`` (with extension), e.g. ``qft/qft_n10.qasm``.
            architecture: Parsed JSON from ``baselines/zap/architectures/<file>.json`` or
                ``NEAT/architecture.py``.
            initial_mapping: Optional per-qubit (x, y) sites; empty list selects automatic layout.
            output_dir: Name of the subdirectory under ``results/`` for artifacts.
            scheduling_strategy: only ``alap_separate`` is implemented in this
                ``NEAT`` tree — ``asap_separate``/``asap_joint``/
                ``mid_separate`` were removed 2026-09-13 (see
                ``scheduler.py``'s removal note); the baseline ``zap``
                compiler still has its own separate implementations.
            placement_strategy: Accepted for call-site compatibility with
                the baseline ``zap`` compiler (which still reads it) — not read anywhere in this ``NEAT``
                tree; ``Router`` here never receives it.
            routing_strategy: Stored and printed for logging/CLI display only
                (see ``Router.__init__``'s 2026-09-13 note) — not forwarded
                to ``Placer``, which never accepted this parameter.
        """
        self.benchmark = benchmark
        self.architecture = architecture
        self.given_initial_mapping = initial_mapping
        self.output_dir = output_dir
        self.stg_slm_sites = []
        self.ent_slm_sites = []
        self.scheduling_strategy = scheduling_strategy
        self.routing_strategy = routing_strategy
        self.parse_benchmark()
        self.results_code = {
            'benchmark': self.benchmark_name,
            'output_dir': output_dir,
            'compilation_time': 0,
            'n_q': 0,
            'n_1q_gate': 0,
            'n_2q_gate': 0,
            'stages': {},
            'instructions': []
        }
        self.parse_slm_sites()
        print(
            f"[INFO] ZAP: architecture name={self.architecture.get('name', '?')!r}, "
            f"storage_traps={len(self.stg_slm_sites)}, entanglement_traps={len(self.ent_slm_sites)}"
        )
        self.set_program()

    def parse_benchmark(self):
        """Derive directory, stem, and file type from ``self.benchmark``."""
        benchmark_path = self.benchmark.replace("\\", "/")
        self.benchmark_dir, benchmark_filename = os.path.split(benchmark_path)
        self.benchmark_name, benchmark_ext = os.path.splitext(benchmark_filename)
        self.benchmark_type = benchmark_ext.lstrip(".")
        self.benchmark_stem = (
            f"{self.benchmark_dir}/{self.benchmark_name}"
            if self.benchmark_dir
            else self.benchmark_name
        )

    def get_sites(self, slm: dict):
        """Return all trap coordinates for one SLM block (grid from ``location``, ``r``, ``c``, ``site_seperation``)."""
        x, y = slm['location']
        sep_x, sep_y = slm['site_seperation']

        return [
            (x + i * sep_x, y + j * sep_y)
            for j in range(slm['r'])
            for i in range(slm['c'])
        ]

    def parse_slm_sites(self):
        """Populate storage and entanglement trap coordinates from the architecture JSON."""
        for zone in self.architecture['storage_zones']:
            for slm in zone['slms']:
                self.stg_slm_sites += self.get_sites(slm)
        self.stg_slm_sites = list(set(self.stg_slm_sites))

        for zone in self.architecture['entanglement_zones']:
            for slm in zone['slms']:
                self.ent_slm_sites += self.get_sites(slm)
        self.ent_slm_sites = list(set(self.ent_slm_sites))

    def set_program(self):
        """Load gates from QASM (transpiled to CZ+U) or JSON edge list; fill ``g_q`` and gate counts."""
        self.n_2q_gate = 0
        self.n_1q_gate = 0
        self.g_q = []
        benchmark_file = os.path.join(
            "benchmark",
            self.benchmark_dir,
            f"{self.benchmark_name}.{self.benchmark_type}"
        )
        with open(benchmark_file, 'r') as f:
            if self.benchmark_type == "qasm":
                qasm_str = f.read()
                if "OPENQASM 2" in qasm_str:
                    circuit = QuantumCircuit.from_qasm_str(qasm_str)
                elif "OPENQASM 3" in qasm_str:
                    circuit = load_qasm3(qasm_str)
                else:
                    raise ValueError("Unsupported QASM version detected.")

                # Strip trailing swaps left by Qiskit decomposition (not native to the
                # atom model). F2 (2026-09-02): guarded the empty/all-swap case, which
                # previously indexed circuit.data[-1] unconditionally and raised
                # IndexError; switched to .operation.name for consistency with the
                # CircuitInstruction access used a few lines below (inst.operation.name
                # at line 212) instead of the deprecated tuple-indexing form. This only
                # strips *trailing* swaps — a swap anywhere else in the circuit is
                # intentionally left as an ordinary 2q gate below, since fixing that
                # would change which gates route as 2q interactions and is a
                # behavioural change, not a correctness fix.
                while circuit.data and circuit.data[-1].operation.name == 'swap':
                    circuit.data.pop()

                n_pre = circuit.num_qubits
                # optimization_level 3 unconditionally (2026-09-03). This used
                # to step down with circuit width — 3/2/1/0 at 24/64/128/more —
                # under a comment claiming high levels take "minutes+" on wide
                # QFT-style circuits. Measured, that is simply not true for
                # anything in this repo's benchmarks: transpiling at level 3
                # costs 0.31 s on qft_n250 and 0.06 s on adder_n316, the two
                # widest circuits here (250 and 316 qubits).
                #
                # It was also actively harmful, because the level decides how
                # big a circuit the compiler is handed:
                #
                #   circuit        L0 2q    L1 2q   L2 2q   L3 2q   old level
                #   qft_n100        9900     5344    3481    3481   L1
                #   qft_n150       22350     8544    5431    5431   L1
                #   qft_n250       62250    14944    9331    9331   L0
                #   random_n300     1500     1500    1500    1500   L0
                #   adder_n316      2275     2275    2275    2275   L0
                #
                # So qft_n250 was being routed with 6.7x more two-qubit gates
                # than necessary, purely from preprocessing. L2 and L3 are
                # identical on every benchmark measured (as are L1/L2/L3 on the
                # <=64-qubit suites, which is why tqe/tqe2/graphs are unchanged
                # by this); 3 is chosen so nothing silently steps down again.
                #
                # This also removes the ZAC comparison's worst confound — see
                # baselines/run_zac.py, whose prepare_qasm2 pins ZAC to whatever
                # level this function picks. ZAC always transpiled at 3.
                opt_level = 3
                print(
                    f"[INFO] ZAP: QASM parsed ({n_pre} qubits), transpiling to CZ basis "
                    f"(optimization_level={opt_level}, may take a while)…"
                )
                cz_circuit = transpile(
                    circuit,
                    basis_gates=["cz", "id", "u2", "u1", "u3"],
                    optimization_level=opt_level,
                    seed_transpiler=0
                )
                print(
                    f"[INFO] ZAP: Transpile done — {cz_circuit.num_qubits} qubits, "
                    f"{len(cz_circuit.data)} operations in DAG"
                )
                instruction = cz_circuit.data
                self.results_code['n_q'] = cz_circuit.num_qubits
                for inst in instruction:
                    if inst.operation.num_qubits == 2:
                        self.results_code['n_2q_gate'] += 1
                        # Global qubit index (2026-10-06). ``_index`` is the index
                        # inside the qubit's own register: on multi-register
                        # circuits (tqe sat_n11, qram_n20) q[0] and anc[0] both
                        # became qubit 0, so CZs between them were routed as 1q
                        # gates (12 and 28 CZ missing) and others hit the wrong
                        # qubits. ZAC/QMAP always used global indices.
                        self.g_q.append((cz_circuit.find_bit(inst.qubits[0]).index,
                                         cz_circuit.find_bit(inst.qubits[1]).index))
                    elif inst.operation.name != "measure" and inst.operation.name != "barrier":
                        self.results_code['n_1q_gate'] += 1
                        _q = cz_circuit.find_bit(inst.qubits[0]).index
                        self.g_q.append((_q, _q))
            elif self.benchmark_type == "json":
                # F2 note: n_q is inferred as (max qubit index seen) + 1, unlike the
                # QASM path above which reads n_q from circuit metadata. A qubit with
                # no gates at all — including a high-index qubit that never appears in
                # any edge — is invisible here and silently excluded from n_q. Left
                # as-is: this is the format every settings/graphs.json benchmark
                # relies on today (dense edge lists with no isolated qubits), and
                # asserting the contrary would need to distinguish "not implemented"
                # from "a real high-index isolated qubit exists" without cluttering
                # the format itself.
                graphs = json.load(f)
                for q0, q1 in graphs:
                    self.results_code['n_q'] = max(self.results_code['n_q'], q0, q1)
                    if q0 == q1:
                        self.g_q.append((q0, q1))
                        self.results_code['n_1q_gate'] += 1
                    else:
                        self.g_q.append((q0, q1))
                        self.results_code['n_2q_gate'] += 1
                self.results_code['n_q'] += 1
            else:
                raise ValueError("Unsupported benchmark file type.")


    def log_results(self, simulator: Simulator):
        """Append one simulation summary record to ``results/<output_dir>/log/<benchmark>.json``."""
        result = {
            "algorithm": self.benchmark_name,
            "n_qubits": simulator.n_q,
            "stage": self.current_stage,
            "n_1q_gate": simulator.n_1q_gate,
            "n_2q_gate": simulator.n_2q_gate,
            "total_duration": round(getattr(simulator, 'total_duration', 0), 6),
            "compilation_time": round(simulator.results_code['compilation_time'], 4),
            "total_fidelity": _sig(simulator.cir_fidelity),
            "fidelity_1q_gate": _sig(simulator.cir_fidelity_1q_gate),
            "fidelity_2q_gate": _sig(simulator.cir_fidelity_2q_gate),
            "fidelity_idle": _sig(simulator.cir_fidelity_2q_gate_for_idle),
            "fidelity_handover": _sig(simulator.cir_fidelity_atom_transfer),
            "fidelity_decoherence": _sig(simulator.cir_fidelity_coherence),
            # Log-space decoherence loss (nats), never underflows -- mirrors
            # baselines/run_zac.py's log_fidelity_decoherence field. Use it
            # whenever fidelity_decoherence has hit exactly 0.0 (see
            # simulator.py's 2026-09-12 underflow fix).
            "log_fidelity_decoherence": simulator.log_fidelity_coherence,
            "timestamp": datetime.now().isoformat()
        }
        ver = self.results_code.get('verification')
        if ver is not None:
            result["aod_legal"] = ver['legal']
            result["aod_violations"] = {r: n for r, n in ver['counts'].items() if n}

        # Phase-0 instrumentation (wave-count reframing investigation): only
        # present when routing.collect_stats=true, so this costs nothing on
        # normal runs. Log-fidelity loss is -log(per-category fidelity factor)
        # so the four categories sum exactly to -log(total fidelity), mirroring
        # the paper's Fig. 7 breakdown using the SAME multiplicative factors
        # Simulator already computes (no parallel fidelity model).
        stats = self.results_code.get('stats')
        if stats is not None:
            result["stats"] = stats
            result["compile_time_s"] = round(self.results_code['compilation_time'], 6)
            result["logloss"] = {
                "gate_1q": -math.log(simulator.cir_fidelity_1q_gate),
                "gate_2q": -math.log(simulator.cir_fidelity_2q_gate),
                "crosstalk": -math.log(simulator.cir_fidelity_2q_gate_for_idle),
                "transfer": -math.log(simulator.cir_fidelity_atom_transfer),
                # cir_fidelity_coherence underflows to exactly 0.0 on wide
                # circuits (qft n>=800, adder n=1000), where -math.log() raises.
                # The simulator already carries the log-space exponent for
                # precisely this case (see simulator.log_fidelity_coherence); use it whenever the linear value
                # has underflowed, so large-n circuits log a real number instead
                # of crashing the compile.
                "decoherence": (-math.log(simulator.cir_fidelity_coherence)
                                if simulator.cir_fidelity_coherence > 0
                                else simulator.log_fidelity_coherence),
            }

        os.makedirs(f"results/{self.output_dir}/log/", exist_ok=True)
        file_path = f"results/{self.output_dir}/log/{self.benchmark_name}.json"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
        else:
            data = []
        data.append(result)
        # Keep only the most recent LOG_HISTORY runs. This file is append-only
        # history and every consumer reads [-1] (suite_report.py, the floor
        # checks, the comparison histogram), but it is re-read and re-written in
        # full on every compile: by 2026-09-03 sat_n11.json held 121 entries /
        # 700 KB and the tqe suite's writes alone were ~1.2 s of a 2.3 s run,
        # growing without bound. The tail stays long enough to eyeball recent
        # history and diff against a few runs back; `suite_report.py --save`
        # is the right tool for a snapshot you actually want to keep.
        if len(data) > self.LOG_HISTORY:
            data = data[-self.LOG_HISTORY:]
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)


    def _blank_results_code(self):
        """A fresh ``results_code`` shell to compile into."""
        return {
            'benchmark': self.benchmark_name,
            'output_dir': self.output_dir,
            'compilation_time': 0,
            'n_q': self.results_code['n_q'],
            'n_1q_gate': self.results_code['n_1q_gate'],
            'n_2q_gate': self.results_code['n_2q_gate'],
            'stages': {},
            'instructions': []
        }

    def _compile_once(self, scheduling_strategy: str, architecture: dict):
        """Schedule and route the circuit. Returns ``(results_code, n_stage)``.

        Writes nothing to disk — ``solve`` emits the artifacts once this returns.
        """
        results_code = self._blank_results_code()

        tmp = time.time()
        scheduler = Scheduler(g_q=self.g_q, results_code=results_code)
        if scheduling_strategy == "alap_separate":
            scheduler.alap_separate()
        else:
            raise ValueError(
                f"[Error] Unsupported scheduling strategy {scheduling_strategy!r} "
                "— this NEAT tree only implements alap_separate "
                "(asap_separate/asap_joint/mid_separate removed 2026-09-13)")
        results_code['compilation_time'] = time.time() - tmp

        list_gate = [[self.g_q[i] for i in gates] for gates in scheduler.list_scheduling]

        router = Router(
            slm_sites=[self.stg_slm_sites, self.ent_slm_sites],
            results_code=results_code,
            list_full_gates=list_gate,
            qubit_mapping=self.given_initial_mapping,
            architecture=architecture,
            routing_strategy=self.routing_strategy,
            write_code=False,
        )
        return router.results_code, len(list_gate)

    def solve(self, simulation: bool = False, animation: bool = False):
        """Schedule → route (placement + moves) → optional fidelity sim and MP4 animation."""
        print(f"[INFO] ZAP: Start solving {self.benchmark}")
        nq = self.results_code["n_q"]
        n_stg = len(self.stg_slm_sites)
        if nq > n_stg:
            raise ValueError(
                f"[Error] Circuit needs {nq} qubits after transpile but this architecture "
                f"defines only {n_stg} storage traps. For large QFT circuits set "
                f"\"architecture\": \"scale_to_500.json\" in the setting JSON "
                f"(see setting/scalability/qft.json)."
            )

        print(f"[INFO] ZAP: Start placing and routing with "
              f"\n\t\trouting strategy: {self.routing_strategy}")

        self.results_code, self.current_stage = self._compile_once(
            self.scheduling_strategy, self.architecture)

        if os.environ.get("ZAP_PEEPHOLE", "1" if os.environ.get("ZAP_STRICT", "1") == "1" else "0") == "1":
            self._merge_transfers(self.results_code)

        # AOD-legality audit (2026-10-02). Runs after _compile_once, like the
        # simulator, so it is not part of compilation_time. Its counts go into
        # the trace (results_code['verification']) and the log record.
        ds = float((self.architecture.get('routing') or {}).get('min_tone_sep', 2.0) or 0.0)
        verifier = Verifier(results_code=self.results_code,
                            traps=list(self.stg_slm_sites) + list(self.ent_slm_sites),
                            d_s=ds)
        print(f"[INFO] ZAP: Verifier: {self.benchmark}: {verifier.summary()}")

        simulator = None
        if simulation:
            simulator = Simulator(results_code=self.results_code,
                                  architecture=self.architecture)

        self.write_code()
        if simulator is not None:
            self.log_results(simulator)
        if (not verifier.legal and os.environ.get("ZAP_STRICT", "1") == "1"
                and os.environ.get("ZAP_STRICT_RAISE", "1") == "1"):
            # Strict mode promises legal traces: refuse loudly (the trace is
            # already written, so it can be inspected).
            raise RuntimeError(f"[Error] {self.benchmark}: illegal trace -- "
                               f"{verifier.summary()}")

        if animation:
            print("[INFO] ZAP: Start animation")
            Animator(
                slm_sites=[self.stg_slm_sites, self.ent_slm_sites],
                results_code=self.results_code,
                architecture=self.architecture
            )

        print(f"[INFO] ZAP: Finish solving {self.benchmark}\n")

    @staticmethod
    def _merge_transfers(results_code):
        """Peephole pass (2026-10-03): fuse back-to-back Activates, and
        back-to-back Deactivates, into one instruction each.

        Lifting A and then B leaves exactly the lit grid of lifting A and B
        together, which the router already ensured is clean; releasing A and
        then B is physically the same as releasing both at once. The
        simulator charges transfers per atom and busy time per atom, so the
        only change is one fewer 15 us instruction per fusion -- total
        duration, hence decoherence, can only go down. Never fuses two
        instructions that touch the same atom.
        """
        # 1) Drop lifts that never move: an atom Activated and then
        #    Deactivated with no move of it in between was lifted for nothing.
        #    Removing it saves two atom transfers. Fewer lit lines remove
        #    constraints, BUT the router sometimes lifts an atom precisely so
        #    that it is held (not a bystander) when another Activate lights a
        #    row x column crossing on its trap: dropping that lift turned the
        #    crossing into a ghost pick-up (adder_n298 q14, 2026-10-06). So an
        #    atom is kept lifted if, at any Activate during its hold, the
        #    OTHER held atoms' lines cross at its position.
        insts = results_code['instructions']
        open_act = {}          # atom -> (instruction index) of its live Activate
        moved = set()
        shield = set()         # held, unmoved atoms whose lift prevents a ghost
        pos = {}
        held = set()
        drop = {}              # instruction index -> atoms to remove from it
        for i, inst in enumerate(insts):
            t = inst['type']
            if t == "Init":
                for l in inst['locs']:
                    pos[l['id']] = (l['x'], l['y'])
            elif t == "Activate":
                for q in inst['qs']:
                    open_act[q] = i
                    moved.discard(q)
                    shield.discard(q)
                held.update(inst['qs'])
                for q in held:
                    if q in moved or q in shield:
                        continue
                    x, y = pos[q]
                    if (any(pos[h][0] == x for h in held if h != q)
                            and any(pos[h][1] == y for h in held if h != q)):
                        shield.add(q)
            elif t in ("BigMove", "Park"):
                for l in inst['locs']:
                    if (l['x_begin'], l['y_begin']) != (l['x_end'], l['y_end']):
                        moved.add(l['id'])
                    pos[l['id']] = (l['x_end'], l['y_end'])
            elif t == "Deactivate":
                for q in inst['qs']:
                    a = open_act.pop(q, None)
                    if a is not None and q not in moved and q not in shield:
                        drop.setdefault(a, set()).add(q)
                        drop.setdefault(i, set()).add(q)
                    held.discard(q)
                    shield.discard(q)
        if drop:
            kept = []
            for i, inst in enumerate(insts):
                d = drop.get(i)
                if d:
                    idx = [k for k, q in enumerate(inst['qs']) if q not in d]
                    if not idx:
                        continue
                    inst['qs'] = [inst['qs'][k] for k in idx]
                    inst['duration'] = [inst['duration'][k] for k in idx]
                    inst['locs'] = [inst['locs'][k] for k in idx]
                kept.append(inst)
            results_code['instructions'] = kept
        # 2) Fuse back-to-back Activates / Deactivates.
        out = []
        for inst in results_code['instructions']:
            t = inst['type']
            if (out and t in ("Activate", "Deactivate") and out[-1]['type'] == t
                    and not set(out[-1]['qs']) & set(inst['qs'])):
                prev = out[-1]
                prev['qs'] = list(prev['qs']) + list(inst['qs'])
                prev['duration'] = list(prev['duration']) + list(inst['duration'])
                prev['locs'] = list(prev['locs']) + list(inst['locs'])
                continue
            out.append(inst)
        results_code['instructions'] = out

    def write_code(self):
        """Write the winning compile's instruction trace to ``results/.../code/``."""
        output_path = os.path.join(
            "results", self.results_code['output_dir'], "code",
            f"{self.results_code['benchmark']}_code.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.results_code, f)
