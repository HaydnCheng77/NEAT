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

from zap.scheduler import Scheduler
from zap.router import Router
from zap.simulator import Simulator
from zap.animator import Animator


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
# the ANTLR importer unchanged. NEAT/neat.py has the same code.
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


class Zap:
    """Compile a circuit benchmark onto a zoned neutral-atom architecture."""

    def __init__(
            self,
            benchmark: str,
            architecture: dict,
            initial_mapping: list,
            output_dir: str = "",
            scheduling_strategy: str = "asap_separate",
            placement_strategy: str = "baseline",
            routing_strategy: str = "baseline"
    ):
        """
        Args:
            benchmark: Path under ``benchmark/`` (with extension), e.g. ``qft/qft_n10.qasm``.
            architecture: Parsed JSON from ``architecture/<file>.json``.
            initial_mapping: Optional per-qubit (x, y) sites; empty list selects automatic layout.
            output_dir: Name of the subdirectory under ``results/`` for artifacts.
            scheduling_strategy: ``asap_separate`` or ``asap_joint``.
            placement_strategy: Passed to the router (reserved / experimental names warn and fall back).
            routing_strategy: Idle-qubit policy and related routing label for the placer/router.
        """
        self.benchmark = benchmark
        self.architecture = architecture
        self.given_initial_mapping = initial_mapping
        self.output_dir = output_dir
        self.stg_slm_sites = []
        self.ent_slm_sites = []
        self.scheduling_strategy = scheduling_strategy
        self.placement_strategy = placement_strategy
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

                # Strip trailing swaps left by Qiskit decomposition (not native to the atom model).
                swap_remain = True
                while swap_remain:
                    if circuit.data[-1][0].name == 'swap':
                        circuit.data.pop()
                    else:
                        swap_remain = False

                n_pre = circuit.num_qubits
                # optimization_level 3 unconditionally (2026-09-03), matching
                # zap_modified — the baseline has to preprocess identically or
                # the two panels are not comparing routing. The width-stepped
                # levels this replaces cost qft_n250 6.7x more two-qubit gates
                # for no measurable transpile-time saving; see the block comment
                # in zap_modified/zap.py for the measurements.
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
            "log_fidelity_decoherence": simulator.log_fidelity_coherence,
            "timestamp": datetime.now().isoformat()
        }
        
        os.makedirs(f"results/{self.output_dir}/log/", exist_ok=True)
        file_path = f"results/{self.output_dir}/log/{self.benchmark_name}.json"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
        else:
            data = []
        data.append(result)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)


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

        print(f"[INFO] ZAP: Start {self.scheduling_strategy} scheduling")
        tmp = time.time()
        scheduler = Scheduler(
            g_q=self.g_q,
            results_code=self.results_code
            )
        
        if self.scheduling_strategy == "asap_separate":
            scheduler.asap_separate()
        elif self.scheduling_strategy == "asap_joint":
            scheduler.asap_joint()
        else:
            raise ValueError("[Error] Unsupported scheduling strategy")

        self.results_code['compilation_time'] = time.time() - tmp
        list_gate = []
        self.current_stage = 0
        for gates in scheduler.list_scheduling:
            tmp = [self.g_q[i] for i in gates]
            list_gate.append(tmp)
            self.current_stage += 1

        print(f"[INFO] ZAP: Start placing and routing with \n\t\tplacement strategy: {self.placement_strategy} \n\t\trouting strategy: {self.routing_strategy}")
        
        router = Router(
            slm_sites=[self.stg_slm_sites, self.ent_slm_sites],
            results_code=self.results_code,
            list_full_gates=list_gate,
            qubit_mapping=self.given_initial_mapping,
            architecture=self.architecture,
            placement_strategy=self.placement_strategy,
            routing_strategy=self.routing_strategy
        )
        self.results_code = router.results_code

        if simulation:
            print("[INFO] ZAP: Start simulation")
            simulator = Simulator(
                results_code=self.results_code,
                architecture=self.architecture
            )
            self.log_results(simulator)

        if animation:
            print("[INFO] ZAP: Start animation")
            Animator(
                slm_sites=[self.stg_slm_sites, self.ent_slm_sites],
                results_code=self.results_code,
                architecture=self.architecture
            )

        print(f"[INFO] ZAP: Finish solving {self.benchmark}\n")
