"""
qaoa_adapter.py — Adapter for QAOA / TSVF-QAOA experiments.

Maps QAOA (Quantum Approximate Optimisation Algorithm) circuits with an
ancilla-based post-selection probe onto qgate's :class:`ParityOutcome`
model, enabling the full trajectory filtering pipeline (scoring →
thresholding → conditioning) to work on combinatorial optimisation —
specifically the MaxCut problem on random graphs.

**Mapping to ParityOutcome:**
  - ``n_subsystems`` = number of graph nodes (qubits).
  - ``n_cycles``     = number of QAOA layers (p).
  - ``parity_matrix[cycle, sub]`` = 0 if qubit *sub* contributes to
    a satisfying cut at layer *cycle* (via cost-function probe),
    1 otherwise.  This lets qgate's score_fusion, thresholding, and
    hierarchical conditioning rules apply naturally.

The adapter supports two algorithm variants via ``algorithm_mode``:
  - ``"standard"`` — Canonical QAOA (cost + mixer layers).
  - ``"tsvf"``     — QAOA + chaotic entangling ansatz + weak-measurement
                     ancilla per layer (post-selection trajectory filter).

**MaxCut problem:**
  Given an undirected graph G = (V, E), find a partition of vertices
  into two sets that maximises the number of edges crossing the cut.
  The QAOA cost operator encodes ``C = Σ_{(i,j)∈E} ½(1 - Z_i·Z_j)``.

Patent pending (see LICENSE)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from qgate.adapters.base import BaseAdapter
from qgate.conditioning import ParityOutcome

# Guard Qiskit imports for load-time tolerance.
try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
    from qiskit.circuit.library import RYGate

    _HAS_QISKIT = True
except ImportError:  # pragma: no cover
    _HAS_QISKIT = False


# ═══════════════════════════════════════════════════════════════════════════
# Graph helpers
# ═══════════════════════════════════════════════════════════════════════════


def random_regular_graph(
    n_nodes: int,
    degree: int = 3,
    seed: int | None = None,
) -> list[tuple[int, int]]:
    """Generate a random regular-ish graph as an edge list.

    Falls back to an Erdős–Rényi-like model when exact regular graph
    construction isn't possible (e.g. odd degree × odd nodes).

    Returns:
        List of (i, j) edges with i < j.
    """
    rng = np.random.default_rng(seed)
    edges: set[tuple[int, int]] = set()

    # Build a random graph with target average degree
    target_edges = (n_nodes * degree) // 2
    attempts = 0
    while len(edges) < target_edges and attempts < target_edges * 50:
        i = int(rng.integers(0, n_nodes))
        j = int(rng.integers(0, n_nodes))
        if i != j:
            edge = (min(i, j), max(i, j))
            edges.add(edge)
        attempts += 1

    # Ensure graph is connected — add a spanning path if needed
    visited = {0}
    queue = [0]
    while queue:
        node = queue.pop(0)
        for a, b in edges:
            other = b if a == node else (a if b == node else None)
            if other is not None and other not in visited:
                visited.add(other)
                queue.append(other)
    for node in range(n_nodes):
        if node not in visited:
            prev = node - 1
            edge = (min(prev, node), max(prev, node))
            edges.add(edge)
            visited.add(node)

    return sorted(edges)


def maxcut_value(bitstring: str, edges: list[tuple[int, int]]) -> int:
    """Compute the MaxCut value for a bitstring partition."""
    cut = 0
    for i, j in edges:
        if i < len(bitstring) and j < len(bitstring) and bitstring[i] != bitstring[j]:
            cut += 1
    return cut


def best_maxcut(n_nodes: int, edges: list[tuple[int, int]]) -> tuple[str, int]:
    """Brute-force the best MaxCut solution (only for small graphs)."""
    best_bs = "0" * n_nodes
    best_val = 0
    for x in range(2**n_nodes):
        bs = format(x, f"0{n_nodes}b")
        val = maxcut_value(bs, edges)
        if val > best_val:
            best_val = val
            best_bs = bs
    return best_bs, best_val


# ═══════════════════════════════════════════════════════════════════════════
# Circuit primitives
# ═══════════════════════════════════════════════════════════════════════════


def _qaoa_cost_layer(
    qc: QuantumCircuit,
    qubits: list[int],
    edges: list[tuple[int, int]],
    gamma: float,
) -> None:
    """Apply the MaxCut cost unitary: exp(-iγC).

    For each edge (i, j): CNOT(i,j) → Rz(2γ) on j → CNOT(i,j).
    This implements exp(-iγ · ½(1 - Z_i·Z_j)) up to global phase.
    """
    for i, j in edges:
        qc.cx(qubits[i], qubits[j])
        qc.rz(2 * gamma, qubits[j])
        qc.cx(qubits[i], qubits[j])


def _qaoa_mixer_layer(
    qc: QuantumCircuit,
    qubits: list[int],
    beta: float,
) -> None:
    """Apply the transverse-field mixer: exp(-iβB) = ∏_k Rx(2β)."""
    for q in qubits:
        qc.rx(2 * beta, q)


def _chaotic_qaoa_ansatz(
    qc: QuantumCircuit,
    qubits: list[int],
    layer: int,
    rng: np.random.Generator,
) -> None:
    """Parameterised entangling ansatz replacing the mixer in TSVF mode.

    Two layers of random single-qubit rotations + all-to-all CNOTs,
    followed by a small random Ry perturbation per qubit.
    """
    n = len(qubits)
    for _sub_layer in range(2):
        for q in qubits:
            qc.rx(rng.uniform(0, 2 * math.pi), q)
            qc.ry(rng.uniform(0, 2 * math.pi), q)
            qc.rz(rng.uniform(0, 2 * math.pi), q)
        for i in range(n):
            for j in range(n):
                if i != j:
                    qc.cx(qubits[i], qubits[j])
        qc.barrier()
    # Small perturbation scaled by layer
    scale = 0.3 / (1 + 0.1 * layer)
    for q in qubits:
        qc.ry(scale * rng.uniform(-1, 1), q)


def _add_cost_probe_ancilla(
    qc: QuantumCircuit,
    qubits: list[int],
    edges: list[tuple[int, int]],
    ancilla_qubit: int,
    ancilla_cbit: Any,
    weak_angle: float = math.pi / 6,
) -> None:
    """Entangle ancilla conditioned on cut quality and measure it.

    Strategy: for each edge (i,j), if qubits i and j differ (good cut),
    we apply a small Ry rotation on the ancilla. The more edges that
    are cut, the larger the accumulated rotation, giving higher P(ancilla=1).

    Implementation: for each edge, use CNOT(i, aux) + CNOT(j, aux) to
    compute XOR(i,j) into a scratch approach, then CRY from that into
    the ancilla. Instead, we use a simpler but effective approach:
    apply a controlled-Ry from each qubit to the ancilla with opposite
    signs, so that when qubits differ the rotations add, and when they
    agree the rotations cancel.

    Simplified approach: Use a multi-controlled Ry conditioned on ALL
    qubits being in a "good" state isn't feasible generically. Instead,
    we use an additive weak-measurement: for each edge (i,j), apply a
    small CRY conditioned on qubit i and an anti-CRY on qubit j (or
    vice versa), so edges that are cut accumulate rotation on the ancilla.
    """
    per_edge_angle = weak_angle / max(len(edges), 1)

    for i, j in edges:
        # XOR-based: if qubits i and j differ, rotate ancilla
        # CNOT(i, anc) → CRY(angle, j, anc) → CNOT(i, anc)
        # This effectively applies RY when bit_i XOR bit_j = 1
        qc.cx(qubits[i], ancilla_qubit)
        # CRY: controlled-Ry from qubit j to ancilla
        cry = RYGate(per_edge_angle).control(1)
        qc.append(cry, [qubits[j], ancilla_qubit])
        qc.cx(qubits[i], ancilla_qubit)

        # Also the symmetric case
        qc.cx(qubits[j], ancilla_qubit)
        cry2 = RYGate(per_edge_angle).control(1)
        qc.append(cry2, [qubits[i], ancilla_qubit])
        qc.cx(qubits[j], ancilla_qubit)

    # Measure ancilla
    qc.measure(ancilla_qubit, ancilla_cbit)


# ═══════════════════════════════════════════════════════════════════════════
# Adapter
# ═══════════════════════════════════════════════════════════════════════════


class QAOATSVFAdapter(BaseAdapter):
    """Adapter for QAOA / TSVF-QAOA MaxCut experiments.

    This adapter builds QAOA circuits for the MaxCut problem, executes
    them on a Qiskit backend, and maps the raw results onto
    ``ParityOutcome`` objects that the rest of qgate can score and
    threshold.

    Args:
        backend:           A Qiskit backend (Aer or IBM Runtime).
        algorithm_mode:    ``"standard"`` or ``"tsvf"`` (default ``"tsvf"``).
        edges:             Edge list for the MaxCut graph.
        n_nodes:           Number of graph nodes (qubits). Required.
        gammas:            Cost layer angles (one per layer, or single float).
        betas:             Mixer layer angles (one per layer, or single float).
        seed:              RNG seed for chaotic ansatz and graph generation.
        weak_angle_base:   Base angle for the post-selection probe (radians).
        weak_angle_ramp:   Per-layer angle increase (radians).
        optimization_level: Transpilation optimisation level (0-3).
    """

    def __init__(
        self,
        backend: Any = None,
        *,
        algorithm_mode: str = "tsvf",
        edges: list[tuple[int, int]] | None = None,
        n_nodes: int = 4,
        gammas: list[float] | float | None = None,
        betas: list[float] | float | None = None,
        seed: int = 42,
        weak_angle_base: float = math.pi / 4,
        weak_angle_ramp: float = math.pi / 8,
        optimization_level: int = 1,
    ) -> None:
        if not _HAS_QISKIT:  # pragma: no cover
            raise ImportError(
                "QAOATSVFAdapter requires Qiskit. Install with: pip install qgate[qiskit]"
            )
        self.backend = backend
        self.algorithm_mode = algorithm_mode
        self.n_nodes = n_nodes
        self.seed = seed
        self.weak_angle_base = weak_angle_base
        self.weak_angle_ramp = weak_angle_ramp
        self.optimization_level = optimization_level

        # Graph edges
        if edges is not None:
            self.edges = edges
        else:
            self.edges = random_regular_graph(n_nodes, degree=3, seed=seed)

        # QAOA angles — default heuristic if not provided
        self._gammas_raw = gammas
        self._betas_raw = betas

    def _get_angles(self, n_layers: int) -> tuple[list[float], list[float]]:
        """Resolve gamma/beta arrays for n_layers."""
        if self._gammas_raw is None:
            # Heuristic: linearly spaced from π/8 to π/4
            gammas = [
                math.pi / 8 + (math.pi / 8) * idx / max(n_layers - 1, 1) for idx in range(n_layers)
            ]
        elif isinstance(self._gammas_raw, (int, float)):
            gammas = [float(self._gammas_raw)] * n_layers
        else:
            gammas = list(self._gammas_raw)
            if len(gammas) < n_layers:
                gammas = gammas + [gammas[-1]] * (n_layers - len(gammas))

        if self._betas_raw is None:
            # Heuristic: linearly spaced from π/4 to π/8
            betas = [
                math.pi / 4 - (math.pi / 8) * idx / max(n_layers - 1, 1) for idx in range(n_layers)
            ]
        elif isinstance(self._betas_raw, (int, float)):
            betas = [float(self._betas_raw)] * n_layers
        else:
            betas = list(self._betas_raw)
            if len(betas) < n_layers:
                betas = betas + [betas[-1]] * (n_layers - len(betas))

        return gammas[:n_layers], betas[:n_layers]

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def build_circuit(
        self,
        n_subsystems: int,
        n_cycles: int,
        **kwargs: Any,
    ) -> QuantumCircuit:
        """Build the QAOA circuit.

        ``n_subsystems`` = number of graph nodes (must match ``n_nodes``).
        ``n_cycles`` = number of QAOA layers (p).

        Returns a :class:`QuantumCircuit`.
        """
        if n_subsystems != self.n_nodes:
            raise ValueError(f"n_subsystems ({n_subsystems}) must match n_nodes ({self.n_nodes})")
        if self.algorithm_mode == "standard":
            return self._build_standard(n_subsystems, n_cycles)
        elif self.algorithm_mode == "tsvf":
            seed_offset = kwargs.get("seed_offset", 0)
            return self._build_tsvf(n_subsystems, n_cycles, seed_offset)
        else:
            raise ValueError(
                f"Unknown algorithm_mode: {self.algorithm_mode!r}. Use 'standard' or 'tsvf'."
            )

    def run(
        self,
        circuit: Any,
        shots: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the circuit and return a raw result dict.

        Tries SamplerV2 first, falls back to ``backend.run()``.
        """
        if self.backend is None:
            raise RuntimeError("No backend configured for QAOATSVFAdapter")

        try:
            from qiskit.transpiler.preset_passmanagers import (
                generate_preset_pass_manager,
            )
            from qiskit_ibm_runtime import SamplerV2 as Sampler

            pm = generate_preset_pass_manager(
                backend=self.backend,
                optimization_level=self.optimization_level,
            )
            isa = pm.run(circuit)
            job = Sampler(mode=self.backend).run([isa], shots=shots)
            result = job.result()
            pub = result[0]
            return {
                "pub_result": pub,
                "circuit": circuit,
                "shots": shots,
            }
        except (ImportError, Exception):
            pass

        transpiled = transpile(
            circuit,
            backend=self.backend,
            optimization_level=self.optimization_level,
        )
        job = self.backend.run(transpiled, shots=shots)
        result = job.result()
        return {
            "counts": result.get_counts(0),
            "circuit": circuit,
            "shots": shots,
        }

    def parse_results(
        self,
        raw_results: Any,
        n_subsystems: int,
        n_cycles: int,
    ) -> list[ParityOutcome]:
        """Parse raw Qiskit results into ParityOutcome objects.

        Each shot → one ParityOutcome.  The parity matrix records per-
        layer, per-qubit: 0 if the qubit contributes to a "good" cut
        partition, 1 otherwise.

        For the TSVF variant the ancilla measurement at each layer
        provides the "cut quality probe".  For the standard variant we
        evaluate from the final measurement against the best-known cut.
        """
        counts = self._extract_counts(raw_results)

        outcomes: list[ParityOutcome] = []
        for bitstring, count in counts.items():
            row = self._bitstring_to_parity_row(bitstring, n_subsystems, n_cycles)
            for _ in range(count):
                outcomes.append(
                    ParityOutcome(
                        n_subsystems=n_subsystems,
                        n_cycles=n_cycles,
                        parity_matrix=row.copy(),
                    )
                )
        return outcomes

    # ------------------------------------------------------------------
    # Public helpers (beyond BaseAdapter)
    # ------------------------------------------------------------------

    def get_transpiled_depth(self, circuit: QuantumCircuit) -> int:
        """Return the depth of the transpiled circuit."""
        transpiled = transpile(
            circuit,
            backend=self.backend,
            optimization_level=self.optimization_level,
        )
        return int(transpiled.depth())

    def extract_cut_quality(
        self,
        raw_results: dict[str, Any],
        postselect: bool = True,
    ) -> tuple[float, float, int]:
        """Extract the mean cut ratio and approximation ratio.

        Returns (mean_cut_ratio, approx_ratio, total_shots_used).

        ``mean_cut_ratio`` = mean(cut_value) / max_possible_edges.
        ``approx_ratio``   = mean(cut_value) / best_known_cut.
        """
        counts = self._extract_counts(raw_results)
        _, best_cut = best_maxcut(self.n_nodes, self.edges)
        max_edges = len(self.edges)

        if not postselect or self.algorithm_mode == "standard":
            total = sum(counts.values())
            if total == 0:
                return 0.0, 0.0, 0
            total_cut = 0.0
            for key, val in counts.items():
                search = self._extract_search_bits(str(key))
                cv = maxcut_value(search, self.edges)
                total_cut += cv * val
            mean_cut = total_cut / total
            cut_ratio = mean_cut / max_edges if max_edges > 0 else 0.0
            approx_ratio = mean_cut / best_cut if best_cut > 0 else 0.0
            return cut_ratio, approx_ratio, total

        # Post-select on ancilla = 1
        accepted_total = 0
        total_cut = 0.0
        for key, val in counts.items():
            key_str = str(key)
            anc_bit, search_bits = self._split_ancilla_search(key_str)
            if anc_bit == "1":
                accepted_total += val
                cv = maxcut_value(search_bits, self.edges)
                total_cut += cv * val
        if accepted_total == 0:
            return 0.0, 0.0, 0
        mean_cut = total_cut / accepted_total
        cut_ratio = mean_cut / max_edges if max_edges > 0 else 0.0
        approx_ratio = mean_cut / best_cut if best_cut > 0 else 0.0
        return cut_ratio, approx_ratio, accepted_total

    def extract_best_bitstring(
        self,
        raw_results: dict[str, Any],
        postselect: bool = True,
    ) -> tuple[str, int, int]:
        """Find the most-sampled bitstring and its cut value.

        Returns (bitstring, cut_value, count).
        """
        counts = self._extract_counts(raw_results)
        best_bs = ""
        best_count = 0
        best_cv = 0

        for key, val in counts.items():
            key_str = str(key)
            if self.algorithm_mode == "tsvf" and postselect:
                anc_bit, search_bits = self._split_ancilla_search(key_str)
                if anc_bit != "1":
                    continue
            else:
                search_bits = self._extract_search_bits(key_str)

            cv = maxcut_value(search_bits, self.edges)
            if val > best_count or (val == best_count and cv > best_cv):
                best_bs = search_bits
                best_count = val
                best_cv = cv

        return best_bs, best_cv, best_count

    # ------------------------------------------------------------------
    # Private circuit builders
    # ------------------------------------------------------------------

    def _build_standard(self, n_sub: int, n_layers: int) -> QuantumCircuit:
        """Standard QAOA: cost + mixer layers, no ancilla."""
        qr = QuantumRegister(n_sub, "q")
        cr = ClassicalRegister(n_sub, "c")
        qc = QuantumCircuit(qr, cr)
        qubits = list(range(n_sub))

        # Initial superposition
        for q in qubits:
            qc.h(q)

        gammas, betas = self._get_angles(n_layers)

        for layer in range(n_layers):
            _qaoa_cost_layer(qc, qubits, self.edges, gammas[layer])
            qc.barrier()
            _qaoa_mixer_layer(qc, qubits, betas[layer])
            qc.barrier()

        qc.measure(qubits, list(range(n_sub)))
        return qc

    def _build_tsvf(
        self,
        n_sub: int,
        n_layers: int,
        seed_offset: int = 0,
    ) -> QuantumCircuit:
        """TSVF QAOA: cost + chaotic ansatz + ancilla probe per layer."""
        qr = QuantumRegister(n_sub, "q")
        anc_r = QuantumRegister(1, "anc")
        cr = ClassicalRegister(n_sub, "c_search")
        cr_anc = ClassicalRegister(1, "c_anc")
        qc = QuantumCircuit(qr, anc_r, cr, cr_anc)

        qubits = list(range(n_sub))
        anc_qubit = n_sub
        rng = np.random.default_rng(self.seed + seed_offset)

        # Initial superposition
        for q in qubits:
            qc.h(q)

        gammas, _betas = self._get_angles(n_layers)

        for layer in range(n_layers):
            # Cost layer (same as standard — problem encoding)
            _qaoa_cost_layer(qc, qubits, self.edges, gammas[layer])
            qc.barrier()

            # Chaotic ansatz instead of mixer
            _chaotic_qaoa_ansatz(qc, qubits, layer=layer, rng=rng)
            qc.barrier()

            # Reset ancilla for reuse (except first layer)
            if layer > 0:
                qc.reset(anc_qubit)

            # Weak-measurement probe
            angle = self.weak_angle_base + self.weak_angle_ramp * min(layer, 4)
            _add_cost_probe_ancilla(
                qc,
                qubits,
                self.edges,
                anc_qubit,
                cr_anc[0],
                weak_angle=angle,
            )
            qc.barrier()

        qc.measure(qubits, list(range(n_sub)))
        return qc

    # ------------------------------------------------------------------
    # Private result parsing helpers
    # ------------------------------------------------------------------

    def _extract_counts(self, raw_results: Any) -> dict[str, int]:
        """Extract a counts dict from raw run() output."""
        if isinstance(raw_results, dict):
            if "counts" in raw_results:
                return self._normalise_counts(raw_results["counts"])
            if "pub_result" in raw_results:
                return self._counts_from_pub(
                    raw_results["pub_result"],
                    raw_results.get("circuit"),
                )
        return self._normalise_counts(raw_results)

    def _normalise_counts(self, counts: dict) -> dict[str, int]:
        """Ensure keys are bitstrings and values are ints."""
        out: dict[str, int] = {}
        for k, v in counts.items():
            out[str(k)] = int(v)
        return out

    def _counts_from_pub(self, pub: Any, circuit: Any) -> dict[str, int]:
        """Extract per-shot combined bitstrings from a SamplerV2 PubResult."""
        creg_names = [cr.name for cr in circuit.cregs] if circuit else []
        if len(creg_names) <= 1:
            name = creg_names[0] if creg_names else "c"
            try:
                return {str(k): int(v) for k, v in pub.data[name].get_counts().items()}
            except Exception:
                return {}

        # Multi-register: reconstruct combined bitstrings
        try:
            reg_bitstrings: dict[str, Any] = {}
            for name in creg_names:
                reg_bitstrings[name] = pub.data[name].get_bitstrings()
            num_shots = len(reg_bitstrings[creg_names[0]])
            combined: dict[str, int] = {}
            for i in range(num_shots):
                parts = []
                for name in reversed(creg_names):
                    parts.append(reg_bitstrings[name][i])
                full = " ".join(parts)
                combined[full] = combined.get(full, 0) + 1
            return combined
        except Exception:
            name = creg_names[0]
            try:
                return {str(k): int(v) for k, v in pub.data[name].get_counts().items()}
            except Exception:
                return {}

    def _extract_search_bits(self, bitstring: str) -> str:
        """Extract the search register bits from a bitstring."""
        key = bitstring.strip()
        if " " in key:
            return key.split()[-1]
        return key[-self.n_nodes :]

    def _split_ancilla_search(self, bitstring: str) -> tuple[str, str]:
        """Split bitstring into (ancilla_bit, search_bits)."""
        key = bitstring.strip()
        if " " in key:
            parts = key.split()
            return parts[0], parts[-1]
        return key[0], key[1:]

    def _bitstring_to_parity_row(
        self,
        bitstring: str,
        n_subsystems: int,
        n_cycles: int,
    ) -> np.ndarray:
        """Convert a measurement bitstring to a parity matrix.

        For the TSVF variant:
          - Ancilla=1 → evaluate per-qubit cut contribution:
            0 if qubit is on the "cut side" of at least one edge, 1 otherwise.
          - Ancilla=0 → row of 1s (all fail — no evidence of good cut).

        For the standard variant:
          - Evaluate per-qubit cut contribution from final measurement.
          - Replicate across all cycles.

        Shape: (n_cycles, n_subsystems).
        """
        if self.algorithm_mode == "tsvf":
            anc_bit, search_bits = self._split_ancilla_search(bitstring)

            if anc_bit == "1":
                qubit_quality = self._compute_qubit_cut_quality(
                    search_bits,
                    n_subsystems,
                )
                matrix = np.tile(qubit_quality, (n_cycles, 1))
            else:
                matrix = np.ones((n_cycles, n_subsystems), dtype=np.int8)
        else:
            search_bits = self._extract_search_bits(bitstring)
            qubit_quality = self._compute_qubit_cut_quality(
                search_bits,
                n_subsystems,
            )
            matrix = np.tile(qubit_quality, (n_cycles, 1))

        return matrix

    def _compute_qubit_cut_quality(
        self,
        search_bits: str,
        n_subsystems: int,
    ) -> np.ndarray:
        """Compute per-qubit cut quality: 0 = contributes to cut, 1 = doesn't.

        A qubit contributes to the cut if it is on the opposite side of
        at least one of its neighbour edges.
        """
        quality = np.ones(n_subsystems, dtype=np.int8)  # default: fail
        for i in range(min(len(search_bits), n_subsystems)):
            # Check if this qubit participates in any cut edge
            for a, b in self.edges:
                other = b if a == i else (a if b == i else None)
                if (
                    other is not None
                    and other < len(search_bits)
                    and search_bits[i] != search_bits[other]
                ):
                    quality[i] = 0  # contributes to a cut
                    break
        return quality
