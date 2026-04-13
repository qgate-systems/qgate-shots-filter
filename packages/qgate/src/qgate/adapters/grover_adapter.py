"""
grover_adapter.py — Adapter for Grover / TSVF-Chaotic Grover experiments.

Maps Grover search circuits with an ancilla-based post-selection probe
onto qgate's :class:`ParityOutcome` model, enabling the full trajectory
filtering pipeline (scoring → thresholding → conditioning) to work on
search algorithms — not only Bell-pair parity monitoring.

**Mapping to ParityOutcome:**
  - ``n_subsystems`` = number of search qubits (e.g. 3 for |101⟩).
  - ``n_cycles``     = number of Grover iterations.
  - ``parity_matrix[cycle, sub]`` = 0 if qubit *sub* was in the
    correct target state at iteration *cycle* (via ancilla probe),
    1 otherwise.  This lets qgate's score_fusion, thresholding, and
    hierarchical conditioning rules apply naturally.

The adapter supports two algorithm variants via ``algorithm_mode``:
  - ``"standard"`` — Oracle + diffusion per iteration.
  - ``"tsvf"``     — Oracle + chaotic ansatz + weak-measurement ancilla
                     per iteration (post-selection trajectory filter).

Patent pending (see LICENSE)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from qgate.adapters.base import BaseAdapter
from qgate.conditioning import ParityOutcome

# We guard Qiskit imports so the module can be imported without qiskit
# installed (load-time tolerance, same as QiskitAdapter).
try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
    from qiskit.circuit.library import RXGate, RYGate, RZGate  # noqa: F401

    _HAS_QISKIT = True
except ImportError:  # pragma: no cover
    _HAS_QISKIT = False


# ═══════════════════════════════════════════════════════════════════════════
# Circuit primitives
# ═══════════════════════════════════════════════════════════════════════════


def _oracle_101(qc: QuantumCircuit, qubits: list[int]) -> None:
    """Mark |101⟩ by flipping its phase via X-CCZ-X."""
    q0, q1, q2 = qubits
    qc.x(q1)
    qc.h(q2)
    qc.ccx(q0, q1, q2)
    qc.h(q2)
    qc.x(q1)


def _grover_diffusion(qc: QuantumCircuit, qubits: list[int]) -> None:
    """Standard 2|s⟩⟨s| − I inversion about mean."""
    for q in qubits:
        qc.h(q)
        qc.x(q)
    q0, q1, q2 = qubits
    qc.h(q2)
    qc.ccx(q0, q1, q2)
    qc.h(q2)
    for q in qubits:
        qc.x(q)
        qc.h(q)


def _chaotic_ansatz(
    qc: QuantumCircuit,
    qubits: list[int],
    iteration: int,
    rng: np.random.Generator,
) -> None:
    """Parameterised entangling ansatz replacing diffusion in TSVF mode."""
    n = len(qubits)
    for _layer in range(2):
        for q in qubits:
            qc.rx(rng.uniform(0, 2 * math.pi), q)
            qc.ry(rng.uniform(0, 2 * math.pi), q)
            qc.rz(rng.uniform(0, 2 * math.pi), q)
        for i in range(n):
            for j in range(n):
                if i != j:
                    qc.cx(qubits[i], qubits[j])
        qc.barrier()
    scale = 0.3 / (1 + 0.1 * iteration)
    for q in qubits:
        qc.ry(scale * rng.uniform(-1, 1), q)


def _add_postselection_ancilla(
    qc: QuantumCircuit,
    search_qubits: list[int],
    ancilla_qubit: int,
    ancilla_cbit: Any,
    weak_angle: float = math.pi / 6,
) -> None:
    """Entangle ancilla conditioned on |101⟩ and measure it.

    Uses Qiskit's multi-controlled RY gate to rotate the ancilla qubit
    by ``weak_angle`` only when the search register is in |101⟩.

    Steps:
      1. Flip q1 so |101⟩ → |111⟩ (all controls active).
      2. Apply MCRy(weak_angle) controlled on q0, q1, q2 → ancilla.
      3. Unflip q1.
      4. Measure ancilla.

    When the search register is in |101⟩ the ancilla picks up a
    rotation of ``weak_angle``, giving P(ancilla=1) = sin²(angle/2).
    Post-selecting on ancilla=1 filters for target-state trajectories.
    """
    from qiskit.circuit.library import RYGate

    _q0, q1, _q2 = search_qubits
    anc = ancilla_qubit

    # Flip q1: |101⟩ → |111⟩
    qc.x(q1)

    # Multi-controlled Ry(weak_angle) on ancilla, controlled by q0, q1, q2
    mcry = RYGate(weak_angle).control(len(search_qubits))
    qc.append(mcry, [*search_qubits, anc])

    # Unflip q1
    qc.x(q1)

    # Measure ancilla
    qc.measure(anc, ancilla_cbit)


# ═══════════════════════════════════════════════════════════════════════════
# Adapter
# ═══════════════════════════════════════════════════════════════════════════


class GroverTSVFAdapter(BaseAdapter):
    """Adapter for Grover / TSVF-Chaotic Grover experiments.

    This adapter builds Grover-search circuits, executes them on a Qiskit
    backend, and maps the raw results onto ``ParityOutcome`` objects that
    the rest of qgate can score and threshold.

    Args:
        backend:         A Qiskit backend (Aer or IBM Runtime).
        algorithm_mode:  ``"standard"`` or ``"tsvf"`` (default ``"tsvf"``).
        target_state:    Target bitstring (default ``"101"``).
        seed:            RNG seed for the chaotic ansatz.
        weak_angle_base: Base angle for the post-selection probe (radians).
        weak_angle_ramp: Per-iteration angle increase (radians).
        optimization_level: Transpilation optimisation level (0-3).
    """

    def __init__(
        self,
        backend: Any = None,
        *,
        algorithm_mode: str = "tsvf",
        target_state: str = "101",
        seed: int = 42,
        weak_angle_base: float = math.pi / 6,
        weak_angle_ramp: float = math.pi / 12,
        optimization_level: int = 1,
    ) -> None:
        if not _HAS_QISKIT:  # pragma: no cover
            raise ImportError(
                "GroverTSVFAdapter requires Qiskit. Install with: pip install qgate[qiskit]"
            )
        self.backend = backend
        self.algorithm_mode = algorithm_mode
        self.target_state = target_state
        self.n_search_qubits = len(target_state)
        self.seed = seed
        self.weak_angle_base = weak_angle_base
        self.weak_angle_ramp = weak_angle_ramp
        self.optimization_level = optimization_level

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def build_circuit(
        self,
        n_subsystems: int,
        n_cycles: int,
        **kwargs: Any,
    ) -> QuantumCircuit:
        """Build the Grover circuit.

        ``n_subsystems`` = number of search qubits (must match
        ``len(target_state)``).
        ``n_cycles`` = number of Grover iterations.

        Returns a :class:`QuantumCircuit`.
        """
        if n_subsystems != self.n_search_qubits:
            raise ValueError(
                f"n_subsystems ({n_subsystems}) must match target_state "
                f"length ({self.n_search_qubits})"
            )
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
            raise RuntimeError("No backend configured for GroverTSVFAdapter")

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
        iteration, per-qubit: 0 = qubit in target state, 1 = not.

        For the TSVF variant the ancilla measurement at each iteration
        provides the "parity probe".  For the standard variant we infer
        from the final measurement only (all cycles share the same row).
        """
        # Extract per-shot bitstrings
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

    def extract_target_probability(
        self,
        raw_results: dict[str, Any],
        postselect: bool = True,
    ) -> tuple[float, int]:
        """Extract P(target) from raw results, optionally post-selecting.

        Returns (probability, total_shots_used).
        """
        counts = self._extract_counts(raw_results)
        if not postselect or self.algorithm_mode == "standard":
            total = sum(counts.values())
            if total == 0:
                return 0.0, 0
            target_count = 0
            for key, val in counts.items():
                search = self._extract_search_bits(str(key))
                if search == self.target_state:
                    target_count += val
            return target_count / total, total

        # Post-select on ancilla = 1
        accepted_total = 0
        target_count = 0
        for key, val in counts.items():
            key_str = str(key)
            anc_bit, search_bits = self._split_ancilla_search(key_str)
            if anc_bit == "1":
                accepted_total += val
                if search_bits == self.target_state:
                    target_count += val
        if accepted_total == 0:
            return 0.0, 0
        return target_count / accepted_total, accepted_total

    # ------------------------------------------------------------------
    # Private circuit builders
    # ------------------------------------------------------------------

    def _build_standard(self, n_sub: int, n_iter: int) -> QuantumCircuit:
        """Standard Grover: oracle + diffusion, no ancilla."""
        qr = QuantumRegister(n_sub, "q")
        cr = ClassicalRegister(n_sub, "c")
        qc = QuantumCircuit(qr, cr)
        search_qubits = list(range(n_sub))
        for q in search_qubits:
            qc.h(q)
        for _ in range(n_iter):
            _oracle_101(qc, search_qubits)
            _grover_diffusion(qc, search_qubits)
        qc.measure(search_qubits, list(range(n_sub)))
        return qc

    def _build_tsvf(
        self,
        n_sub: int,
        n_iter: int,
        seed_offset: int = 0,
    ) -> QuantumCircuit:
        """TSVF chaotic Grover: oracle + chaotic ansatz + ancilla probe."""
        qr = QuantumRegister(n_sub, "q")
        anc_r = QuantumRegister(1, "anc")
        cr = ClassicalRegister(n_sub, "c_search")
        cr_anc = ClassicalRegister(1, "c_anc")
        qc = QuantumCircuit(qr, anc_r, cr, cr_anc)

        search_qubits = list(range(n_sub))
        anc_qubit = n_sub
        rng = np.random.default_rng(self.seed + seed_offset)

        for q in search_qubits:
            qc.h(q)

        for it in range(n_iter):
            _oracle_101(qc, search_qubits)
            qc.barrier()
            _chaotic_ansatz(qc, search_qubits, iteration=it, rng=rng)
            qc.barrier()
            if it > 0:
                qc.reset(anc_qubit)
            angle = self.weak_angle_base + self.weak_angle_ramp * min(it, 4)
            _add_postselection_ancilla(
                qc,
                search_qubits,
                anc_qubit,
                cr_anc[0],
                weak_angle=angle,
            )
            qc.barrier()

        qc.measure(search_qubits, list(range(n_sub)))
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
        # Already a counts dict
        return self._normalise_counts(raw_results)

    def _normalise_counts(self, counts: dict) -> dict[str, int]:
        """Ensure keys are bitstrings and values are ints."""
        out: dict[str, int] = {}
        for k, v in counts.items():
            out[str(k)] = int(v)
        return out

    def _counts_from_pub(self, pub, circuit: Any) -> dict[str, int]:
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
            reg_bitstrings = {}
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
            # Fallback: first register
            name = creg_names[0]
            try:
                return {str(k): int(v) for k, v in pub.data[name].get_counts().items()}
            except Exception:
                return {}

    def _extract_search_bits(self, bitstring: str) -> str:
        """Extract the search register bits from a bitstring."""
        key = bitstring.strip()
        if " " in key:
            # Space-separated: last part is first register (search)
            return key.split()[-1]
        # Concatenated: last n_search_qubits chars
        return key[-self.n_search_qubits :]

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
          - Ancilla=1 → row of 0s (all pass — the probe confirmed target).
          - Ancilla=0 → row of 1s (all fail — no evidence of target).

        For the standard variant:
          - Compare each qubit to target — 0 if match, 1 if mismatch.
          - Replicate the same row across all cycles (no mid-circuit info).

        Shape: (n_cycles, n_subsystems).
        """
        if self.algorithm_mode == "tsvf":
            anc_bit, search_bits = self._split_ancilla_search(bitstring)
            # Per-qubit match check
            qubit_match = np.array(
                [
                    0 if (i < len(search_bits) and search_bits[i] == self.target_state[i]) else 1
                    for i in range(n_subsystems)
                ],
                dtype=np.int8,
            )
            # Ancilla probed once at the end; replicate across cycles
            if anc_bit == "1":
                # Post-selection probe fired → use qubit-level match
                matrix = np.tile(qubit_match, (n_cycles, 1))
            else:
                # Probe didn't fire → mark all as fail
                matrix = np.ones((n_cycles, n_subsystems), dtype=np.int8)
        else:
            # Standard Grover — compare final measurement to target
            search_bits = self._extract_search_bits(bitstring)
            qubit_match = np.array(
                [
                    0 if (i < len(search_bits) and search_bits[i] == self.target_state[i]) else 1
                    for i in range(n_subsystems)
                ],
                dtype=np.int8,
            )
            matrix = np.tile(qubit_match, (n_cycles, 1))

        return matrix
