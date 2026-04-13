"""
qpe_adapter.py — Adapter for QPE / TSVF-QPE experiments.

Maps Quantum Phase Estimation (QPE) circuits with an ancilla-based
post-selection probe onto qgate's :class:`ParityOutcome` model, enabling
the full trajectory filtering pipeline (scoring → thresholding →
conditioning) to work on eigenvalue estimation.

**Problem — Quantum Phase Estimation:**
  Given a unitary U and its eigenstate |ψ⟩ such that U|ψ⟩ = e^{2πiφ}|ψ⟩,
  QPE estimates the phase φ to t-bit binary precision using a register
  of t "precision" (counting) qubits.

  Target unitary: U = Rz(2πφ), with eigenstate |1⟩ (eigenvalue e^{-iπφ})
  or equivalently a diagonal unitary diag(1, e^{2πiφ}).

**Mapping to ParityOutcome:**
  - ``n_subsystems`` = number of precision qubits (t).
  - ``n_cycles``     = 1 (QPE is a single-shot algorithm per run).
  - ``parity_matrix[0, k]`` = 0 if precision qubit *k* matches the
    correct phase bit, 1 otherwise.  This lets qgate's score_fusion,
    thresholding, and hierarchical conditioning rules apply naturally.

The adapter supports two algorithm variants via ``algorithm_mode``:
  - ``"standard"`` — Canonical QPE (Hadamards + controlled-U^{2^k}
                     + inverse QFT).
  - ``"tsvf"``     — QPE + chaotic entangling ansatz + weak-measurement
                     ancilla (post-selection trajectory filter) that
                     rewards phase states close to the correct answer.

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
# Phase-estimation helpers
# ═══════════════════════════════════════════════════════════════════════════


def phase_to_binary_fraction(phi: float, n_bits: int) -> str:
    """Convert a phase φ ∈ [0, 1) to its best n-bit binary fraction string.

    The binary fraction 0.b₁b₂…bₜ represents φ ≈ Σ bₖ / 2ᵏ.
    We return the string "b₁b₂…bₜ".

    Example: φ = 0.375 with n_bits=3 → "011"  (0.011₂ = 3/8)
    """
    val = round(phi * (2**n_bits)) % (2**n_bits)
    return format(val, f"0{n_bits}b")


def binary_fraction_to_phase(bitstring: str) -> float:
    """Convert a binary fraction string to a phase value.

    "011" → 0.011₂ = 3/8 = 0.375
    """
    n = len(bitstring)
    val = int(bitstring, 2)
    return float(val / (2**n))


def phase_error(measured_phase: float, true_phase: float) -> float:
    """Circular phase error in [0, 0.5].

    Accounts for the wraparound: |0.9 − 0.1| should be 0.2 not 0.8.
    """
    diff = abs(measured_phase - true_phase)
    return min(diff, 1.0 - diff)


def histogram_entropy(counts: dict[str, int]) -> float:
    """Shannon entropy of a measurement histogram (in bits).

    Lower entropy → sharper distribution (more peaked).
    For a uniform distribution over 2^t outcomes, entropy = t bits.
    A perfect delta function has entropy = 0.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for cnt in counts.values():
        if cnt > 0:
            p = cnt / total
            entropy -= p * math.log2(p)
    return entropy


def phase_fidelity(
    counts: dict[str, int],
    correct_bitstring: str,
) -> float:
    """Fraction of shots that measured the correct phase bitstring."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    correct_count = 0
    for key, val in counts.items():
        bs = key.strip().replace(" ", "")
        # Handle multi-register keys — take the last n bits
        n = len(correct_bitstring)
        if len(bs) >= n and bs[-n:] == correct_bitstring:
            correct_count += val
    return correct_count / total


def mean_phase_error(
    counts: dict[str, int],
    true_phase: float,
    n_bits: int,
) -> float:
    """Weighted mean circular phase error over all measurement outcomes."""
    total = sum(counts.values())
    if total == 0:
        return 0.5
    total_err = 0.0
    for key, val in counts.items():
        bs = key.strip().replace(" ", "")
        measured = binary_fraction_to_phase(bs[-n_bits:])
        total_err += phase_error(measured, true_phase) * val
    return total_err / total


# ═══════════════════════════════════════════════════════════════════════════
# Circuit primitives
# ═══════════════════════════════════════════════════════════════════════════


def _controlled_phase_rotation(
    qc: QuantumCircuit,
    control_qubit: int,
    target_qubit: int,
    angle: float,
) -> None:
    """Apply controlled-U^(2^k) where U = diag(1, e^{i·angle}).

    This is a controlled phase gate: |1⟩⟨1| ⊗ P(angle).
    Implemented as: CP(angle) = controlled-P(angle).
    """
    qc.cp(angle, control_qubit, target_qubit)


def _inverse_qft(
    qc: QuantumCircuit,
    qubits: list[int],
) -> None:
    """Apply inverse QFT on the given qubits (MSB-first convention).

    The inverse QFT transforms the phase-encoded amplitudes into
    a computational-basis state representing the binary fraction.
    """
    n = len(qubits)
    for i in range(n // 2):
        qc.swap(qubits[i], qubits[n - 1 - i])
    for target in range(n):
        for ctrl in range(target):
            angle = -math.pi / (2 ** (target - ctrl))
            qc.cp(angle, qubits[ctrl], qubits[target])
        qc.h(qubits[target])


def _chaotic_qpe_ansatz(
    qc: QuantumCircuit,
    qubits: list[int],
    iteration: int,
    rng: np.random.Generator,
    n_precision: int = 0,
) -> None:
    """Mild perturbation ansatz for the TSVF QPE variant.

    QPE encodes phase information in the **relative phases** between
    precision qubits.  A full-strength chaotic ansatz (as used in
    Grover / QAOA) would destroy this delicate structure entirely.

    Instead we apply a **mild noise injection**:
      1. Small random Rz rotations per qubit — perturbs the relative
         phases without completely scrambling them.
      2. Nearest-neighbour CZ entangling layer — couples qubits weakly.
      3. A second round of even smaller Ry+Rz rotations.

    The perturbation strength scales as ``π / (4 · n_precision)`` so
    that larger registers receive proportionally smaller per-qubit
    kicks.  The TSVF post-selection probe then filters for trajectories
    where the phase survived the perturbation — the "anchoring" effect.
    """
    n = len(qubits)
    effective_n = max(n_precision, n)

    # Scale: perturbation gets weaker for more precision qubits
    base_scale = math.pi / (4.0 * max(effective_n, 1))

    # Layer 1: small Rz perturbations
    for q in qubits:
        qc.rz(base_scale * rng.uniform(-1, 1), q)

    # Nearest-neighbour CZ entangling
    for i in range(n - 1):
        qc.cz(qubits[i], qubits[i + 1])

    qc.barrier()

    # Layer 2: even smaller Ry + Rz perturbations
    tiny_scale = base_scale * 0.5
    for q in qubits:
        qc.ry(tiny_scale * rng.uniform(-1, 1), q)
        qc.rz(tiny_scale * rng.uniform(-1, 1), q)


def _add_phase_probe_ancilla(
    qc: QuantumCircuit,
    precision_qubits: list[int],
    ancilla_qubit: int,
    ancilla_cbit: Any,
    correct_phase_bits: str,
    weak_angle: float = math.pi / 4,
) -> None:
    """Entangle ancilla conditioned on proximity to correct phase and measure.

    Strategy: For each precision qubit k, apply a controlled-Ry
    rotation on the ancilla that rewards the qubit being in the
    correct state (matching the k-th bit of the true phase binary
    fraction).

    Implementation — per-qubit conditional reward:
      If correct_bit[k] == "0":
        X(q_k) → CRY(angle, q_k, anc) → X(q_k)
        Fires when q_k = |0⟩ (correct).
      If correct_bit[k] == "1":
        CRY(angle, q_k, anc)
        Fires when q_k = |1⟩ (correct).

    Net effect: ancilla accumulates rotation proportional to the
    number of precision qubits matching the correct phase bits.
    Higher match → higher P(anc=|1⟩) → post-selection retains
    trajectories closer to the true phase.
    """
    n = len(precision_qubits)
    per_bit_angle = weak_angle / max(n, 1)

    for k in range(n):
        q_k = precision_qubits[k]
        target_bit = correct_phase_bits[k] if k < len(correct_phase_bits) else "0"

        if target_bit == "0":
            # Reward |0⟩: flip → CRY → un-flip
            qc.x(q_k)
            cry = RYGate(per_bit_angle).control(1)
            qc.append(cry, [q_k, ancilla_qubit])
            qc.x(q_k)
        else:
            # Reward |1⟩: CRY directly
            cry = RYGate(per_bit_angle).control(1)
            qc.append(cry, [q_k, ancilla_qubit])

    # Measure ancilla
    qc.measure(ancilla_qubit, ancilla_cbit)


# ═══════════════════════════════════════════════════════════════════════════
# Adapter
# ═══════════════════════════════════════════════════════════════════════════


class QPETSVFAdapter(BaseAdapter):
    """Adapter for QPE / TSVF-QPE phase estimation experiments.

    This adapter builds QPE circuits for estimating the eigenphase of
    a unitary operator, executes them on a Qiskit backend, and maps
    the raw results onto ``ParityOutcome`` objects that the rest of
    qgate can score and threshold.

    **Target unitary:** ``U = diag(1, e^{2πiφ})``
    with eigenstate |1⟩ and eigenphase φ.

    Args:
        backend:           A Qiskit backend (Aer or IBM Runtime).
        algorithm_mode:    ``"standard"`` or ``"tsvf"`` (default ``"tsvf"``).
        eigenphase:        The true phase φ ∈ [0, 1) (default 1/3).
        seed:              RNG seed for chaotic ansatz.
        weak_angle_base:   Base angle for the phase probe (radians).
        weak_angle_ramp:   Per-precision-qubit angle increase (radians).
        optimization_level: Transpilation optimisation level (0-3).
    """

    def __init__(
        self,
        backend: Any = None,
        *,
        algorithm_mode: str = "tsvf",
        eigenphase: float = 1.0 / 3.0,
        seed: int = 42,
        weak_angle_base: float = math.pi / 4,
        weak_angle_ramp: float = math.pi / 8,
        optimization_level: int = 1,
    ) -> None:
        if not _HAS_QISKIT:  # pragma: no cover
            raise ImportError(
                "QPETSVFAdapter requires Qiskit. Install with: pip install qgate[qiskit]"
            )
        self.backend = backend
        self.algorithm_mode = algorithm_mode
        self.eigenphase = eigenphase
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
        """Build the QPE circuit.

        ``n_subsystems`` = number of precision qubits (t).
        ``n_cycles`` = 1 (QPE is a single-pass algorithm).  Accepted
        but ignored (always 1 effective cycle).

        Returns a :class:`QuantumCircuit`.
        """
        if self.algorithm_mode == "standard":
            return self._build_standard(n_subsystems)
        elif self.algorithm_mode == "tsvf":
            seed_offset = kwargs.get("seed_offset", 0)
            return self._build_tsvf(n_subsystems, seed_offset)
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
            raise RuntimeError("No backend configured for QPETSVFAdapter")

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
        precision-qubit: 0 if the qubit matches the correct phase bit,
        1 otherwise.

        For the TSVF variant the ancilla measurement provides the
        "phase quality probe".  For the standard variant we evaluate
        from the final measurement against the ideal phase bits.
        """
        counts = self._extract_counts(raw_results)

        # Determine the correct phase bitstring
        correct_bits = phase_to_binary_fraction(self.eigenphase, n_subsystems)

        outcomes: list[ParityOutcome] = []
        for bitstring, count in counts.items():
            row = self._bitstring_to_parity_row(
                bitstring,
                n_subsystems,
                n_cycles,
                correct_bits,
            )
            for _ in range(count):
                outcomes.append(
                    ParityOutcome(
                        n_subsystems=n_subsystems,
                        n_cycles=max(n_cycles, 1),
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

    def get_correct_phase_bits(self, n_precision: int) -> str:
        """Return the ideal binary-fraction bitstring for the eigenphase."""
        return phase_to_binary_fraction(self.eigenphase, n_precision)

    def extract_phase_metrics(
        self,
        raw_results: dict[str, Any],
        n_precision: int,
        postselect: bool = True,
    ) -> dict[str, float]:
        """Extract comprehensive phase estimation metrics.

        Returns a dict with:
          - ``fidelity``: P(correct phase bitstring)
          - ``mean_phase_error``: weighted-mean circular phase error
          - ``entropy``: Shannon entropy of the phase histogram (bits)
          - ``measured_phase``: most-probable measured phase
          - ``true_phase``: the target eigenphase
          - ``total_shots``: number of shots used (after post-selection)
          - ``acceptance_rate``: fraction of shots accepted (TSVF only)
        """
        counts = self._extract_counts(raw_results)
        correct_bits = phase_to_binary_fraction(self.eigenphase, n_precision)

        if postselect and self.algorithm_mode == "tsvf":
            phase_counts, total_original, accepted_total = self._postselect_phase_counts(
                counts, n_precision
            )
            acceptance_rate = accepted_total / total_original if total_original > 0 else 0.0
        else:
            phase_counts = self._extract_phase_counts(counts, n_precision)
            accepted_total = sum(phase_counts.values())
            acceptance_rate = 1.0

        fid = phase_fidelity(phase_counts, correct_bits)
        mean_err = mean_phase_error(
            phase_counts,
            self.eigenphase,
            n_precision,
        )
        ent = histogram_entropy(phase_counts)

        # Most-probable phase
        if phase_counts:
            best_bs = max(phase_counts, key=phase_counts.get)  # type: ignore[arg-type]
            measured = binary_fraction_to_phase(best_bs[-n_precision:])
        else:
            measured = 0.0

        return {
            "fidelity": fid,
            "mean_phase_error": mean_err,
            "entropy": ent,
            "measured_phase": measured,
            "true_phase": self.eigenphase,
            "total_shots": accepted_total,
            "acceptance_rate": acceptance_rate,
        }

    def extract_best_phase(
        self,
        raw_results: dict[str, Any],
        n_precision: int,
        postselect: bool = True,
    ) -> tuple[str, float, int]:
        """Find the most-sampled phase bitstring.

        Returns (bitstring, phase_value, count).
        """
        counts = self._extract_counts(raw_results)

        if postselect and self.algorithm_mode == "tsvf":
            phase_counts, _, _ = self._postselect_phase_counts(
                counts,
                n_precision,
            )
        else:
            phase_counts = self._extract_phase_counts(counts, n_precision)

        if not phase_counts:
            return "0" * n_precision, 0.0, 0

        best_bs = max(phase_counts, key=phase_counts.get)  # type: ignore[arg-type]
        best_count = phase_counts[best_bs]
        best_phase = binary_fraction_to_phase(best_bs[-n_precision:])
        return best_bs, best_phase, best_count

    # ------------------------------------------------------------------
    # Private circuit builders
    # ------------------------------------------------------------------

    def _build_standard(self, n_precision: int) -> QuantumCircuit:
        """Standard QPE: Hadamards + controlled-U^{2^k} + inverse QFT."""
        # Registers: t precision qubits + 1 eigenstate qubit
        prec_r = QuantumRegister(n_precision, "prec")
        eig_r = QuantumRegister(1, "eig")
        cr = ClassicalRegister(n_precision, "c_phase")
        qc = QuantumCircuit(prec_r, eig_r, cr)

        prec_qubits = list(range(n_precision))
        eig_qubit = n_precision

        # Prepare eigenstate |1⟩
        qc.x(eig_qubit)

        # Hadamard on all precision qubits
        for q in prec_qubits:
            qc.h(q)

        # Controlled-U^{2^k} gates
        # U = diag(1, e^{2πiφ}) → controlled-U^{2^k} is CP(2π·φ·2^k)
        for k in range(n_precision):
            angle = 2 * math.pi * self.eigenphase * (2**k)
            _controlled_phase_rotation(
                qc,
                prec_qubits[k],
                eig_qubit,
                angle,
            )

        # Inverse QFT on precision register
        _inverse_qft(qc, prec_qubits)

        # Measure precision register
        for k in range(n_precision):
            qc.measure(prec_qubits[k], cr[k])

        return qc

    def _build_tsvf(
        self,
        n_precision: int,
        seed_offset: int = 0,
    ) -> QuantumCircuit:
        """TSVF QPE: standard QPE + chaotic ansatz + phase probe ancilla.

        The chaotic ansatz is applied BEFORE the inverse QFT, perturbing
        the phase-encoded state.  The ancilla probe then post-selects
        for trajectories where the precision register still encodes a
        phase close to the correct answer — the TSVF "anchoring" effect.
        """
        prec_r = QuantumRegister(n_precision, "prec")
        eig_r = QuantumRegister(1, "eig")
        anc_r = QuantumRegister(1, "anc")
        cr = ClassicalRegister(n_precision, "c_phase")
        cr_anc = ClassicalRegister(1, "c_anc")
        qc = QuantumCircuit(prec_r, eig_r, anc_r, cr, cr_anc)

        prec_qubits = list(range(n_precision))
        eig_qubit = n_precision
        anc_qubit = n_precision + 1

        rng = np.random.default_rng(self.seed + seed_offset)

        # Prepare eigenstate |1⟩
        qc.x(eig_qubit)

        # Hadamard on all precision qubits
        for q in prec_qubits:
            qc.h(q)

        # Controlled-U^{2^k} gates (same as standard)
        for k in range(n_precision):
            angle = 2 * math.pi * self.eigenphase * (2**k)
            _controlled_phase_rotation(
                qc,
                prec_qubits[k],
                eig_qubit,
                angle,
            )

        qc.barrier()

        # ── TSVF mild perturbation on precision register ──
        _chaotic_qpe_ansatz(
            qc,
            prec_qubits,
            iteration=0,
            rng=rng,
            n_precision=n_precision,
        )
        qc.barrier()

        # ── Inverse QFT on precision register ──
        _inverse_qft(qc, prec_qubits)
        qc.barrier()

        # ── Phase probe ancilla ──
        correct_bits = phase_to_binary_fraction(self.eigenphase, n_precision)
        angle = self.weak_angle_base + self.weak_angle_ramp * min(n_precision, 6)
        _add_phase_probe_ancilla(
            qc,
            prec_qubits,
            anc_qubit,
            cr_anc[0],
            correct_phase_bits=correct_bits,
            weak_angle=angle,
        )
        qc.barrier()

        # Measure precision register
        for k in range(n_precision):
            qc.measure(prec_qubits[k], cr[k])

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
            name = creg_names[0] if creg_names else "c_phase"
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

    def _extract_phase_bits(self, bitstring: str, n_precision: int) -> str:
        """Extract the precision-register bits from a bitstring."""
        key = bitstring.strip()
        if " " in key:
            # Space-separated: last part is first register (c_phase)
            return key.split()[-1]
        return key[-n_precision:]

    def _split_ancilla_phase(
        self,
        bitstring: str,
        n_precision: int,
    ) -> tuple[str, str]:
        """Split bitstring into (ancilla_bit, phase_bits).

        For space-separated keys: "anc_bit phase_bits"
        For concatenated: first char is ancilla, rest is phase.
        """
        key = bitstring.strip()
        if " " in key:
            parts = key.split()
            # parts[0] is the LAST classical register written (c_anc)
            # parts[-1] is the FIRST classical register written (c_phase)
            return parts[0], parts[-1]
        # Concatenated: ancilla is MSB (leftmost)
        return key[0], key[1:]

    def _extract_phase_counts(
        self,
        counts: dict[str, int],
        n_precision: int,
    ) -> dict[str, int]:
        """Extract phase-register-only counts from full bitstrings."""
        phase_counts: dict[str, int] = {}
        for key, val in counts.items():
            if self.algorithm_mode == "tsvf":
                _, phase_bits = self._split_ancilla_phase(key, n_precision)
            else:
                phase_bits = self._extract_phase_bits(key, n_precision)
            phase_counts[phase_bits] = phase_counts.get(phase_bits, 0) + val
        return phase_counts

    def _postselect_phase_counts(
        self,
        counts: dict[str, int],
        n_precision: int,
    ) -> tuple[dict[str, int], int, int]:
        """Post-select on ancilla=1 and return phase counts.

        Returns (phase_counts, total_original, accepted_total).
        """
        total_original = sum(counts.values())
        phase_counts: dict[str, int] = {}
        accepted_total = 0

        for key, val in counts.items():
            anc_bit, phase_bits = self._split_ancilla_phase(key, n_precision)
            if anc_bit == "1":
                accepted_total += val
                phase_counts[phase_bits] = phase_counts.get(phase_bits, 0) + val

        return phase_counts, total_original, accepted_total

    def _bitstring_to_parity_row(
        self,
        bitstring: str,
        n_subsystems: int,
        n_cycles: int,
        correct_bits: str,
    ) -> np.ndarray:
        """Convert a measurement bitstring to a parity matrix.

        For the TSVF variant:
          - Ancilla=1 → compare each precision qubit to the correct
            phase bit: 0 if match, 1 if mismatch.
          - Ancilla=0 → row of 1s (all fail — no evidence of correct phase).

        For the standard variant:
          - Compare each precision qubit to the correct phase bit.

        Shape: (max(n_cycles, 1), n_subsystems).
        """
        effective_cycles = max(n_cycles, 1)

        if self.algorithm_mode == "tsvf":
            anc_bit, phase_bits = self._split_ancilla_phase(
                bitstring,
                n_subsystems,
            )
            if anc_bit == "1":
                qubit_match = self._compute_phase_match(
                    phase_bits,
                    n_subsystems,
                    correct_bits,
                )
                matrix = np.tile(qubit_match, (effective_cycles, 1))
            else:
                matrix = np.ones(
                    (effective_cycles, n_subsystems),
                    dtype=np.int8,
                )
        else:
            phase_bits = self._extract_phase_bits(bitstring, n_subsystems)
            qubit_match = self._compute_phase_match(
                phase_bits,
                n_subsystems,
                correct_bits,
            )
            matrix = np.tile(qubit_match, (effective_cycles, 1))

        return matrix

    def _compute_phase_match(
        self,
        phase_bits: str,
        n_subsystems: int,
        correct_bits: str,
    ) -> np.ndarray:
        """Compute per-qubit phase match: 0 = correct bit, 1 = wrong.

        Returns an array of shape (n_subsystems,).
        """
        match = np.ones(n_subsystems, dtype=np.int8)
        for k in range(min(len(phase_bits), n_subsystems, len(correct_bits))):
            if phase_bits[k] == correct_bits[k]:
                match[k] = 0
        return match
