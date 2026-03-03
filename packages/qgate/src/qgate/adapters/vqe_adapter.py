"""
vqe_adapter.py — Adapter for VQE / TSVF-VQE experiments.

Maps Variational Quantum Eigensolver (VQE) circuits with an ancilla-based
post-selection probe onto qgate's :class:`ParityOutcome` model, enabling
the full trajectory filtering pipeline (scoring → thresholding →
conditioning) to work on ground-state-energy estimation — specifically
the Transverse-Field Ising Model (TFIM).

**Problem — Transverse-Field Ising Model (TFIM):**
  H = −J Σ_{<i,j>} Z_i Z_j  −  h Σ_i X_i

  where J is the coupling strength and h is the transverse field.
  For a 1D chain of n qubits with open boundary conditions:
    H = −J Σ_{i=0}^{n-2} Z_i Z_{i+1}  −  h Σ_{i=0}^{n-1} X_i

  The ground-state energy can be computed classically for benchmarking.

**Mapping to ParityOutcome:**
  - ``n_subsystems`` = number of qubits in the system.
  - ``n_cycles``     = number of ansatz layers (depth).
  - ``parity_matrix[cycle, sub]`` = 0 if qubit *sub* contributes to
    a low-energy configuration at layer *cycle* (via energy probe),
    1 otherwise.  This lets qgate's score_fusion, thresholding, and
    hierarchical conditioning rules apply naturally.

The adapter supports two algorithm variants via ``algorithm_mode``:
  - ``"standard"`` — Hardware-efficient ansatz (Ry+Rz + CNOT entangling).
  - ``"tsvf"``     — Hardware-efficient ansatz + chaotic entangling layers
                     + weak-measurement ancilla per layer (post-selection
                     trajectory filter).

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
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
# Hamiltonian helpers
# ═══════════════════════════════════════════════════════════════════════════


def tfim_exact_ground_energy(
    n_qubits: int,
    j_coupling: float = 1.0,
    h_field: float = 1.0,
) -> float:
    """Compute exact ground-state energy of the 1D TFIM by diagonalisation.

    H = −J Σ_{i} Z_i Z_{i+1}  −  h Σ_{i} X_i

    Only feasible for small n_qubits (≤ ~16).

    Returns:
        The minimum eigenvalue of H.
    """
    dim = 2**n_qubits

    # Build Pauli matrices
    eye2 = np.eye(2)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)

    def _kron_chain(ops: list[np.ndarray]) -> np.ndarray:
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    ham = np.zeros((dim, dim), dtype=complex)

    # ZZ coupling: -J Sum Z_i Z_{i+1}
    for i in range(n_qubits - 1):
        ops = [eye2] * n_qubits
        ops[i] = pauli_z
        ops[i + 1] = pauli_z
        ham -= j_coupling * _kron_chain(ops)

    # Transverse field: -h Sum X_i
    for i in range(n_qubits):
        ops = [eye2] * n_qubits
        ops[i] = pauli_x
        ham -= h_field * _kron_chain(ops)

    eigenvalues = np.linalg.eigvalsh(ham.real)
    return float(eigenvalues[0])


def compute_energy_from_bitstring(
    bitstring: str,
    n_qubits: int,
    j_coupling: float = 1.0,
    h_field: float = 0.0,
) -> float:
    """Compute the diagonal (ZZ) energy of a computational-basis state.

    Since X_i terms are off-diagonal, they don't contribute to
    individual computational-basis expectations.  The ZZ part gives:
      E_ZZ = −J Σ_{i} s_i · s_{i+1}
    where s_i = +1 if bit=0, −1 if bit=1.

    This is the energy that can be estimated from measurement counts.

    Returns:
        The ZZ contribution to the energy.
    """
    bits = [int(b) for b in bitstring[-n_qubits:]]
    spins = [1 - 2 * b for b in bits]  # 0->+1, 1->-1

    energy = 0.0
    for i in range(len(spins) - 1):
        energy -= j_coupling * spins[i] * spins[i + 1]
    return energy


def estimate_energy_from_counts(
    counts: dict[str, int],
    n_qubits: int,
    j_coupling: float = 1.0,
    h_field: float = 0.0,
) -> float:
    """Estimate the ZZ energy from measurement counts.

    Returns the weighted-average diagonal energy.
    """
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0
    total_energy = 0.0
    for bs, cnt in counts.items():
        e = compute_energy_from_bitstring(bs, n_qubits, j_coupling, h_field)
        total_energy += e * cnt
    return total_energy / total_shots


def energy_error(
    estimated: float,
    exact: float,
) -> float:
    """Absolute energy error: |estimated − exact|."""
    return abs(estimated - exact)


def energy_ratio(
    estimated: float,
    exact: float,
) -> float:
    """Energy ratio: estimated / exact.

    For ground-state search, a ratio closer to 1.0 is better
    (the estimated energy approaches the true ground-state energy).
    The exact ground-state energy is negative for TFIM, so
    ratio > 1 means we overestimate (too negative = too good),
    ratio < 1 means we underestimate (not negative enough).
    """
    if abs(exact) < 1e-12:
        return 0.0
    return estimated / exact


# ═══════════════════════════════════════════════════════════════════════════
# Circuit primitives
# ═══════════════════════════════════════════════════════════════════════════


def _hardware_efficient_layer(
    qc: QuantumCircuit,
    qubits: list[int],
    params: np.ndarray,
) -> None:
    """One layer of hardware-efficient ansatz: Ry + Rz per qubit + CNOT ladder.

    ``params`` shape: (n_qubits, 2) — [θ_ry, θ_rz] per qubit.
    """
    n = len(qubits)
    for i, q in enumerate(qubits):
        qc.ry(float(params[i, 0]), q)
        qc.rz(float(params[i, 1]), q)

    # CNOT entangling ladder
    for i in range(n - 1):
        qc.cx(qubits[i], qubits[i + 1])


def _chaotic_vqe_ansatz(
    qc: QuantumCircuit,
    qubits: list[int],
    layer: int,
    rng: np.random.Generator,
) -> None:
    """Parameterised entangling ansatz for the TSVF variant.

    Two sub-layers of random single-qubit rotations + all-to-all CNOTs,
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
    # Small perturbation scaled by layer depth
    scale = 0.3 / (1 + 0.1 * layer)
    for q in qubits:
        qc.ry(scale * rng.uniform(-1, 1), q)


def _add_energy_probe_ancilla(
    qc: QuantumCircuit,
    qubits: list[int],
    ancilla_qubit: int,
    ancilla_cbit: Any,
    n_qubits: int,
    weak_angle: float = math.pi / 6,
) -> None:
    """Entangle ancilla conditioned on low-energy configurations and measure.

    Strategy: For each nearest-neighbour pair (i, i+1), reward spin
    **alignment** (both |00⟩ or both |11⟩ → low ZZ energy) by
    rotating the ancilla toward |1⟩.

    Implementation — two complementary controlled-Ry paths per edge:

      Path A — both qubits |0⟩  (aligned, Z_i=+1, Z_j=+1):
        X(qi) → X(qj) → CRY(angle, [qi,qj] → anc) → X(qj) → X(qi)
        The 2-controlled Ry fires only when both are originally |0⟩.

      Path B — both qubits |1⟩  (aligned, Z_i=−1, Z_j=−1):
        CRY(angle, [qi,qj] → anc)
        Fires only when both are |1⟩.

    Net effect: ancilla accumulates rotation proportional to the
    number of aligned nearest-neighbour pairs, giving higher P(|1⟩)
    for lower-energy states.  The approach is fully unitary (no
    mid-circuit measurements or resets inside the probe) and adds
    only 2-controlled Ry gates, which Qiskit decomposes natively.
    """
    n_pairs = max(n_qubits - 1, 1)
    per_pair_angle = weak_angle / n_pairs

    for i in range(n_qubits - 1):
        qi = qubits[i]
        qj = qubits[i + 1]

        # Path A: reward |00⟩ alignment (flip both, 2-CRY, un-flip)
        qc.x(qi)
        qc.x(qj)
        cry_00 = RYGate(per_pair_angle).control(2)
        qc.append(cry_00, [qi, qj, ancilla_qubit])
        qc.x(qj)
        qc.x(qi)

        # Path B: reward |11⟩ alignment (2-CRY directly)
        cry_11 = RYGate(per_pair_angle).control(2)
        qc.append(cry_11, [qi, qj, ancilla_qubit])

    # Measure ancilla
    qc.measure(ancilla_qubit, ancilla_cbit)


# ═══════════════════════════════════════════════════════════════════════════
# Adapter
# ═══════════════════════════════════════════════════════════════════════════


class VQETSVFAdapter(BaseAdapter):
    """Adapter for VQE / TSVF-VQE ground-state energy experiments.

    This adapter builds VQE circuits for the Transverse-Field Ising
    Model (TFIM), executes them on a Qiskit backend, and maps the raw
    results onto ``ParityOutcome`` objects that the rest of qgate can
    score and threshold.

    The TFIM Hamiltonian:
      H = −J Σ Z_i Z_{i+1}  −  h Σ X_i

    Args:
        backend:           A Qiskit backend (Aer or IBM Runtime).
        algorithm_mode:    ``"standard"`` or ``"tsvf"`` (default ``"tsvf"``).
        n_qubits:          Number of system qubits.
        j_coupling:        ZZ coupling strength J (default 1.0).
        h_field:           Transverse field strength h (default 1.0).
        params:            Variational parameters. If None, random init.
        seed:              RNG seed for parameter init and chaotic ansatz.
        weak_angle_base:   Base angle for the energy probe (radians).
        weak_angle_ramp:   Per-layer angle increase (radians).
        optimization_level: Transpilation optimisation level (0-3).
    """

    def __init__(
        self,
        backend: Any = None,
        *,
        algorithm_mode: str = "tsvf",
        n_qubits: int = 4,
        j_coupling: float = 1.0,
        h_field: float = 1.0,
        params: np.ndarray | None = None,
        seed: int = 42,
        weak_angle_base: float = math.pi / 4,
        weak_angle_ramp: float = math.pi / 8,
        optimization_level: int = 1,
    ) -> None:
        if not _HAS_QISKIT:  # pragma: no cover
            raise ImportError(
                "VQETSVFAdapter requires Qiskit. Install with: pip install qgate[qiskit]"
            )
        self.backend = backend
        self.algorithm_mode = algorithm_mode
        self.n_qubits = n_qubits
        self.j_coupling = j_coupling
        self.h_field = h_field
        self.seed = seed
        self.weak_angle_base = weak_angle_base
        self.weak_angle_ramp = weak_angle_ramp
        self.optimization_level = optimization_level

        # Variational parameters: shape (n_layers, n_qubits, 2)
        # Will be set per build_circuit call if not pre-specified.
        self._params = params

    def _get_params(
        self,
        n_layers: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Resolve variational parameters for n_layers.

        Shape: (n_layers, n_qubits, 2) — [θ_ry, θ_rz] per qubit per layer.
        """
        if self._params is not None:
            p = np.array(self._params)
            if p.ndim == 2:
                # (n_qubits, 2) → replicate for all layers
                return np.tile(p, (n_layers, 1, 1))
            if p.ndim == 3 and p.shape[0] >= n_layers:
                return p[:n_layers]
            # Pad with random if not enough layers
            if p.ndim == 3:
                extra = rng.uniform(
                    -math.pi,
                    math.pi,
                    size=(n_layers - p.shape[0], self.n_qubits, 2),
                )
                return np.concatenate([p, extra], axis=0)

        # Random initialisation — identity-biased with layer-scaled
        # perturbation to avoid barren plateaus.  Early layers get larger
        # rotations (π/4 scale) to break symmetry, later layers get smaller
        # rotations (decay ∝ 1/√L) to preserve learned structure.
        params = np.zeros((n_layers, self.n_qubits, 2))
        for layer in range(n_layers):
            scale = (math.pi / 4) / math.sqrt(1 + layer)
            params[layer] = rng.uniform(-scale, scale, size=(self.n_qubits, 2))
        return params

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def build_circuit(
        self,
        n_subsystems: int,
        n_cycles: int,
        **kwargs: Any,
    ) -> QuantumCircuit:
        """Build the VQE circuit.

        ``n_subsystems`` = number of qubits (must match ``n_qubits``).
        ``n_cycles`` = number of ansatz layers (depth).

        Returns a :class:`QuantumCircuit`.
        """
        if n_subsystems != self.n_qubits:
            raise ValueError(
                f"n_subsystems ({n_subsystems}) must match n_qubits ({self.n_qubits})"
            )
        if self.algorithm_mode == "standard":
            return self._build_standard(n_subsystems, n_cycles, **kwargs)
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
            raise RuntimeError("No backend configured for VQETSVFAdapter")

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
        layer, per-qubit: 0 if the qubit contributes to a low-energy
        configuration, 1 otherwise.

        For the TSVF variant the ancilla measurement at each layer
        provides the "energy quality probe".  For the standard variant
        we evaluate from the final measurement against the ground state.
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

    def extract_energy(
        self,
        raw_results: dict[str, Any],
        postselect: bool = True,
    ) -> tuple[float, int]:
        """Extract the estimated ZZ energy from measurement results.

        Returns (estimated_energy, total_shots_used).

        For TSVF mode with postselect=True, only ancilla=1 shots are used.
        """
        counts = self._extract_counts(raw_results)

        if not postselect or self.algorithm_mode == "standard":
            search_counts = self._to_search_counts(counts, postselect=False)
            total = sum(search_counts.values())
            if total == 0:
                return 0.0, 0
            e = estimate_energy_from_counts(
                search_counts,
                self.n_qubits,
                self.j_coupling,
                self.h_field,
            )
            return e, total

        # Post-select on ancilla = 1
        search_counts = self._to_search_counts(counts, postselect=True)
        total = sum(search_counts.values())
        if total == 0:
            return 0.0, 0
        e = estimate_energy_from_counts(
            search_counts,
            self.n_qubits,
            self.j_coupling,
            self.h_field,
        )
        return e, total

    def extract_energy_ratio(
        self,
        raw_results: dict[str, Any],
        postselect: bool = True,
    ) -> tuple[float, float, int]:
        """Extract energy ratio relative to exact ground state.

        Returns (energy_ratio, energy_error, n_shots_used).
        energy_ratio = estimated / exact (closer to 1.0 is better).
        """
        est_energy, n_used = self.extract_energy(raw_results, postselect)
        exact = tfim_exact_ground_energy(
            self.n_qubits,
            self.j_coupling,
            self.h_field,
        )
        ratio = energy_ratio(est_energy, exact)
        err = energy_error(est_energy, exact)
        return ratio, err, n_used

    def extract_best_bitstring(
        self,
        raw_results: dict[str, Any],
        postselect: bool = True,
    ) -> tuple[str, float, int]:
        """Find the most-sampled bitstring and its energy.

        Returns (bitstring, energy, count).
        """
        counts = self._extract_counts(raw_results)
        best_bs = ""
        best_count = 0
        best_energy = 0.0

        for key, val in counts.items():
            key_str = str(key)
            if self.algorithm_mode == "tsvf" and postselect:
                anc_bit, search_bits = self._split_ancilla_search(key_str)
                if anc_bit != "1":
                    continue
            else:
                search_bits = self._extract_search_bits(key_str)

            e = compute_energy_from_bitstring(
                search_bits,
                self.n_qubits,
                self.j_coupling,
                self.h_field,
            )
            if val > best_count or (val == best_count and e < best_energy):
                best_bs = search_bits
                best_count = val
                best_energy = e

        return best_bs, best_energy, best_count

    def get_exact_ground_energy(self) -> float:
        """Return the exact ground-state energy for this TFIM instance."""
        return tfim_exact_ground_energy(
            self.n_qubits,
            self.j_coupling,
            self.h_field,
        )

    # ------------------------------------------------------------------
    # Private circuit builders
    # ------------------------------------------------------------------

    def _build_standard(
        self,
        n_sub: int,
        n_layers: int,
        **kwargs: Any,
    ) -> QuantumCircuit:
        """Standard VQE: hardware-efficient ansatz, no ancilla."""
        qr = QuantumRegister(n_sub, "q")
        cr = ClassicalRegister(n_sub, "c")
        qc = QuantumCircuit(qr, cr)
        qubits = list(range(n_sub))

        seed_offset = kwargs.get("seed_offset", 0)
        rng = np.random.default_rng(self.seed + seed_offset)
        params = self._get_params(n_layers, rng)

        # Initial state: |+⟩^n (good for TFIM with transverse field)
        for q in qubits:
            qc.h(q)

        for layer in range(n_layers):
            _hardware_efficient_layer(qc, qubits, params[layer])
            qc.barrier()

        qc.measure(qubits, list(range(n_sub)))
        return qc

    def _build_tsvf(
        self,
        n_sub: int,
        n_layers: int,
        seed_offset: int = 0,
    ) -> QuantumCircuit:
        """TSVF VQE: HW-efficient ansatz + chaotic layers + ancilla probe."""
        qr = QuantumRegister(n_sub, "q")
        anc_r = QuantumRegister(1, "anc")
        cr = ClassicalRegister(n_sub, "c_sys")
        cr_anc = ClassicalRegister(1, "c_anc")
        qc = QuantumCircuit(qr, anc_r, cr, cr_anc)

        qubits = list(range(n_sub))
        anc_qubit = n_sub
        rng = np.random.default_rng(self.seed + seed_offset)
        params = self._get_params(n_layers, rng)

        # Initial state: |+⟩^n
        for q in qubits:
            qc.h(q)

        for layer in range(n_layers):
            # Hardware-efficient ansatz layer
            _hardware_efficient_layer(qc, qubits, params[layer])
            qc.barrier()

            # Chaotic entangling ansatz
            _chaotic_vqe_ansatz(qc, qubits, layer=layer, rng=rng)
            qc.barrier()

            # Reset ancilla for reuse (except first layer)
            if layer > 0:
                qc.reset(anc_qubit)

            # Weak-measurement energy probe
            angle = self.weak_angle_base + self.weak_angle_ramp * min(layer, 4)
            _add_energy_probe_ancilla(
                qc,
                qubits,
                anc_qubit,
                cr_anc[0],
                n_qubits=self.n_qubits,
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

    def _to_search_counts(
        self,
        counts: dict[str, int],
        postselect: bool,
    ) -> dict[str, int]:
        """Convert raw counts to search-register-only counts.

        If postselect=True and mode is tsvf, only keep ancilla=1 shots.
        """
        search_counts: dict[str, int] = {}
        for key, val in counts.items():
            key_str = str(key)
            if self.algorithm_mode == "tsvf" and postselect:
                anc_bit, search_bits = self._split_ancilla_search(key_str)
                if anc_bit != "1":
                    continue
            else:
                search_bits = self._extract_search_bits(key_str)
            search_counts[search_bits] = search_counts.get(search_bits, 0) + val
        return search_counts

    def _extract_search_bits(self, bitstring: str) -> str:
        """Extract the system register bits from a bitstring."""
        key = bitstring.strip()
        if " " in key:
            return key.split()[-1]
        return key[-self.n_qubits :]

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
          - Ancilla=1 → evaluate per-qubit energy contribution:
            0 if qubit is in an aligned pair (low energy), 1 otherwise.
          - Ancilla=0 → row of 1s (all fail — no evidence of low energy).

        For the standard variant:
          - Evaluate per-qubit alignment from final measurement.
          - Replicate across all cycles.

        Shape: (n_cycles, n_subsystems).
        """
        if self.algorithm_mode == "tsvf":
            anc_bit, search_bits = self._split_ancilla_search(bitstring)

            if anc_bit == "1":
                qubit_quality = self._compute_qubit_energy_quality(
                    search_bits,
                    n_subsystems,
                )
                matrix = np.tile(qubit_quality, (n_cycles, 1))
            else:
                matrix = np.ones((n_cycles, n_subsystems), dtype=np.int8)
        else:
            search_bits = self._extract_search_bits(bitstring)
            qubit_quality = self._compute_qubit_energy_quality(
                search_bits,
                n_subsystems,
            )
            matrix = np.tile(qubit_quality, (n_cycles, 1))

        return matrix

    def _compute_qubit_energy_quality(
        self,
        search_bits: str,
        n_subsystems: int,
    ) -> np.ndarray:
        """Compute per-qubit energy quality.

        0 = qubit is aligned with at least one neighbour (low ZZ energy),
        1 = qubit is anti-aligned with all neighbours (high ZZ energy).

        For the TFIM with nearest-neighbour coupling, qubit i is "good"
        if it matches at least one of its neighbours (i−1 or i+1).
        """
        quality = np.ones(n_subsystems, dtype=np.int8)  # default: fail
        bits = search_bits[-n_subsystems:]
        for i in range(min(len(bits), n_subsystems)):
            # Check left neighbour
            if i > 0 and i - 1 < len(bits) and bits[i] == bits[i - 1]:
                quality[i] = 0
                continue
            # Check right neighbour
            if i < n_subsystems - 1 and i + 1 < len(bits) and bits[i] == bits[i + 1]:
                quality[i] = 0
        return quality
