"""
circuits.py – Quantum circuit construction for post-selection conditioning.

Builds parameterised circuits with:
  • N Bell-pair subsystems (2 qubits each, up to N=8 → 16 qubits).
  • D layers of scramble depth (random single-qubit gates + barriers for
    idle-noise exposure).
  • W monitoring cycles, each containing a mid-circuit Z-parity measurement
    per subsystem (even parity → pass).
  • Multi-rate labels: HF probe every cycle, LF probe every 2 cycles.

All mid-circuit measurement results are stored in classical registers so
that the Conditioning engine can post-process them.

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
"""
from __future__ import annotations

import math
import hashlib
from typing import Optional

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RXGate, RYGate, RZGate
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deterministic_rng(seed_str: str) -> np.random.Generator:
    """SHA-256–based deterministic RNG from an arbitrary string key."""
    h = hashlib.sha256(seed_str.encode()).hexdigest()
    seed = int(h[:16], 16) % (2**63)
    return np.random.default_rng(seed)


def _scramble_layer(qc: QuantumCircuit, qubits: list[int],
                    rng: np.random.Generator, layer_idx: int) -> None:
    """Apply one scramble layer: random single-qubit rotations + barrier.

    The barrier forces the transpiler to keep the layer intact, exposing
    qubits to idle noise on real hardware.
    """
    gates = [RXGate, RYGate, RZGate]
    for q in qubits:
        gate_cls = gates[rng.integers(0, 3)]
        angle = rng.uniform(0, 2 * math.pi)
        qc.append(gate_cls(angle), [q])
    qc.barrier()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_bell_pairs(qc: QuantumCircuit, n_subsystems: int,
                     qubit_offset: int = 0) -> list[tuple[int, int]]:
    """Create N Bell pairs |Φ+⟩ = (|00⟩ + |11⟩)/√2.

    Returns list of (qubit_a, qubit_b) pairs.
    """
    pairs = []
    for i in range(n_subsystems):
        qa = qubit_offset + 2 * i
        qb = qubit_offset + 2 * i + 1
        qc.h(qa)
        qc.cx(qa, qb)
        pairs.append((qa, qb))
    qc.barrier()
    return pairs


def add_scramble_block(qc: QuantumCircuit, pairs: list[tuple[int, int]],
                       depth: int, rng: np.random.Generator) -> None:
    """Apply D layers of scramble (random rotations + barriers)."""
    all_qubits = [q for pair in pairs for q in pair]
    for d in range(depth):
        _scramble_layer(qc, all_qubits, rng, d)


def add_zparity_measurement(
    qc: QuantumCircuit,
    pairs: list[tuple[int, int]],
    cycle: int,
    ancilla_reg: QuantumRegister,
    creg: ClassicalRegister,
) -> None:
    """Mid-circuit Z-parity measurement for each subsystem.

    For Bell pair (qa, qb), parity = Z_a ⊗ Z_b.  We use an ancilla:
        CNOT(qa, anc); CNOT(qb, anc); measure(anc) → 0 means even parity (pass).
    The ancilla is reset after each measurement so it can be reused.

    Args:
        qc:           target circuit
        pairs:        list of (qa, qb) tuples
        cycle:        monitoring cycle index (used for classical bit offset)
        ancilla_reg:  single-qubit ancilla register
        creg:         classical register sized N_subsystems * W_cycles
    """
    n = len(pairs)
    for sub_idx, (qa, qb) in enumerate(pairs):
        anc = ancilla_reg[0]
        # Reset ancilla to |0⟩
        qc.reset(anc)
        # Compute parity onto ancilla
        qc.cx(qa, anc)
        qc.cx(qb, anc)
        # Mid-circuit measurement
        bit_idx = cycle * n + sub_idx
        qc.measure(anc, creg[bit_idx])
    qc.barrier()


def build_monitoring_circuit(
    n_subsystems: int,
    depth: int,
    n_cycles: int,
    seed: Optional[str] = None,
) -> QuantumCircuit:
    """Build a complete monitoring circuit.

    Layout:
        [Bell-pair creation] → for each cycle:
            [Scramble block (depth D)] → [Z-parity mid-circuit measure]
        → [final barrier]

    Registers:
        q_data:   2*N data qubits
        q_anc:    1 ancilla (reused across subsystems/cycles)
        c_parity: N*W classical bits for parity outcomes
                  bit index = cycle * N + subsystem_index

    The circuit stores ALL mid-circuit measurement results so that the
    conditioning engine can classify each shot.
    """
    if seed is None:
        seed = f"monitor_N{n_subsystems}_D{depth}_W{n_cycles}"
    rng = _deterministic_rng(seed)

    n_qubits = 2 * n_subsystems
    q_data = QuantumRegister(n_qubits, name="q")
    q_anc = QuantumRegister(1, name="anc")
    c_parity = ClassicalRegister(n_subsystems * n_cycles, name="parity")

    qc = QuantumCircuit(q_data, q_anc, c_parity,
                        name=f"monitor_N{n_subsystems}_D{depth}_W{n_cycles}")

    # --- Bell pair creation ---
    pairs = build_bell_pairs(qc, n_subsystems, qubit_offset=0)

    # --- Monitoring cycles ---
    for w in range(n_cycles):
        # Scramble block (noise exposure)
        add_scramble_block(qc, pairs, depth, rng)
        # Z-parity measurement
        add_zparity_measurement(qc, pairs, w, q_anc, c_parity)

    qc.barrier()
    return qc


def build_probe_circuit(
    n_subsystems: int,
    depth: int,
    seed: Optional[str] = None,
) -> QuantumCircuit:
    """Build a short W=1 probe circuit for batch-level abort.

    Identical to the full monitoring circuit but with only 1 cycle.
    Used to screen configurations before committing to the full run.
    """
    return build_monitoring_circuit(n_subsystems, depth, n_cycles=1,
                                   seed=seed or f"probe_N{n_subsystems}_D{depth}")
