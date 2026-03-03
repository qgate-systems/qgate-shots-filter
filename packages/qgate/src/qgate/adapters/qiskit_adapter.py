"""
qiskit_adapter.py — Full Qiskit adapter for qgate.

Builds dynamic circuits with Bell-pair subsystems, scramble layers,
and ancilla-based mid-circuit Z-parity measurements.

Requires the ``qiskit`` extra::

    pip install qgate[qiskit]

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from qgate.adapters.base import BaseAdapter
from qgate.conditioning import ParityOutcome

logger = logging.getLogger("qgate.adapters.qiskit")

try:
    from qiskit import QuantumCircuit  # type: ignore[import-untyped]
    from qiskit.circuit import ClassicalRegister  # type: ignore[import-untyped]

    HAS_QISKIT = True
except ImportError:  # pragma: no cover
    HAS_QISKIT = False


def _require_qiskit() -> None:
    if not HAS_QISKIT:
        raise ImportError(
            "Qiskit is required for QiskitAdapter.  "
            "Install with:  pip install qgate[qiskit]"
        )


class QiskitAdapter(BaseAdapter):
    """Adapter for IBM Qiskit circuits.

    Builds dynamic circuits with:
      * N Bell pairs (2N data qubits)
      * W monitoring cycles each containing:
        - Random single-qubit scramble rotations
        - Ancilla-based Z⊗Z parity measurement per pair
        - Ancilla reset & reuse

    Args:
        backend:         Qiskit backend or ``None`` for Aer simulator.
        scramble_depth:  Number of random-rotation layers per cycle.
        optimization_level: Transpiler optimization level (0–3).
    """

    def __init__(
        self,
        backend: Any = None,
        scramble_depth: int = 1,
        optimization_level: int = 1,
    ) -> None:
        _require_qiskit()
        self._backend = backend
        self.scramble_depth = scramble_depth
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
        """Build a dynamic Qiskit circuit.

        Qubit layout:
          * data qubits   : 0 .. 2N-1  (pairs: [0,1], [2,3], …)
          * ancilla qubits: 2N .. 3N-1  (one per pair)

        Classical registers — one per cycle, each of width N
        (bit *i* records the parity of pair *i*).
        """
        import numpy as np

        n_data = 2 * n_subsystems
        n_anc = n_subsystems
        qc = QuantumCircuit(n_data + n_anc)

        # Classical registers — one per monitoring cycle
        cregs = []
        for w in range(n_cycles):
            cr = ClassicalRegister(n_subsystems, name=f"par_c{w}")
            qc.add_register(cr)
            cregs.append(cr)

        # Bell-pair preparation
        for i in range(n_subsystems):
            qc.h(2 * i)
            qc.cx(2 * i, 2 * i + 1)
        qc.barrier()

        rng = np.random.default_rng(kwargs.get("seed"))

        for w in range(n_cycles):
            # ── Scramble rotations ──
            for _ in range(self.scramble_depth):
                for q in range(n_data):
                    theta, phi, lam = rng.uniform(0, 0.3, size=3)
                    qc.u(theta, phi, lam, q)
            qc.barrier()

            # ── Z⊗Z parity measurement via ancilla ──
            for i in range(n_subsystems):
                anc = n_data + i
                qc.cx(2 * i, anc)
                qc.cx(2 * i + 1, anc)
                qc.measure(anc, cregs[w][i])
                qc.reset(anc)
            qc.barrier()

        return qc

    def run(
        self,
        circuit: Any,
        shots: int,
        **kwargs: Any,
    ) -> Any:
        """Execute via the configured backend (Aer if none)."""
        backend = self._backend
        if backend is None:
            from qiskit_aer import AerSimulator  # type: ignore[import-untyped]

            backend = AerSimulator()

        from qiskit import transpile  # type: ignore[import-untyped]

        transpiled = transpile(
            circuit,
            backend=backend,
            optimization_level=self.optimization_level,
        )
        job = backend.run(transpiled, shots=shots, **kwargs)
        return job.result()

    def parse_results(
        self,
        raw_results: Any,
        n_subsystems: int,
        n_cycles: int,
    ) -> list[ParityOutcome]:
        """Parse Qiskit ``Result`` into ``ParityOutcome`` objects."""
        counts: dict[str, int] = raw_results.get_counts()
        outcomes: list[ParityOutcome] = []

        for bitstring, count in counts.items():
            # Qiskit returns bits in reverse register order
            # Format: "par_cW-1 … par_c0" each of width n_subsystems
            segments = bitstring.strip().split(" ")
            # Reverse to get cycle order 0 → W-1
            segments = list(reversed(segments))

            matrix: list[list[int]] = []
            for seg in segments[:n_cycles]:
                # Reverse each segment so bit 0 = subsystem 0
                bits = [int(b) for b in reversed(seg)]
                # Pad or truncate to n_subsystems
                bits = (bits + [0] * n_subsystems)[:n_subsystems]
                matrix.append(bits)

            base_matrix = np.array(matrix, dtype=np.int8)
            # Each shot gets its own independent ndarray copy
            # to prevent aliasing mutations across shots.
            for _ in range(count):
                outcomes.append(ParityOutcome(
                    n_subsystems=n_subsystems,
                    n_cycles=n_cycles,
                    parity_matrix=base_matrix.copy(),
                ))

        logger.debug("Parsed %d outcomes from Qiskit result", len(outcomes))
        return outcomes
