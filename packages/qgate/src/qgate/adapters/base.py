"""
base.py — Adapter protocol and mock implementation.

Every adapter must implement :class:`BaseAdapter` so that
:class:`~qgate.filter.TrajectoryFilter` can work with any quantum
framework.

Patent pending (see LICENSE)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from qgate.conditioning import ParityOutcome


class BaseAdapter(ABC):
    """Abstract base class that all qgate adapters must implement.

    The adapter is responsible for:
      1. Building circuits with Bell-pair subsystems and parity checks.
      2. Executing shots on a backend (simulator or hardware).
      3. Parsing raw results into ``ParityOutcome`` objects.
    """

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def build_circuit(
        self,
        n_subsystems: int,
        n_cycles: int,
        **kwargs: Any,
    ) -> Any:
        """Construct a circuit with *n_subsystems* Bell pairs and
        *n_cycles* mid-circuit parity checks.

        Returns a framework-native circuit object.
        """

    @abstractmethod
    def run(
        self,
        circuit: Any,
        shots: int,
        **kwargs: Any,
    ) -> Any:
        """Execute *circuit* for *shots* repetitions.

        Returns the framework-native result object.
        """

    @abstractmethod
    def parse_results(
        self,
        raw_results: Any,
        n_subsystems: int,
        n_cycles: int,
    ) -> list[ParityOutcome]:
        """Parse framework-native results into a list of
        ``ParityOutcome`` objects (one per shot).
        """

    # ------------------------------------------------------------------
    # Optional helpers
    # ------------------------------------------------------------------

    def build_and_run(
        self,
        n_subsystems: int,
        n_cycles: int,
        shots: int,
        circuit_kwargs: dict[str, Any] | None = None,
        run_kwargs: dict[str, Any] | None = None,
    ) -> list[ParityOutcome]:
        """Convenience: build → run → parse in one call."""
        circuit = self.build_circuit(n_subsystems, n_cycles, **(circuit_kwargs or {}))
        raw = self.run(circuit, shots, **(run_kwargs or {}))
        return self.parse_results(raw, n_subsystems, n_cycles)


# ---------------------------------------------------------------------------
# Mock adapter (for testing)
# ---------------------------------------------------------------------------


class MockAdapter(BaseAdapter):
    """In-memory adapter that generates synthetic parity outcomes.

    Useful for unit tests and demonstrations without a real backend.

    Args:
        error_rate: Per-subsystem per-cycle probability of a parity flip
                    (default 0.05 → 5 %).
        seed:       Optional random seed for reproducibility.
    """

    def __init__(self, error_rate: float = 0.05, seed: int | None = None) -> None:
        self.error_rate = error_rate
        self._rng = random.Random(seed)

    def build_circuit(
        self,
        n_subsystems: int,
        n_cycles: int,
        **kwargs: Any,
    ) -> dict[str, int]:
        """Return a lightweight descriptor (no real circuit)."""
        return {"n_subsystems": n_subsystems, "n_cycles": n_cycles}

    def run(
        self,
        circuit: Any,
        shots: int,
        **kwargs: Any,
    ) -> list[list[list[int]]]:
        """Generate *shots* synthetic parity matrices."""
        n_sub = circuit["n_subsystems"]
        n_cyc = circuit["n_cycles"]
        results: list[list[list[int]]] = []
        for _ in range(shots):
            matrix = [
                [1 if self._rng.random() < self.error_rate else 0 for _ in range(n_sub)]
                for _ in range(n_cyc)
            ]
            results.append(matrix)
        return results

    def parse_results(
        self,
        raw_results: Any,
        n_subsystems: int,
        n_cycles: int,
    ) -> list[ParityOutcome]:
        """Wrap raw matrices in ParityOutcome."""
        return [
            ParityOutcome(
                n_subsystems=n_subsystems,
                n_cycles=n_cycles,
                parity_matrix=matrix,
            )
            for matrix in raw_results
        ]
