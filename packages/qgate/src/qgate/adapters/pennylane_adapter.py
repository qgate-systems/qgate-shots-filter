"""
pennylane_adapter.py — PennyLane adapter stub for qgate.

.. note::
   This is a **stub** — contributions welcome!

   Requires the ``pennylane`` extra::

       pip install qgate[pennylane]

Patent pending (see LICENSE)
"""

from __future__ import annotations

from typing import Any

from qgate.adapters.base import BaseAdapter
from qgate.conditioning import ParityOutcome

try:
    import pennylane  # type: ignore[import-untyped]  # noqa: F401

    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


class PennyLaneAdapter(BaseAdapter):
    """PennyLane adapter — **stub implementation**.

    Raises ``NotImplementedError`` for all operations.
    Pull requests welcome!
    """

    def __init__(self, **kwargs: Any) -> None:
        if not HAS_PENNYLANE:
            raise ImportError(
                "PennyLane is required for PennyLaneAdapter.  "
                "Install with:  pip install qgate[pennylane]"
            )

    def build_circuit(self, n_subsystems: int, n_cycles: int, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "PennyLaneAdapter.build_circuit is a stub — contributions welcome!"
        )

    def run(self, circuit: Any, shots: int, **kwargs: Any) -> Any:
        raise NotImplementedError("PennyLaneAdapter.run is a stub — contributions welcome!")

    def parse_results(
        self, raw_results: Any, n_subsystems: int, n_cycles: int
    ) -> list[ParityOutcome]:
        raise NotImplementedError(
            "PennyLaneAdapter.parse_results is a stub — contributions welcome!"
        )
