"""
cirq_adapter.py — Cirq adapter stub for qgate.

.. note::
   This is a **stub** — contributions welcome!

   Requires the ``cirq`` extra::

       pip install qgate[cirq]

Patent pending (see LICENSE)
"""

from __future__ import annotations

from typing import Any

from qgate.adapters.base import BaseAdapter
from qgate.conditioning import ParityOutcome

try:
    import cirq  # type: ignore[import-untyped]  # noqa: F401

    HAS_CIRQ = True
except ImportError:
    HAS_CIRQ = False


class CirqAdapter(BaseAdapter):
    """Cirq adapter — **stub implementation**.

    Raises ``NotImplementedError`` for all operations.
    Pull requests welcome!
    """

    def __init__(self, **kwargs: Any) -> None:
        if not HAS_CIRQ:
            raise ImportError(
                "Cirq is required for CirqAdapter.  Install with:  pip install qgate[cirq]"
            )

    def build_circuit(self, n_subsystems: int, n_cycles: int, **kwargs: Any) -> Any:
        raise NotImplementedError("CirqAdapter.build_circuit is a stub — contributions welcome!")

    def run(self, circuit: Any, shots: int, **kwargs: Any) -> Any:
        raise NotImplementedError("CirqAdapter.run is a stub — contributions welcome!")

    def parse_results(
        self, raw_results: Any, n_subsystems: int, n_cycles: int
    ) -> list[ParityOutcome]:
        raise NotImplementedError("CirqAdapter.parse_results is a stub — contributions welcome!")
