"""
qgate.adapters — Backend adapter protocol and implementations.

Each adapter translates between a quantum framework's native circuit /
result types and qgate's internal ``ParityOutcome`` representation.

Available adapters:
  • ``QiskitAdapter``       — Full implementation for IBM Qiskit.
  • ``GroverTSVFAdapter``   — Grover/TSVF search experiment adapter.
  • ``QAOATSVFAdapter``     — QAOA/TSVF MaxCut experiment adapter.
  • ``VQETSVFAdapter``      — VQE/TSVF ground-state energy adapter.
  • ``QPETSVFAdapter``      — QPE/TSVF phase estimation adapter.
  • ``CirqAdapter``         — Stub (contrib welcome).
  • ``PennyLaneAdapter``    — Stub (contrib welcome).
  • ``MockAdapter``         — In-memory adapter for testing.

Adapter discovery::

    from qgate.adapters import list_adapters, load_adapter

    print(list_adapters())            # {"mock": "qgate.adapters.base:MockAdapter", ...}
    AdapterCls = load_adapter("mock")
"""

from qgate.adapters.base import BaseAdapter, MockAdapter
from qgate.adapters.registry import list_adapters, load_adapter

__all__ = ["BaseAdapter", "MockAdapter", "list_adapters", "load_adapter"]
