"""
monitors.py — Backward-compatible shim.

.. deprecated:: 0.3.0
   The canonical location is :mod:`qgate.compat.monitors`.
   This module re-exports all public symbols so that existing
   ``from qgate.monitors import ...`` statements continue to work.

Usage unchanged::

    from qgate.monitors import MultiRateMonitor, score_fusion
"""
from qgate.compat.monitors import (
    MultiRateMonitor,
    compute_window_metric,
    score_fusion,
    should_abort_batch,
)

__all__ = [
    "MultiRateMonitor",
    "compute_window_metric",
    "score_fusion",
    "should_abort_batch",
]
