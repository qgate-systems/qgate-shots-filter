"""
conditioning.py — Backward-compatible shim.

.. deprecated:: 0.3.0
   The canonical location is :mod:`qgate.compat.conditioning`.
   This module re-exports all public symbols so that existing
   ``from qgate.conditioning import ...`` statements continue to work.

Usage unchanged::

    from qgate.conditioning import ParityOutcome, decide_global
"""
from qgate.compat.conditioning import (
    ConditioningStats,
    ParityOutcome,
    apply_rule_to_batch,
    decide_global,
    decide_hierarchical,
    decide_score_fusion,
)

__all__ = [
    "ConditioningStats",
    "ParityOutcome",
    "apply_rule_to_batch",
    "decide_global",
    "decide_hierarchical",
    "decide_score_fusion",
]
