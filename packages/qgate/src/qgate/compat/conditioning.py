"""
conditioning.py — Post-selection conditioning decision rules.

Three strategies for deciding whether a quantum computation's mid-circuit
measurement outcomes should be accepted or rejected:

  1. **Global** — Every subsystem must pass every monitored cycle.
  2. **Hierarchical k-of-N** — At least ⌈k·N⌉ subsystems pass per cycle.
  3. **Score fusion** — Weighted blend of multi-rate scores exceeds a
     continuous threshold.

All functions are pure (no side effects) and operate on plain Python /
NumPy data structures so they work with any backend.

The :class:`ParityOutcome` dataclass stores parity matrices as
``np.ndarray`` for vectorised scoring.  The constructor transparently
accepts both ``list[list[int]]`` and ``np.ndarray`` inputs so all
existing code continues to work.

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Union

import numpy as np

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class ParityOutcome:
    """Parsed mid-circuit measurement outcomes for one shot.

    Attributes:
        n_subsystems: Number of subsystems (Bell pairs or logical units).
        n_cycles:     Number of monitoring cycles.
        parity_matrix: Shape ``(n_cycles, n_subsystems)`` — ``0`` = pass
                       (even parity), ``1`` = fail (odd parity).
                       Accepts ``list[list[int]]`` *or* ``np.ndarray``
                       on construction (coerced to ``np.ndarray``).
    """

    n_subsystems: int
    n_cycles: int
    parity_matrix: Union[np.ndarray, list] = field(default_factory=list)  # noqa: UP007

    def __post_init__(self) -> None:
        if not isinstance(self.parity_matrix, np.ndarray):
            self.parity_matrix = np.asarray(self.parity_matrix, dtype=np.int8)
        if self.parity_matrix.ndim == 0 or self.parity_matrix.size == 0:
            self.parity_matrix = np.zeros((self.n_cycles, self.n_subsystems), dtype=np.int8)

    # Convenience -----------------------------------------------------------

    def subsystem_pass_count(self, cycle: int) -> int:
        """Number of subsystems that passed in *cycle*."""
        return int(np.sum(self.parity_matrix[cycle] == 0))

    def subsystem_pass_rate(self, cycle: int) -> float:
        """Fraction of subsystems that passed in *cycle*."""
        return float(1.0 - self.parity_matrix[cycle].mean())

    def cycle_all_pass(self, cycle: int) -> bool:
        """True if every subsystem passed in *cycle*."""
        return bool(np.all(self.parity_matrix[cycle] == 0))

    @property
    def pass_rates(self) -> np.ndarray:
        """Per-cycle pass rate — shape ``(n_cycles,)``."""
        mat: np.ndarray = self.parity_matrix  # type: ignore[assignment]  # coerced in __post_init__
        result: np.ndarray = 1.0 - mat.astype(np.float64).mean(axis=1)
        return result


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------


def decide_global(outcome: ParityOutcome) -> bool:
    """Global conditioning — all subsystems pass all cycles.

    Args:
        outcome: Parity outcome for one shot.

    Returns:
        True if the shot should be accepted.

    Example::

        outcome = ParityOutcome(n_subsystems=4, n_cycles=2,
                                parity_matrix=[[0,0,0,0], [0,0,0,0]])
        assert decide_global(outcome) is True
    """
    return bool(np.all(outcome.parity_matrix == 0))


def decide_hierarchical(
    outcome: ParityOutcome,
    k_fraction: float = 0.9,
) -> bool:
    """Hierarchical k-of-N conditioning.

    Accepts if at least ⌈k·N⌉ subsystems pass in **every** cycle.

    Args:
        outcome:    Parity outcome for one shot.
        k_fraction: Required pass fraction (0 < k_fraction ≤ 1).

    Returns:
        True if the shot should be accepted.

    Example::

        outcome = ParityOutcome(n_subsystems=4, n_cycles=1,
                                parity_matrix=[[0, 0, 1, 0]])
        # ceil(0.75 * 4) = 3  →  3 passed  →  accept
        assert decide_hierarchical(outcome, k_fraction=0.75) is True
    """
    if not 0 < k_fraction <= 1:
        raise ValueError(f"k_fraction must be in (0, 1], got {k_fraction}")
    threshold = math.ceil(k_fraction * outcome.n_subsystems)
    # Per-cycle pass counts via vectorised sum
    pass_counts = np.sum(outcome.parity_matrix == 0, axis=1)  # (n_cycles,)
    return bool(np.all(pass_counts >= threshold))


def decide_score_fusion(
    outcome: ParityOutcome,
    alpha: float = 0.5,
    threshold_combined: float = 0.65,
    hf_cycles: Sequence[int] | None = None,
    lf_cycles: Sequence[int] | None = None,
) -> tuple[bool, float]:
    """Score-fusion conditioning.

    Computes a weighted combination of low-frequency (LF) and
    high-frequency (HF) subsystem pass-rates and compares to a
    continuous threshold:

        combined = α · mean(LF pass-rates) + (1-α) · mean(HF pass-rates)

    By default:
      - HF cycles = every cycle
      - LF cycles = every 2nd cycle (0, 2, 4, …)

    Args:
        outcome:            Parity outcome for one shot.
        alpha:              Weight for LF component (0 ≤ α ≤ 1).
        threshold_combined: Accept if combined ≥ this value.
        hf_cycles:          Override which cycles count as HF.
        lf_cycles:          Override which cycles count as LF.

    Returns:
        (accepted, combined_score)

    Example::

        outcome = ParityOutcome(n_subsystems=2, n_cycles=4,
                                parity_matrix=[[0,0],[0,1],[0,0],[1,0]])
        accepted, score = decide_score_fusion(outcome, alpha=0.5)
    """
    if hf_cycles is None:
        hf_cycles = list(range(outcome.n_cycles))
    if lf_cycles is None:
        lf_cycles = [w for w in range(outcome.n_cycles) if w % 2 == 0]

    def _mean_rate(cycles: Sequence[int]) -> float:
        if not cycles:
            return 0.0
        rates = outcome.pass_rates
        return float(np.mean(rates[list(cycles)]))

    score_lf = _mean_rate(lf_cycles)
    score_hf = _mean_rate(hf_cycles)
    combined = alpha * score_lf + (1.0 - alpha) * score_hf
    return combined >= threshold_combined, float(combined)


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


@dataclass
class ConditioningStats:
    """Aggregated statistics after applying a conditioning rule to many shots."""

    variant: str
    total_shots: int = 0
    accepted_shots: int = 0
    scores: list[float] = field(default_factory=list)

    @property
    def acceptance_probability(self) -> float:
        return self.accepted_shots / self.total_shots if self.total_shots else 0.0

    @property
    def tts(self) -> float:
        """Time-to-solution: 1 / acceptance_probability."""
        p = self.acceptance_probability
        return 1.0 / p if p > 0 else float("inf")

    def as_dict(self) -> dict:
        return {
            "variant": self.variant,
            "total_shots": self.total_shots,
            "accepted_shots": self.accepted_shots,
            "acceptance_probability": self.acceptance_probability,
            "TTS": self.tts,
            "mean_score": float(np.mean(self.scores)) if self.scores else None,
        }


def apply_rule_to_batch(
    outcomes: Sequence[ParityOutcome],
    variant: str = "global",
    k_fraction: float = 0.9,
    alpha: float = 0.5,
    threshold_combined: float = 0.65,
) -> ConditioningStats:
    """Apply a conditioning rule to a batch of parity outcomes.

    Args:
        outcomes:           Sequence of ParityOutcome (one per shot).
        variant:            ``"global"`` | ``"hierarchical"`` | ``"score_fusion"``.
        k_fraction:         For hierarchical rule.
        alpha:              For score fusion.
        threshold_combined: For score fusion.

    Returns:
        ConditioningStats with acceptance statistics.
    """
    stats = ConditioningStats(variant=variant)
    for outcome in outcomes:
        stats.total_shots += 1
        if variant == "global":
            if decide_global(outcome):
                stats.accepted_shots += 1
        elif variant == "hierarchical":
            if decide_hierarchical(outcome, k_fraction):
                stats.accepted_shots += 1
        elif variant == "score_fusion":
            accepted, score = decide_score_fusion(outcome, alpha, threshold_combined)
            stats.scores.append(score)
            if accepted:
                stats.accepted_shots += 1
        else:
            raise ValueError(f"Unknown variant: {variant}")
    return stats
