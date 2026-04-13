"""
scoring.py — Score computation and fusion logic.

Extracted from the original ``monitors.py`` module to provide a clean,
stateless scoring API alongside the stateful :class:`MultiRateMonitor`.

Scoring is vectorised with NumPy:  :func:`score_batch` processes all
shots in a single array operation, avoiding per-shot Python loops.

Patent pending (see LICENSE)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

import numpy as np

from qgate.conditioning import ParityOutcome

logger = logging.getLogger("qgate.scoring")

# ---------------------------------------------------------------------------
# Single-shot fusion
# ---------------------------------------------------------------------------


def fuse_scores(
    lf_score: float,
    hf_score: float,
    alpha: float = 0.5,
    threshold: float = 0.65,
) -> tuple[bool, float]:
    """α-weighted fusion of LF and HF scores.

    combined = α · lf_score + (1 − α) · hf_score

    Args:
        lf_score:  Low-frequency score (0–1).
        hf_score:  High-frequency score (0–1).
        alpha:     LF weight (0 ≤ α ≤ 1).
        threshold: Accept if combined ≥ threshold.

    Returns:
        (accepted, combined_score)
    """
    combined = alpha * lf_score + (1.0 - alpha) * hf_score
    return combined >= threshold, float(combined)


# ---------------------------------------------------------------------------
# Outcome-level scoring
# ---------------------------------------------------------------------------


def score_outcome(
    outcome: ParityOutcome,
    alpha: float = 0.5,
    hf_cycles: Sequence[int] | None = None,
    lf_cycles: Sequence[int] | None = None,
) -> tuple[float, float, float]:
    """Compute LF, HF, and combined scores for a single shot outcome.

    Args:
        outcome:   Parity outcome.
        alpha:     LF weight in the combined score.
        hf_cycles: Explicit HF cycle indices (default: all).
        lf_cycles: Explicit LF cycle indices (default: even).

    Returns:
        (lf_score, hf_score, combined_score)
    """
    if hf_cycles is None:
        hf_cycles = list(range(outcome.n_cycles))
    if lf_cycles is None:
        lf_cycles = [w for w in range(outcome.n_cycles) if w % 2 == 0]

    rates = outcome.pass_rates  # (n_cycles,) — vectorised

    lf = float(np.mean(rates[list(lf_cycles)])) if lf_cycles else 0.0
    hf = float(np.mean(rates[list(hf_cycles)])) if hf_cycles else 0.0
    combined = alpha * lf + (1.0 - alpha) * hf
    return lf, hf, float(combined)


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------


def score_batch(
    outcomes: Sequence[ParityOutcome],
    alpha: float = 0.5,
    hf_cycles: Sequence[int] | None = None,
    lf_cycles: Sequence[int] | None = None,
) -> list[tuple[float, float, float]]:
    """Score every outcome in a batch (vectorised).

    When all outcomes share the same shape the scoring is performed as a
    single NumPy operation on a stacked 3-D array.  Falls back to
    per-shot scoring when shapes differ.

    Returns:
        List of ``(lf_score, hf_score, combined_score)`` per shot.
    """
    if not outcomes:
        return []

    n_cyc = outcomes[0].n_cycles
    hf_idx = np.arange(n_cyc) if hf_cycles is None else np.asarray(hf_cycles)
    lf_idx = np.arange(0, n_cyc, 2) if lf_cycles is None else np.asarray(lf_cycles)

    # Fast path: stack all matrices and compute in one shot
    try:
        all_matrices = np.stack([o.parity_matrix for o in outcomes])  # (N, cycles, subs)
        pass_rates = 1.0 - all_matrices.astype(np.float64).mean(axis=2)  # (N, cycles)
        lf_scores = pass_rates[:, lf_idx].mean(axis=1) if lf_idx.size else np.zeros(len(outcomes))
        hf_scores = pass_rates[:, hf_idx].mean(axis=1) if hf_idx.size else np.zeros(len(outcomes))
        combined = alpha * lf_scores + (1.0 - alpha) * hf_scores
        return list(zip(lf_scores.tolist(), hf_scores.tolist(), combined.tolist()))
    except (ValueError, IndexError):
        logger.debug("score_batch: shapes differ — falling back to per-shot scoring")
        return [
            score_outcome(o, alpha=alpha, hf_cycles=hf_cycles, lf_cycles=lf_cycles)
            for o in outcomes
        ]


# ---------------------------------------------------------------------------
# Window metrics (fidelity trajectories)
# ---------------------------------------------------------------------------


def compute_window_metric(
    times: np.ndarray,
    values: np.ndarray,
    window: float = 1.0,
    mode: Literal["max", "mean"] = "max",
) -> tuple[float, float, float]:
    """Compute a metric over a trailing time window.

    Examines [t_final − window, t_final] and returns the max or mean
    of *values* within that interval.

    Args:
        times:  1-D monotonic time array.
        values: 1-D values array (same length).
        window: Width of the trailing window.
        mode:   ``"max"`` or ``"mean"``.

    Returns:
        (metric, window_start, window_end)
    """
    t_final = float(times[-1])
    window_start = max(0.0, t_final - window)
    mask = (times >= window_start) & (times <= t_final)
    window_values = values[mask]

    if len(window_values) == 0:
        metric = float(values[-1])
    elif mode == "max":
        metric = float(np.max(window_values))
    elif mode == "mean":
        metric = float(np.mean(window_values))
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    return metric, window_start, t_final
