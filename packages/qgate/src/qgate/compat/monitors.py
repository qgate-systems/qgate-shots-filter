"""
monitors.py — Multi-rate fidelity monitoring and score fusion.

Provides the continuous monitoring primitives that feed into the
conditioning decision rules:

  • **Window metrics** — Compute max / mean fidelity over a trailing
    time window (used for QuTiP / analytical simulations).
  • **Score fusion** — α-weighted blend of low-frequency and
    high-frequency scores.
  • **MultiRateMonitor** — Stateful monitor that tracks HF and LF
    scores over multiple cycles and produces a fused decision.
  • **Batch abort** — Probe-based early rejection of configurations
    unlikely to yield useful data.

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Re-export from the canonical location to avoid code duplication.
from qgate.scoring import compute_window_metric as compute_window_metric

# ---------------------------------------------------------------------------
# Score fusion
# ---------------------------------------------------------------------------


def score_fusion(
    lf_score: float,
    hf_score: float,
    alpha: float = 0.5,
    threshold: float = 0.65,
) -> tuple[bool, float]:
    """Compute α-weighted score fusion and compare to threshold.

    combined = α · lf_score + (1 − α) · hf_score

    Args:
        lf_score:  Low-frequency monitoring score (0–1).
        hf_score:  High-frequency monitoring score (0–1).
        alpha:     Weight for the LF component (0 ≤ α ≤ 1).
        threshold: Accept if combined ≥ threshold.

    Returns:
        (accepted, combined_score)

    Example::

        accepted, score = score_fusion(0.8, 0.6, alpha=0.5, threshold=0.65)
        # score = 0.5*0.8 + 0.5*0.6 = 0.70  → accepted = True
    """
    combined = alpha * lf_score + (1.0 - alpha) * hf_score
    return combined >= threshold, float(combined)


# ---------------------------------------------------------------------------
# Multi-rate monitor
# ---------------------------------------------------------------------------


@dataclass
class MultiRateMonitor:
    """Stateful monitor tracking HF and LF parity scores across cycles.

    Usage::

        mon = MultiRateMonitor(n_subsystems=4, alpha=0.5,
                               threshold_combined=0.65)
        mon.record_cycle(0, pass_rates=0.75)   # cycle 0 → HF + LF
        mon.record_cycle(1, pass_rates=0.50)   # cycle 1 → HF only
        mon.record_cycle(2, pass_rates=0.80)   # cycle 2 → HF + LF
        accepted, score = mon.fused_decision()

    Attributes:
        n_subsystems:       Number of subsystems being monitored.
        alpha:              LF weight in fusion formula.
        threshold_combined: Accept if fused score ≥ this.
        hf_scores:          Recorded HF scores (every cycle).
        lf_scores:          Recorded LF scores (even cycles only).
    """

    n_subsystems: int = 1
    alpha: float = 0.5
    threshold_combined: float = 0.65
    hf_scores: list[float] = field(default_factory=list)
    lf_scores: list[float] = field(default_factory=list)

    def record_cycle(self, cycle_idx: int, pass_rate: float) -> None:
        """Record the subsystem pass-rate for a cycle.

        The score is always recorded as HF.  If the cycle index is even,
        it is also recorded as LF.

        Args:
            cycle_idx: Zero-based cycle index.
            pass_rate: Fraction of subsystems that passed (0–1).
        """
        self.hf_scores.append(pass_rate)
        if cycle_idx % 2 == 0:
            self.lf_scores.append(pass_rate)

    def fused_decision(self) -> tuple[bool, float]:
        """Compute the fused decision from accumulated scores.

        Returns:
            (accepted, combined_score)
        """
        lf = float(np.mean(self.lf_scores)) if self.lf_scores else 0.0
        hf = float(np.mean(self.hf_scores)) if self.hf_scores else 0.0
        return score_fusion(lf, hf, self.alpha, self.threshold_combined)

    def reset(self) -> None:
        """Clear all recorded scores."""
        self.hf_scores.clear()
        self.lf_scores.clear()


# ---------------------------------------------------------------------------
# Batch-level abort
# ---------------------------------------------------------------------------


def should_abort_batch(
    probe_pass_rate: float,
    theta: float = 0.65,
) -> bool:
    """Decide whether to abort a full batch based on a probe result.

    Args:
        probe_pass_rate: Fraction of probe shots that passed (0–1).
        theta:           Proceed only if probe_pass_rate ≥ θ.

    Returns:
        True if the batch should be **aborted** (i.e. probe failed).

    Example::

        # Probe returned 30% pass-rate  →  abort
        assert should_abort_batch(0.30, theta=0.65) is True
    """
    return probe_pass_rate < theta
