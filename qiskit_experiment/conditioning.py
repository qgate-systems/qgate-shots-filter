"""
conditioning.py – Post-selection conditioning & decision rules.

Implements three conditioning strategies applied to mid-circuit Z-parity
measurement outcomes:

  1. **Global** – All subsystems must pass every monitored cycle.
  2. **Hierarchical k-of-N** – At least ⌈k·N⌉ subsystems pass each cycle.
  3. **Score fusion** – Weighted blend of HF and LF parity scores with a
     continuous threshold.

Also implements **batch-level abort**: run a short W=1 probe circuit first;
only submit the full circuit if the probe pass-rate exceeds θ.

Multi-rate labelling:
  • HF (high-frequency): every cycle
  • LF (low-frequency):  every 2 cycles (even-indexed: 0, 2, 4, …)

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ShotOutcome:
    """Parsed mid-circuit measurement outcome for a single shot."""
    n_subsystems: int
    n_cycles: int
    # parity_matrix[cycle][subsystem] = 0 (even/pass) or 1 (odd/fail)
    parity_matrix: list[list[int]] = field(default_factory=list)

    @classmethod
    def from_bitstring(cls, bitstring: str, n_subsystems: int,
                       n_cycles: int) -> "ShotOutcome":
        """Parse a Qiskit bitstring (LSB-first) into a parity matrix.

        Qiskit returns bitstrings in *reverse* bit order, so we reverse
        first, then index as bit[cycle * N + sub].
        """
        # Qiskit bitstrings: rightmost bit = classical bit 0
        bits = bitstring.replace(" ", "")
        bits_list = [int(b) for b in reversed(bits)]

        matrix: list[list[int]] = []
        for w in range(n_cycles):
            row = []
            for s in range(n_subsystems):
                idx = w * n_subsystems + s
                row.append(bits_list[idx] if idx < len(bits_list) else 1)
            matrix.append(row)

        return cls(n_subsystems=n_subsystems, n_cycles=n_cycles,
                   parity_matrix=matrix)

    # convenience ----------------------------------------------------------

    def subsystem_pass_rate(self, cycle: int) -> float:
        """Fraction of subsystems that passed (even parity) in a cycle."""
        row = self.parity_matrix[cycle]
        return sum(1 for v in row if v == 0) / len(row)

    def cycle_all_pass(self, cycle: int) -> bool:
        return all(v == 0 for v in self.parity_matrix[cycle])

    def hf_cycles(self) -> list[int]:
        """HF = every cycle."""
        return list(range(self.n_cycles))

    def lf_cycles(self) -> list[int]:
        """LF = every 2nd cycle (0, 2, 4, …)."""
        return [w for w in range(self.n_cycles) if w % 2 == 0]


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------

def decide_global(outcome: ShotOutcome) -> bool:
    """Global conditioning: ALL subsystems pass ALL cycles."""
    for w in range(outcome.n_cycles):
        if not outcome.cycle_all_pass(w):
            return False
    return True


def decide_hierarchical(outcome: ShotOutcome, k_fraction: float) -> bool:
    """Hierarchical k-of-N: ≥ ⌈k·N⌉ subsystems pass each cycle."""
    threshold = math.ceil(k_fraction * outcome.n_subsystems)
    for w in range(outcome.n_cycles):
        n_pass = sum(1 for v in outcome.parity_matrix[w] if v == 0)
        if n_pass < threshold:
            return False
    return True


def _rate_for_cycles(outcome: ShotOutcome, cycles: list[int]) -> float:
    """Average subsystem pass rate across the given cycles."""
    if not cycles:
        return 0.0
    return np.mean([outcome.subsystem_pass_rate(w) for w in cycles])


def decide_score_fusion(outcome: ShotOutcome, alpha: float,
                        threshold_combined: float) -> tuple[bool, float]:
    """Score fusion: combined = α·score_LF + (1-α)·score_HF ≥ θ.

    Returns (accepted, combined_score).
    """
    score_lf = _rate_for_cycles(outcome, outcome.lf_cycles())
    score_hf = _rate_for_cycles(outcome, outcome.hf_cycles())
    combined = alpha * score_lf + (1.0 - alpha) * score_hf
    return combined >= threshold_combined, float(combined)


# ---------------------------------------------------------------------------
# Batch-level abort
# ---------------------------------------------------------------------------

def evaluate_probe_batch(
    counts: dict[str, int],
    n_subsystems: int,
    theta: float = 0.65,
) -> tuple[bool, float]:
    """Decide whether to proceed based on a W=1 probe circuit.

    Args:
        counts:       Sampler result counts for the probe circuit.
        n_subsystems: number of Bell-pair subsystems.
        theta:        pass-rate threshold for the batch to proceed.

    Returns:
        (proceed, pass_rate)
    """
    total_shots = sum(counts.values())
    if total_shots == 0:
        return False, 0.0

    n_pass = 0
    for bitstring, count in counts.items():
        outcome = ShotOutcome.from_bitstring(bitstring, n_subsystems,
                                             n_cycles=1)
        if outcome.cycle_all_pass(0):
            n_pass += count

    pass_rate = n_pass / total_shots
    return pass_rate >= theta, pass_rate


# ---------------------------------------------------------------------------
# Aggregate statistics over a batch of shots
# ---------------------------------------------------------------------------

@dataclass
class ConditioningResult:
    """Aggregated results for one configuration."""
    variant: str                  # "global", "hierarchical", "score_fusion"
    n_subsystems: int
    depth: int
    n_cycles: int
    k_fraction: Optional[float] = None
    alpha: Optional[float] = None
    threshold_combined: Optional[float] = None
    total_shots: int = 0
    accepted_shots: int = 0
    # Per-shot scores (for fusion variant)
    scores: list[float] = field(default_factory=list)
    # Batch abort info
    probe_pass_rate: Optional[float] = None
    batch_aborted: bool = False

    @property
    def acceptance_probability(self) -> float:
        if self.total_shots == 0:
            return 0.0
        return self.accepted_shots / self.total_shots

    @property
    def tts(self) -> float:
        """Time-to-solution: 1/acceptance_probability (or inf)."""
        p = self.acceptance_probability
        return 1.0 / p if p > 0 else float("inf")

    def as_dict(self) -> dict:
        return {
            "variant": self.variant,
            "N": self.n_subsystems,
            "D": self.depth,
            "W": self.n_cycles,
            "k_fraction": self.k_fraction,
            "alpha": self.alpha,
            "threshold_combined": self.threshold_combined,
            "total_shots": self.total_shots,
            "accepted_shots": self.accepted_shots,
            "acceptance_probability": self.acceptance_probability,
            "TTS": self.tts,
            "probe_pass_rate": self.probe_pass_rate,
            "batch_aborted": self.batch_aborted,
            "mean_score": float(np.mean(self.scores)) if self.scores else None,
        }


def apply_conditioning(
    counts: dict[str, int],
    n_subsystems: int,
    n_cycles: int,
    depth: int,
    variant: str = "global",
    k_fraction: float = 0.8,
    alpha: float = 0.5,
    threshold_combined: float = 0.65,
) -> ConditioningResult:
    """Apply a conditioning rule to sampler output counts.

    Args:
        counts:             {bitstring: count} from Sampler.
        n_subsystems:       N.
        n_cycles:           W.
        depth:              D (for record-keeping).
        variant:            "global" | "hierarchical" | "score_fusion".
        k_fraction:         fraction for hierarchical rule.
        alpha:              weight for score fusion.
        threshold_combined: threshold for score fusion.

    Returns:
        ConditioningResult with acceptance statistics.
    """
    result = ConditioningResult(
        variant=variant,
        n_subsystems=n_subsystems,
        depth=depth,
        n_cycles=n_cycles,
        k_fraction=k_fraction if variant == "hierarchical" else None,
        alpha=alpha if variant == "score_fusion" else None,
        threshold_combined=(threshold_combined
                            if variant == "score_fusion" else None),
    )

    for bitstring, count in counts.items():
        outcome = ShotOutcome.from_bitstring(bitstring, n_subsystems, n_cycles)
        result.total_shots += count

        if variant == "global":
            if decide_global(outcome):
                result.accepted_shots += count

        elif variant == "hierarchical":
            if decide_hierarchical(outcome, k_fraction):
                result.accepted_shots += count

        elif variant == "score_fusion":
            accepted, score = decide_score_fusion(outcome, alpha,
                                                  threshold_combined)
            result.scores.extend([score] * count)
            if accepted:
                result.accepted_shots += count
        else:
            raise ValueError(f"Unknown variant: {variant}")

    return result
