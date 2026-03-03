#!/usr/bin/env python3
"""
multi_rate_monitor.py — Example of multi-rate monitoring and score fusion.

Simulates a streaming scenario where a monitor accumulates HF / LF
parity scores across cycles and makes a fused accept / reject decision.

Install first:
    pip install qgate

Then run:
    python examples/multi_rate_monitor.py
"""
import numpy as np

from qgate.monitors import (
    compute_window_metric,
    score_fusion,
    MultiRateMonitor,
    should_abort_batch,
)


def example_window_metric() -> None:
    """Demonstrate trailing-window fidelity metric on a synthetic trajectory."""
    print("=== Window Metric (synthetic fidelity trajectory) ===\n")

    t = np.linspace(0, 12, 500)
    fidelity = np.exp(-0.05 * t) * (0.5 + 0.5 * np.cos(t))

    for win in (0.5, 1.0, 2.0):
        metric, ws, we = compute_window_metric(t, fidelity, window=win, mode="max")
        print(f"  window={win:.1f}  →  max fidelity in [{ws:.1f}, {we:.1f}] = {metric:.4f}")

    print()


def example_score_fusion() -> None:
    """One-shot score fusion at several α values."""
    print("=== Score Fusion (one-shot) ===\n")
    lf, hf = 0.85, 0.60
    for alpha in (0.0, 0.3, 0.5, 0.7, 1.0):
        accepted, score = score_fusion(lf, hf, alpha=alpha, threshold=0.65)
        tag = "✓" if accepted else "✗"
        print(f"  α={alpha:.1f}  →  combined={score:.3f}  {tag}")
    print()


def example_multi_rate_monitor() -> None:
    """Streaming multi-rate monitor across 6 cycles."""
    print("=== MultiRateMonitor (streaming) ===\n")

    rng = np.random.default_rng(42)
    n_cycles = 6
    n_subsystems = 8

    mon = MultiRateMonitor(n_subsystems=n_subsystems, alpha=0.5, threshold_combined=0.65)

    for c in range(n_cycles):
        # Simulate: each subsystem passes with 70% probability
        passes = rng.random(n_subsystems) < 0.70
        rate = float(passes.mean())
        mon.record_cycle(c, rate)
        kind = "HF+LF" if c % 2 == 0 else "HF   "
        print(f"  cycle {c} ({kind}): pass_rate = {rate:.3f}")

    accepted, score = mon.fused_decision()
    tag = "ACCEPTED" if accepted else "REJECTED"
    print(f"\n  Fused score = {score:.4f}  →  {tag}\n")


def example_batch_abort() -> None:
    """Demonstrate probe-based batch abort."""
    print("=== Batch Abort (probe check) ===\n")
    for probe_rate in (0.80, 0.65, 0.40, 0.10):
        abort = should_abort_batch(probe_rate, theta=0.65)
        action = "ABORT — skip full batch" if abort else "PROCEED"
        print(f"  probe_pass_rate={probe_rate:.2f}  →  {action}")
    print()


def main() -> None:
    example_window_metric()
    example_score_fusion()
    example_multi_rate_monitor()
    example_batch_abort()
    print("All examples complete.")


if __name__ == "__main__":
    main()
