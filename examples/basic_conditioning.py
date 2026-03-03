#!/usr/bin/env python3
"""
basic_conditioning.py — Minimal example demonstrating the three conditioning
strategies provided by qgate.

Install first:
    pip install qgate

Then run:
    python examples/basic_conditioning.py
"""
from qgate.conditioning import (
    ParityOutcome,
    decide_global,
    decide_hierarchical,
    decide_score_fusion,
    apply_rule_to_batch,
)


def main() -> None:
    # ------------------------------------------------------------------
    # 1.  Build synthetic parity outcomes
    # ------------------------------------------------------------------
    # 4 Bell-pair subsystems, 3 monitoring cycles per shot
    shots = [
        # Shot 0: all pass
        ParityOutcome(4, 3, [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        # Shot 1: one failure in cycle 1
        ParityOutcome(4, 3, [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]),
        # Shot 2: two failures in cycle 0
        ParityOutcome(4, 3, [[1, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        # Shot 3: heavy noise
        ParityOutcome(4, 3, [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0]]),
    ]

    # ------------------------------------------------------------------
    # 2.  Per-shot decisions
    # ------------------------------------------------------------------
    print("=== Per-shot decisions ===\n")
    for i, shot in enumerate(shots):
        g = decide_global(shot)
        h = decide_hierarchical(shot, k_fraction=0.75)
        sf_ok, sf_score = decide_score_fusion(shot, alpha=0.5, threshold_combined=0.65)
        print(
            f"  Shot {i}: global={g!s:5s}  hier(k=0.75)={h!s:5s}  "
            f"score_fusion={sf_ok!s:5s} (score={sf_score:.3f})"
        )

    # ------------------------------------------------------------------
    # 3.  Batch statistics
    # ------------------------------------------------------------------
    print("\n=== Batch statistics ===\n")
    for variant in ("global", "hierarchical", "score_fusion"):
        stats = apply_rule_to_batch(
            shots,
            variant=variant,
            k_fraction=0.75,
            alpha=0.5,
            threshold_combined=0.65,
        )
        d = stats.as_dict()
        print(
            f"  {variant:18s}  accepted={d['accepted_shots']}/{d['total_shots']}  "
            f"P_acc={d['acceptance_probability']:.2f}  TTS={d['TTS']:.2f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
