#!/usr/bin/env python3
"""
Example: Basic qgate TrajectoryFilter with MockAdapter.

Demonstrates the new primary API:
  1. Create a GateConfig
  2. Instantiate a MockAdapter
  3. Run the TrajectoryFilter
  4. Inspect FilterResult
"""
from qgate import GateConfig, TrajectoryFilter
from qgate.adapters import MockAdapter

# ── 1. Configure ──────────────────────────────────────────────────────────
config = GateConfig(
    n_subsystems=4,
    n_cycles=2,
    shots=1000,
    variant="score_fusion",
    fusion={"alpha": 0.5, "threshold": 0.65},
)

# ── 2. Adapter (synthetic noise for demo) ────────────────────────────────
adapter = MockAdapter(error_rate=0.08, seed=42)

# ── 3. Run ────────────────────────────────────────────────────────────────
tf = TrajectoryFilter(config, adapter)
result = tf.run()

# ── 4. Results ────────────────────────────────────────────────────────────
print(f"Variant           : {result.variant}")
print(f"Total shots       : {result.total_shots}")
print(f"Accepted shots    : {result.accepted_shots}")
print(f"Acceptance prob.  : {result.acceptance_probability:.4f}")
print(f"TTS               : {result.tts:.2f}")
print(f"Mean combined score: {result.mean_combined_score:.4f}")
print(f"Threshold used    : {result.threshold_used:.4f}")
