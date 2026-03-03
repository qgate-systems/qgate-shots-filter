#!/usr/bin/env python3
"""
Example: Dynamic threshold adaptation.

Shows how the DynamicThreshold adjusts over multiple batches based
on rolling z-score gating.
"""
from qgate import DynamicThresholdConfig, GateConfig, TrajectoryFilter
from qgate.adapters import MockAdapter

config = GateConfig(
    n_subsystems=4,
    n_cycles=2,
    shots=200,
    variant="score_fusion",
    dynamic_threshold=DynamicThresholdConfig(
        enabled=True,
        baseline=0.65,
        z_factor=1.0,
        window_size=5,
        min_threshold=0.4,
        max_threshold=0.9,
    ),
)

adapter = MockAdapter(error_rate=0.10, seed=42)
tf = TrajectoryFilter(config, adapter)

print("Batch | Threshold | P_accept | Mean Score")
print("------+-----------+----------+-----------")

for batch_idx in range(8):
    result = tf.run()
    print(
        f"  {batch_idx:3d}  | "
        f"  {tf.current_threshold:.4f}  | "
        f"  {result.acceptance_probability:.4f}  | "
        f"  {result.mean_combined_score:.4f}"
    )

print(f"\nFinal dynamic threshold: {tf.current_threshold:.4f}")
