#!/usr/bin/env python3
"""
Example: Qiskit adapter — local Aer simulation.

Requires:  pip install qgate[qiskit]
"""
try:
    from qgate import GateConfig, TrajectoryFilter
    from qgate.adapters.qiskit_adapter import QiskitAdapter
except ImportError:
    raise SystemExit(
        "Qiskit not installed.  Run:  pip install qgate[qiskit]"
    )

config = GateConfig(
    n_subsystems=3,
    n_cycles=2,
    shots=500,
    variant="score_fusion",
    fusion={"alpha": 0.5, "threshold": 0.65},
    adapter="qiskit",
)

adapter = QiskitAdapter(scramble_depth=1, optimization_level=1)
tf = TrajectoryFilter(config, adapter)
result = tf.run()

print(f"Variant       : {result.variant}")
print(f"Accepted      : {result.accepted_shots}/{result.total_shots}")
print(f"P_accept      : {result.acceptance_probability:.4f}")
print(f"Mean score    : {result.mean_combined_score:.4f}")
