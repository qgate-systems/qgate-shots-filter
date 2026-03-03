#!/usr/bin/env python3
"""
qgate v0.4.0 — End-to-end simulation demo.

Runs all three conditioning strategies across a range of error rates
and subsystem counts, logs results to JSONL, and prints a summary table.
"""
import logging
import tempfile
from pathlib import Path

import qgate
from qgate import (
    DynamicThresholdConfig,
    FusionConfig,
    GateConfig,
    TrajectoryFilter,
)
from qgate.adapters.base import MockAdapter
from qgate.run_logging import RunLogger

# ── Setup ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)  # keep output clean

print(f"qgate version {qgate.__version__}")
print("=" * 80)

# ── 1.  Basic run: one config, one adapter ────────────────────────────────
print("\n▶ 1. Basic run (score_fusion, N=4, W=2, 1024 shots)")
config = GateConfig(
    n_subsystems=4,
    n_cycles=2,
    shots=1024,
    variant="score_fusion",
    fusion=FusionConfig(alpha=0.5, threshold=0.65),
)
adapter = MockAdapter(error_rate=0.05, seed=42)
tf = TrajectoryFilter(config, adapter)
print(f"   repr: {tf!r}")
result = tf.run()
print(f"   Accepted  : {result.accepted_shots}/{result.total_shots}")
print(f"   P_accept  : {result.acceptance_probability:.4f}")
print(f"   TTS       : {result.tts:.2f}")
print(f"   Mean score: {result.mean_combined_score:.4f}")
print(f"   Run ID    : {result.run_id}")

# ── 2.  Compare all three strategies ──────────────────────────────────────
print("\n▶ 2. Strategy comparison (error_rate=0.10, N=8, W=3, 2000 shots)")
print(f"   {'Strategy':<18} {'Accepted':>8} {'P_acc':>8} {'TTS':>8} {'Score':>8}")
print("   " + "-" * 52)

for variant in ("global", "hierarchical", "score_fusion"):
    cfg = GateConfig(
        n_subsystems=8,
        n_cycles=3,
        shots=2000,
        variant=variant,
        k_fraction=0.75,
        fusion=FusionConfig(alpha=0.5, threshold=0.65),
    )
    tf = TrajectoryFilter(cfg, MockAdapter(error_rate=0.10, seed=7))
    r = tf.run()
    score_str = f"{r.mean_combined_score:.4f}" if r.mean_combined_score else "N/A"
    print(f"   {variant:<18} {r.accepted_shots:>8} {r.acceptance_probability:>8.4f} {r.tts:>8.2f} {score_str:>8}")

# ── 3.  Sweep error rates ────────────────────────────────────────────────
print("\n▶ 3. Error-rate sweep (score_fusion, N=6, W=2, 500 shots)")
print(f"   {'Error Rate':>12} {'P_acc':>8} {'TTS':>10}")
print("   " + "-" * 32)

for er in [0.0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.0]:
    cfg = GateConfig(n_subsystems=6, n_cycles=2, shots=500, variant="score_fusion")
    tf = TrajectoryFilter(cfg, MockAdapter(error_rate=er, seed=99))
    r = tf.run()
    print(f"   {er:>12.2f} {r.acceptance_probability:>8.4f} {r.tts:>10.2f}")

# ── 4.  Dynamic thresholding over 10 batches ─────────────────────────────
print("\n▶ 4. Dynamic thresholding (10 batches, z=1.0)")
cfg = GateConfig(
    n_subsystems=4,
    n_cycles=2,
    shots=200,
    variant="score_fusion",
    dynamic_threshold=DynamicThresholdConfig(enabled=True, baseline=0.65, z_factor=1.0),
)
tf = TrajectoryFilter(cfg, MockAdapter(error_rate=0.08, seed=123))
for i in range(10):
    r = tf.run()
    print(f"   batch {i+1:>2}: θ={tf.current_threshold:.4f}  "
          f"P_acc={r.acceptance_probability:.4f}  accepted={r.accepted_shots}")

# ── 5.  Subsystem scaling ────────────────────────────────────────────────
print("\n▶ 5. Subsystem scaling (global vs hierarchical vs score_fusion, error=0.05)")
print(f"   {'N':>4}  {'Global':>10}  {'Hier k=0.75':>12}  {'Score Fusion':>13}")
print("   " + "-" * 44)

for n_sub in [1, 2, 4, 8, 16, 32]:
    row = f"   {n_sub:>4}"
    for variant in ("global", "hierarchical", "score_fusion"):
        cfg = GateConfig(
            n_subsystems=n_sub, n_cycles=2, shots=500,
            variant=variant, k_fraction=0.75,
        )
        tf = TrajectoryFilter(cfg, MockAdapter(error_rate=0.05, seed=42))
        r = tf.run()
        row += f"  {r.acceptance_probability:>12.4f}"
    print(row)

# ── 6.  Run logging (JSONL) ──────────────────────────────────────────────
print("\n▶ 6. Run logging to JSONL")
log_path = Path(tempfile.mkdtemp()) / "demo_results.jsonl"

cfg = GateConfig(n_subsystems=4, n_cycles=2, shots=500, variant="score_fusion")
with RunLogger(log_path) as rl:
    for seed in range(5):
        tf = TrajectoryFilter(cfg, MockAdapter(error_rate=0.05, seed=seed), logger=rl)
        r = tf.run()

print(f"   Wrote {sum(1 for _ in open(log_path))} records to {log_path}")

import json
with open(log_path) as f:
    first = json.loads(f.readline())
print(f"   Sample keys: {list(first.keys())}")
print(f"   Sample run_id: {first['run_id']}, P_acc: {first['acceptance_probability']:.4f}")

# ── 7.  Low-level API: ParityOutcome + decision rules ────────────────────
print("\n▶ 7. Low-level API: ParityOutcome + decision rules")
from qgate import ParityOutcome, decide_global, decide_hierarchical, decide_score_fusion

outcome = ParityOutcome(4, 2, [[0, 0, 1, 0], [0, 0, 0, 0]])
print(f"   Matrix:\n{outcome.parity_matrix}")
print(f"   Pass rates: {outcome.pass_rates}")
print(f"   Global:       {decide_global(outcome)}")
print(f"   Hierarchical: {decide_hierarchical(outcome, k_fraction=0.75)}")
accepted, combined = decide_score_fusion(outcome, threshold_combined=0.65)
print(f"   Score fusion: accepted={accepted}, combined={combined:.3f}")

# ── 8.  filter_counts (Qiskit-style count dict) ──────────────────────────
print("\n▶ 8. filter_counts (synthetic count dict)")
cfg = GateConfig(n_subsystems=3, n_cycles=1, shots=20, variant="score_fusion")
mock = MockAdapter(error_rate=0.1, seed=55)
tf = TrajectoryFilter(cfg, mock)

# Build + run to get raw results, then filter_counts
circuit = mock.build_circuit(3, 1)
raw = mock.run(circuit, 20)
r = tf.filter_counts(raw, n_subsystems=3, n_cycles=1)
print(f"   Accepted: {r.accepted_shots}/{r.total_shots}  P_acc={r.acceptance_probability:.4f}")

# ── 9.  Adapter discovery ────────────────────────────────────────────────
print("\n▶ 9. Adapter discovery")
from qgate import list_adapters
adapters = list_adapters()
for name, target in sorted(adapters.items()):
    print(f"   {name:<15} → {target}")

# ── Done ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("✅  All simulations completed successfully.")
