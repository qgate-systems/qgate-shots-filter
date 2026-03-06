#!/usr/bin/env python3
"""
validate_os_e2e.py — End-to-End physics validation of QgateSampler OS.

Proves that the QgateSampler transparent SamplerV2 wrapper achieves
measurable MSE reduction compared to a standard (unfiltered) SamplerV2
on a physically meaningful problem: the 1D Transverse-Field Ising Model.

Protocol:
  1. Build a TFIM VQE-style circuit (8 qubits, 3 ansatz layers).
  2. Set up a noisy AerSimulator with Heron-class noise model.
  3. Baseline run:  standard SamplerV2 → raw energy estimate.
  4. Qgate OS run:  QgateSampler (same circuit, same backend) → filtered energy.
  5. Repeat for N_TRIALS independent seeds → per-trial MSE.
  6. Report comparative statistics and assert MSE_qgate < MSE_baseline.

**NOTICE — PRE-PATENT PROPRIETARY CODE**
Do NOT distribute, publish, or push to any public repository.

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915

Usage:
    python simulations/qgate_os/validate_os_e2e.py
    python simulations/qgate_os/validate_os_e2e.py --trials 5 --shots 50000
    python simulations/qgate_os/validate_os_e2e.py --dry-run
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Configuration constants
# ═══════════════════════════════════════════════════════════════════════════

N_QUBITS = 8
N_LAYERS = 3          # ansatz depth
J_COUPLING = 1.0
H_FIELD = 1.0
SHOTS = 100_000
N_TRIALS = 10         # independent random seeds for statistical power
SEED_BASE = 42


# ═══════════════════════════════════════════════════════════════════════════
# Noise model — IBM Heron-class
# ═══════════════════════════════════════════════════════════════════════════

def build_heron_noisy_backend() -> Any:
    """AerSimulator with IBM Heron-class realistic noise.

    Parameters match published ibm_torino calibration data:
      T1 = 300 µs, T2 = 150 µs
      1Q depolarisation = 1e-3, 2Q depolarisation = 1e-2
      Gate times: 1Q = 60 ns, 2Q = 660 ns, measurement = 1600 ns
    """
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        thermal_relaxation_error,
    )

    t1, t2 = 300e-6, 150e-6
    g1q, g2q, g_meas = 60e-9, 660e-9, 1600e-9
    depol_1q, depol_2q = 1e-3, 1e-2

    model = NoiseModel()

    # --- 1-qubit thermal + depolarising ---
    err_1q = thermal_relaxation_error(t1, t2, g1q)
    composite_1q = err_1q.compose(depolarizing_error(depol_1q, 1))
    model.add_all_qubit_quantum_error(
        composite_1q, ["u1", "u2", "u3", "rx", "ry", "rz", "x", "h"],
    )

    # --- 2-qubit thermal + depolarising ---
    err_2q = thermal_relaxation_error(t1, t2, g2q).expand(
        thermal_relaxation_error(t1, t2, g2q),
    )
    composite_2q = err_2q.compose(depolarizing_error(depol_2q, 2))
    model.add_all_qubit_quantum_error(composite_2q, ["cx"])

    # --- Measurement error ---
    err_meas = thermal_relaxation_error(t1, t2, g_meas)
    model.add_all_qubit_quantum_error(err_meas, ["measure"])

    return AerSimulator(noise_model=model, method="statevector")


# ═══════════════════════════════════════════════════════════════════════════
# TFIM circuit builder
# ═══════════════════════════════════════════════════════════════════════════

def build_tfim_vqe_circuit(
    n_qubits: int,
    n_layers: int,
    seed: int = 42,
) -> Any:
    """Build a hardware-efficient VQE ansatz circuit for the 1D TFIM.

    Architecture:
      |+⟩^n → [Ry(θ) Rz(φ) per qubit + CNOT ladder] × n_layers → measure all

    The parameters are seeded randomly — this simulates a snapshot of a
    VQE optimisation loop at some intermediate parameter vector.
    """
    from qiskit import QuantumCircuit

    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits, name=f"TFIM_{n_qubits}q_d{n_layers}")

    # Initial state: |+⟩^n (good starting point for TFIM with h > 0)
    for q in range(n_qubits):
        qc.h(q)

    # Hardware-efficient ansatz layers
    for layer in range(n_layers):
        # Parameterised rotations
        for q in range(n_qubits):
            qc.ry(float(rng.uniform(-math.pi, math.pi)), q)
            qc.rz(float(rng.uniform(-math.pi, math.pi)), q)
        # CNOT entangling ladder (linear nearest-neighbour)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        qc.barrier()

    # Measure all system qubits
    qc.measure_all()
    return qc


# ═══════════════════════════════════════════════════════════════════════════
# Exact ground state energy (reuse from qgate)
# ═══════════════════════════════════════════════════════════════════════════

def compute_exact_ground_energy(
    n_qubits: int,
    j_coupling: float = 1.0,
    h_field: float = 1.0,
) -> float:
    """Exact TFIM ground-state energy via sparse diagonalisation."""
    from qgate.adapters.vqe_adapter import tfim_exact_ground_energy
    return tfim_exact_ground_energy(n_qubits, j_coupling, h_field)


# ═══════════════════════════════════════════════════════════════════════════
# Energy extraction from SamplerV2 BitArray results
# ═══════════════════════════════════════════════════════════════════════════

def bitstring_zz_energy(
    bitstring: str,
    n_qubits: int,
    j_coupling: float = 1.0,
) -> float:
    """Diagonal ZZ energy for a computational-basis state.

    E_ZZ = −J Σ_{i=0}^{n-2} s_i · s_{i+1}  where s = +1 (bit=0), −1 (bit=1).
    """
    bits = [int(b) for b in bitstring[-n_qubits:]]
    spins = [1 - 2 * b for b in bits]
    return -j_coupling * sum(
        spins[i] * spins[i + 1] for i in range(len(spins) - 1)
    )


def energy_from_bitarray(
    bitarray: Any,
    n_qubits: int,
    j_coupling: float = 1.0,
) -> tuple[float, float, int]:
    """Extract mean ZZ energy, per-shot variance, and shot count from a BitArray.

    Returns:
        (mean_energy, variance, n_shots)
    """
    counts = bitarray.get_counts()
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0, 0.0, 0

    energies = []
    for bs, cnt in counts.items():
        e = bitstring_zz_energy(bs, n_qubits, j_coupling)
        energies.extend([e] * cnt)

    arr = np.array(energies)
    return float(arr.mean()), float(arr.var()), total_shots


# ═══════════════════════════════════════════════════════════════════════════
# Trial data container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrialResult:
    """Results from a single trial (one circuit seed)."""
    seed: int
    raw_energy: float
    raw_variance: float
    raw_shots: int
    qgate_energy: float
    qgate_variance: float
    qgate_shots: int
    qgate_acceptance_rate: float
    exact_energy: float

    @property
    def raw_mse(self) -> float:
        return (self.raw_energy - self.exact_energy) ** 2

    @property
    def qgate_mse(self) -> float:
        return (self.qgate_energy - self.exact_energy) ** 2

    @property
    def mse_reduction_pct(self) -> float:
        if self.raw_mse < 1e-30:
            return 0.0
        return 100.0 * (1.0 - self.qgate_mse / self.raw_mse)

    @property
    def variance_ratio(self) -> float:
        if self.qgate_variance < 1e-30:
            return float("inf")
        return self.raw_variance / self.qgate_variance


# ═══════════════════════════════════════════════════════════════════════════
# Core experiment: single trial
# ═══════════════════════════════════════════════════════════════════════════

def run_single_trial(
    backend: Any,
    circuit_seed: int,
    shots: int,
    exact_energy: float,
    dry_run: bool = False,
) -> TrialResult:
    """Run one trial: baseline SamplerV2 vs QgateSampler, same circuit.

    Both samplers use the SAME backend and the SAME circuit. The only
    difference is that QgateSampler transparently injects probes and
    applies Galton filtering before returning the result.
    """
    from qiskit_ibm_runtime import SamplerV2
    from qgate.sampler import QgateSampler, SamplerConfig

    # --- Build the circuit ---
    qc = build_tfim_vqe_circuit(N_QUBITS, N_LAYERS, seed=circuit_seed)

    if dry_run:
        shots = min(shots, 1024)

    # ── 1) Baseline run: standard SamplerV2 ──────────────────────────
    baseline_sampler = SamplerV2(mode=backend)
    baseline_job = baseline_sampler.run([(qc,)], shots=shots)
    baseline_result = baseline_job.result()

    # Extract system measurement register
    baseline_pub = baseline_result[0]
    baseline_ba = baseline_pub.data.meas
    raw_energy, raw_var, raw_n = energy_from_bitarray(
        baseline_ba, N_QUBITS, J_COUPLING,
    )

    # ── 2) Qgate OS run: QgateSampler ────────────────────────────────
    qgate_cfg = SamplerConfig(
        target_acceptance=0.20,
        probe_angle=math.pi / 6,
        window_size=4096,
        min_window_size=100,
        baseline_threshold=0.65,
        optimization_level=1,
    )
    qgate_sampler = QgateSampler(backend=backend, config=qgate_cfg)
    qgate_job = qgate_sampler.run([(qc,)], shots=shots)
    qgate_result = qgate_job.result()

    # Extract the filtered system measurement register
    qgate_pub = qgate_result[0]
    qgate_ba = qgate_pub.data.meas
    qgate_energy, qgate_var, qgate_n = energy_from_bitarray(
        qgate_ba, N_QUBITS, J_COUPLING,
    )

    # Get acceptance metadata
    filter_meta = qgate_pub.metadata.get("qgate_filter", {})
    acceptance_rate = filter_meta.get("acceptance_rate", 0.0)

    return TrialResult(
        seed=circuit_seed,
        raw_energy=raw_energy,
        raw_variance=raw_var,
        raw_shots=raw_n,
        qgate_energy=qgate_energy,
        qgate_variance=qgate_var,
        qgate_shots=qgate_n,
        qgate_acceptance_rate=acceptance_rate,
        exact_energy=exact_energy,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Multi-trial experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(
    n_trials: int = N_TRIALS,
    shots: int = SHOTS,
    dry_run: bool = False,
) -> list[TrialResult]:
    """Run the full multi-trial E2E validation experiment."""

    # --- Setup ---
    print("=" * 72)
    print("  QgateSampler OS — End-to-End Physics Validation")
    print("=" * 72)

    print(f"\n  TFIM:     {N_QUBITS} qubits, J={J_COUPLING}, h={H_FIELD}")
    print(f"  Ansatz:   {N_LAYERS} HW-efficient layers")
    print(f"  Shots:    {shots:,} per trial {'(dry-run)' if dry_run else ''}")
    print(f"  Trials:   {n_trials}")
    print(f"  Noise:    IBM Heron (T1=300µs, T2=150µs, dep_1q=1e-3, dep_2q=1e-2)")

    exact_e = compute_exact_ground_energy(N_QUBITS, J_COUPLING, H_FIELD)
    print(f"  Exact GS: {exact_e:.6f}")
    print()

    backend = build_heron_noisy_backend()
    print("  Backend ready: AerSimulator (Heron noise)")

    # --- Run trials ---
    results: list[TrialResult] = []
    t0 = time.perf_counter()

    for trial in range(n_trials):
        seed = SEED_BASE + trial * 137  # well-separated seeds
        t_trial = time.perf_counter()

        result = run_single_trial(
            backend=backend,
            circuit_seed=seed,
            shots=shots,
            exact_energy=exact_e,
            dry_run=dry_run,
        )
        results.append(result)

        dt = time.perf_counter() - t_trial
        sign = "+" if result.mse_reduction_pct > 0 else ""
        print(
            f"  Trial {trial + 1:2d}/{n_trials} "
            f"(seed={seed:5d}) | "
            f"Raw E={result.raw_energy:+.4f}  "
            f"QG E={result.qgate_energy:+.4f}  "
            f"MSE↓ {sign}{result.mse_reduction_pct:.1f}%  "
            f"Accept={result.qgate_acceptance_rate:.1%}  "
            f"[{dt:.1f}s]"
        )

    total_time = time.perf_counter() - t0
    print(f"\n  Total time: {total_time:.1f}s")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Report + assertion
# ═══════════════════════════════════════════════════════════════════════════

def print_report(results: list[TrialResult]) -> bool:
    """Print comparative statistics and return True if validation passes."""
    from scipy import stats

    raw_mses = np.array([r.raw_mse for r in results])
    qgate_mses = np.array([r.qgate_mse for r in results])
    raw_energies = np.array([r.raw_energy for r in results])
    qgate_energies = np.array([r.qgate_energy for r in results])
    acceptance_rates = np.array([r.qgate_acceptance_rate for r in results])
    variance_ratios = np.array([r.variance_ratio for r in results])

    exact_e = results[0].exact_energy
    mean_raw_mse = float(raw_mses.mean())
    mean_qgate_mse = float(qgate_mses.mean())
    overall_reduction = 100.0 * (1.0 - mean_qgate_mse / mean_raw_mse) if mean_raw_mse > 1e-30 else 0.0

    # Paired t-test: MSE_raw vs MSE_qgate (one-sided: raw > qgate)
    t_stat, p_two = stats.ttest_rel(raw_mses, qgate_mses)
    p_one = p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0

    # Wilcoxon signed-rank (non-parametric backup)
    try:
        w_stat, w_p_two = stats.wilcoxon(raw_mses, qgate_mses, alternative="greater")
        w_p = w_p_two
    except ValueError:
        w_p = 1.0  # all differences are zero

    # --- Console report ---
    print("\n" + "=" * 72)
    print("  RESULTS — QgateSampler OS E2E Validation")
    print("=" * 72)

    print(f"\n  Exact ground-state energy:  {exact_e:.6f}")
    print(f"  Number of trials:           {len(results)}")
    print()

    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │                  Baseline (raw)        QgateSampler (OS)   │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │ Mean energy     {np.mean(raw_energies):+12.6f}         {np.mean(qgate_energies):+12.6f}        │")
    print(f"  │ Std energy      {np.std(raw_energies):12.6f}         {np.std(qgate_energies):12.6f}        │")
    print(f"  │ Mean MSE        {mean_raw_mse:12.6f}         {mean_qgate_mse:12.6f}        │")
    print(f"  │ Std MSE         {np.std(raw_mses):12.6f}         {np.std(qgate_mses):12.6f}        │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    print(f"  MSE reduction:        {overall_reduction:+.2f}%")
    print(f"  Mean acceptance rate: {np.mean(acceptance_rates):.2%}")
    print(f"  Mean variance ratio:  {np.mean(variance_ratios):.1f}× (raw/qgate)")
    print()
    print(f"  Paired t-test:        t={t_stat:.3f}, p={p_one:.4e} (one-sided)")
    print(f"  Wilcoxon signed-rank: p={w_p:.4e} (one-sided)")
    print()

    # --- Pass/fail ---
    passed = mean_qgate_mse < mean_raw_mse
    if passed:
        print("  ✅ VALIDATION PASSED — QgateSampler MSE < Baseline MSE")
    else:
        print("  ❌ VALIDATION FAILED — QgateSampler MSE >= Baseline MSE")

    # Report per-trial detail
    print()
    print("  Per-trial breakdown:")
    print("  ─────────────────────────────────────────────────────────────────")
    print("  Trial  Seed   Raw Energy   QG Energy    Raw MSE     QG MSE   MSE↓%")
    for i, r in enumerate(results):
        print(
            f"  {i + 1:5d}  {r.seed:5d}  {r.raw_energy:+10.4f}  {r.qgate_energy:+10.4f}"
            f"  {r.raw_mse:10.4f}  {r.qgate_mse:10.4f}  {r.mse_reduction_pct:+6.1f}%"
        )

    print("=" * 72)
    return passed


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QgateSampler OS — End-to-End physics validation",
    )
    parser.add_argument(
        "--trials", type=int, default=N_TRIALS,
        help=f"Number of independent trials (default: {N_TRIALS})",
    )
    parser.add_argument(
        "--shots", type=int, default=SHOTS,
        help=f"Shots per trial (default: {SHOTS:,})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Quick test with 1024 shots and 3 trials",
    )
    args = parser.parse_args()

    n_trials = 3 if args.dry_run else args.trials
    shots = 1024 if args.dry_run else args.shots

    results = run_experiment(n_trials=n_trials, shots=shots, dry_run=args.dry_run)
    passed = print_report(results)

    if not passed:
        print("\n  ASSERT FAILED: exiting with code 1")
        sys.exit(1)
    else:
        print("\n  All assertions passed. QgateSampler OS validated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
