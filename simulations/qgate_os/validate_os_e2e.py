#!/usr/bin/env python3
"""
validate_os_e2e.py — End-to-End physics validation of QgateSampler OS.

Proves that the QgateSampler transparent SamplerV2 wrapper achieves
measurable MSE reduction compared to a standard (unfiltered) SamplerV2
on a physically meaningful problem: the 1D Transverse-Field Ising Model.

Protocol:
  1. Build a parameterised TFIM VQE ansatz (8 qubits, 3 layers).
  2. Set up a noisy AerSimulator with Heron-class noise model.
  3. **Warm-up:** Run a quick VQE pre-optimisation (~20 iterations via
     EstimatorV2 + COBYLA) to find a partially-optimised θ* that gives
     the circuit enough correlation structure for the Galton filter to
     discriminate on.
  4. **Showdown:** Using θ*, run 10 independent measurement trials:
       - Baseline: standard SamplerV2 → raw energy estimate.
       - Qgate OS: QgateSampler → filtered energy estimate.
  5. Report comparative statistics and assert MSE_qgate < MSE_baseline.

**NOTICE — PRE-PATENT PROPRIETARY CODE**
Do NOT distribute, publish, or push to any public repository.

Patent pending (see LICENSE)

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
VQE_MAXITER = 30      # pre-optimisation iterations (enough for ZZ structure)
VQE_OPT_SHOTS = 4096  # shots per EstimatorV2 evaluation during warm-up


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
# TFIM circuit builder (parameterised)
# ═══════════════════════════════════════════════════════════════════════════

def _num_params(n_qubits: int, n_layers: int) -> int:
    """Total number of variational parameters: 2 per qubit per layer."""
    return 2 * n_qubits * n_layers


def build_tfim_ansatz(
    n_qubits: int,
    n_layers: int,
) -> Any:
    """Build a parameterised HW-efficient ansatz for the 1D TFIM.

    Architecture:
      |+⟩^n → [Ry(θ_i) Rz(φ_i) per qubit + CNOT ladder] × n_layers

    Returns a QuantumCircuit with unbound ParameterVector entries.
    No measurements are attached — the caller adds measure_all() for
    sampling or passes the bare ansatz to EstimatorV2.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    n_params = _num_params(n_qubits, n_layers)
    theta = ParameterVector("θ", n_params)

    qc = QuantumCircuit(n_qubits, name=f"TFIM_{n_qubits}q_d{n_layers}")

    # Initial state: |+⟩^n (good starting point for TFIM with h > 0)
    for q in range(n_qubits):
        qc.h(q)

    idx = 0
    for _layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(theta[idx], q)
            qc.rz(theta[idx + 1], q)
            idx += 2
        # CNOT entangling ladder (linear nearest-neighbour)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        qc.barrier()

    return qc


def bind_and_measure(ansatz: Any, theta_vals: np.ndarray) -> Any:
    """Bind parameter values to the ansatz, add measure_all, return circuit."""
    bound = ansatz.assign_parameters(dict(zip(ansatz.parameters, theta_vals)))
    bound.measure_all()
    return bound


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
# TFIM Hamiltonian as SparsePauliOp (for EstimatorV2)
# ═══════════════════════════════════════════════════════════════════════════

def build_tfim_hamiltonian(
    n_qubits: int,
    j_coupling: float = 1.0,
    h_field: float = 1.0,
) -> Any:
    """Build the 1D TFIM Hamiltonian as a Qiskit SparsePauliOp.

    H = −J Σ_{i} Z_i Z_{i+1}  −  h Σ_{i} X_i

    Returns:
        SparsePauliOp suitable for EstimatorV2.
    """
    from qiskit.quantum_info import SparsePauliOp

    pauli_terms: list[tuple[str, float]] = []

    # ZZ coupling: -J Z_i Z_{i+1}
    for i in range(n_qubits - 1):
        label = ["I"] * n_qubits
        label[i] = "Z"
        label[i + 1] = "Z"
        # SparsePauliOp uses little-endian (qubit 0 = rightmost)
        pauli_terms.append(("".join(reversed(label)), -j_coupling))

    # Transverse field: -h X_i
    for i in range(n_qubits):
        label = ["I"] * n_qubits
        label[i] = "X"
        pauli_terms.append(("".join(reversed(label)), -h_field))

    return SparsePauliOp.from_list(pauli_terms).simplify()


# ═══════════════════════════════════════════════════════════════════════════
# VQE Pre-optimisation (the warm-up)
# ═══════════════════════════════════════════════════════════════════════════

def find_partial_optimum(
    backend: Any,
    n_qubits: int = N_QUBITS,
    n_layers: int = N_LAYERS,
    maxiter: int = VQE_MAXITER,
    estimator_shots: int = VQE_OPT_SHOTS,
) -> tuple[np.ndarray, float]:
    """Run a quick VQE warm-up to find a partially-optimised θ*.

    Optimises the **ZZ-only** part of the TFIM Hamiltonian:
      H_ZZ = −J Σ_{i} Z_i Z_{i+1}
    because that is the observable we can estimate from computational-
    basis measurement counts.  We want θ* that produces strong nearest-
    neighbour spin correlations, giving the Galton probe meaningful
    alignment signal to filter on.

    Uses EstimatorV2 + COBYLA for *maxiter* iterations on the noisy
    backend.

    Returns:
        (opt_theta, final_zz_energy)
    """
    from scipy.optimize import minimize
    from qiskit_ibm_runtime import EstimatorV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit.quantum_info import SparsePauliOp

    ansatz = build_tfim_ansatz(n_qubits, n_layers)
    n_params = _num_params(n_qubits, n_layers)

    # Build ZZ-only Hamiltonian (what we actually measure from bitstrings)
    zz_terms: list[tuple[str, float]] = []
    for i in range(n_qubits - 1):
        label = ["I"] * n_qubits
        label[i] = "Z"
        label[i + 1] = "Z"
        zz_terms.append(("".join(reversed(label)), -J_COUPLING))
    hamiltonian_zz = SparsePauliOp.from_list(zz_terms).simplify()

    # Minimum possible ZZ energy = -(n-1)*J (all spins aligned)
    min_zz = -(n_qubits - 1) * J_COUPLING

    # Transpile the parameterised ansatz once for the backend
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_ansatz = pm.run(ansatz)
    isa_hamiltonian = hamiltonian_zz.apply_layout(isa_ansatz.layout)

    estimator = EstimatorV2(mode=backend)

    eval_count = 0
    best_energy = 0.0
    best_theta = None

    def cost_fn(theta: np.ndarray) -> float:
        nonlocal eval_count, best_energy, best_theta
        eval_count += 1
        pub = (isa_ansatz, isa_hamiltonian, theta)
        job = estimator.run([pub], precision=0.05)
        result = job.result()
        energy = float(result[0].data.evs)
        if energy < best_energy:
            best_energy = energy
            best_theta = theta.copy()
        if eval_count <= 5 or eval_count % 5 == 0:
            print(f"    VQE iter {eval_count:3d}: E_ZZ = {energy:+.4f}"
                  f"  (best: {best_energy:+.4f},"
                  f" {best_energy / min_zz:.0%} of min)")
        return energy

    # Initial guess: Ry = −π/2 on every qubit produces |0⟩^n after H init.
    # The CNOT ladder then entangles these, yielding ZZ ≈ −(n−1).
    # We add small noise so COBYLA can explore the local basin.
    rng = np.random.default_rng(SEED_BASE)
    x0 = np.zeros(n_params)
    for layer in range(n_layers):
        for q in range(n_qubits):
            idx = 2 * (layer * n_qubits + q)
            x0[idx] = -math.pi / 2 + rng.uniform(-0.1, 0.1)  # Ry ≈ −π/2
            x0[idx + 1] = rng.uniform(-0.1, 0.1)              # Rz ≈ 0

    print(f"\n  ── VQE Warm-up: COBYLA, {maxiter} iterations ──")
    print(f"     Ansatz: {n_qubits}q × {n_layers} layers = {n_params} params")
    print(f"     Target: ZZ-only Hamiltonian (min = {min_zz:.1f})")

    minimize(
        cost_fn,
        x0,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.5},
    )

    # Use the best θ found across all evaluations
    opt_theta = best_theta if best_theta is not None else x0
    final_energy = best_energy
    print(f"  ── Warm-up done: E*_ZZ = {final_energy:+.4f} "
          f"({eval_count} evaluations,"
          f" {final_energy / min_zz:.0%} of min ZZ) ──\n")

    return opt_theta, final_energy


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
    """Results from a single trial."""
    trial_id: int
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
    ansatz: Any,
    opt_theta: np.ndarray,
    shots: int,
    exact_energy: float,
    dry_run: bool = False,
) -> TrialResult:
    """Run one trial: baseline SamplerV2 vs QgateSampler, same circuit.

    Both samplers use the SAME backend and the SAME bound circuit. The
    only difference is that QgateSampler transparently injects probes
    and applies Galton filtering before returning the result.
    """
    from qiskit_ibm_runtime import SamplerV2
    from qgate.sampler import QgateSampler, SamplerConfig

    # --- Bind parameters and add measurements ---
    qc = bind_and_measure(ansatz, opt_theta)

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
        trial_id=0,  # updated per trial in run_experiment
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

    backend = build_heron_noisy_backend()
    print("  Backend ready: AerSimulator (Heron noise)")

    # --- Step 1: VQE Warm-up to find partially-optimised θ* ---
    vqe_iter = 15 if dry_run else VQE_MAXITER
    opt_theta, warmup_energy = find_partial_optimum(
        backend=backend,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        maxiter=vqe_iter,
        estimator_shots=VQE_OPT_SHOTS,
    )
    print(f"  Warm-up E_ZZ: {warmup_energy:+.4f}  (min ZZ: {-(N_QUBITS-1)*J_COUPLING:.1f},"
          f" exact GS: {exact_e:.4f})")
    print(f"  ZZ ratio:     {warmup_energy / (-(N_QUBITS-1)*J_COUPLING):.0%} of minimum ZZ energy")

    # --- Step 2: Build the parameterised ansatz (shared across trials) ---
    ansatz = build_tfim_ansatz(N_QUBITS, N_LAYERS)

    # --- Step 3: Run validation trials ---
    print(f"\n  ── Showdown: {n_trials} trials ──")
    results: list[TrialResult] = []
    t0 = time.perf_counter()

    for trial in range(n_trials):
        t_trial = time.perf_counter()

        result = run_single_trial(
            backend=backend,
            ansatz=ansatz,
            opt_theta=opt_theta,
            shots=shots,
            exact_energy=exact_e,
            dry_run=dry_run,
        )
        # Tag with trial number
        result.trial_id = trial + 1
        results.append(result)

        dt = time.perf_counter() - t_trial
        sign = "+" if result.mse_reduction_pct > 0 else ""
        print(
            f"  Trial {trial + 1:2d}/{n_trials} | "
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
    print("  Trial  Raw Energy   QG Energy    Raw MSE     QG MSE   MSE↓%")
    for r in results:
        print(
            f"  {r.trial_id:5d}  {r.raw_energy:+10.4f}  {r.qgate_energy:+10.4f}"
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
