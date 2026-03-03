#!/usr/bin/env python3
"""
run_tfim_dryrun.py — 16-Qubit TFIM Critical-Phase Dry Run
==========================================================

Pre-flight validation for the 127-qubit TFIM experiment:
"Beating Zero-Noise Extrapolation: Solving the 127-Qubit TFIM Critical
Phase via Time-Symmetric Trajectory Filtering"

This script validates the full qgate TSVF pipeline on a 16-qubit 1D
Transverse-Field Ising Model at the critical point (h/J ≈ 3.04) where
classical methods struggle with critical correlations.

**Pipeline:**
  1. Build a VQE ansatz for the 16-qubit 1D TFIM Hamiltonian.
  2. Apply qgate TSVF chaotic-ansatz + energy probe post-selection.
  3. Score-fusion + Galton adaptive thresholding filters thermal noise.
  4. Extract purified energy expectations.

**Modes:**
  --mode aer    Local AerSimulator with realistic noise (DEFAULT — free)
  --mode ibm    Real IBM Quantum hardware (reads IBMQ_TOKEN or .secrets.json)

**What to check before authorising the 127-qubit Torino run:**
  ✓ Transpilation depth blow-up: 3×–5× is golden, >100× means re-routing needed
  ✓ Galton threshold: must NOT be NaN or None
  ✓ Acceptance rate: 5%–20% is the sweet spot
  ✓ Energy estimate: closer to exact ground-state = better filtering

Usage:
    # Safe local dry-run (no IBM credits):
    python run_tfim_dryrun.py --mode aer

    # IBM hardware dry-run (uses token):
    python run_tfim_dryrun.py --mode ibm

    # Specify backend explicitly:
    python run_tfim_dryrun.py --mode ibm --backend ibm_brisbane

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── Ensure qgate is importable ──
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "packages" / "qgate" / "src"))

from qgate import (
    ConditioningVariant,
    GateConfig,
    TrajectoryFilter,
    VQETSVFAdapter,
)
from qgate.adapters.vqe_adapter import (
    energy_error,
    estimate_energy_from_counts,
    tfim_exact_ground_energy,
)
from qgate.config import DynamicThresholdConfig, FusionConfig

# ═══════════════════════════════════════════════════════════════════════════
# Constants — 16-qubit TFIM at critical point
# ═══════════════════════════════════════════════════════════════════════════

N_QUBITS = 16
J_COUPLING = 1.0           # Nearest-neighbour ZZ coupling
H_FIELD = 3.04             # Transverse field — critical point h/J ≈ 3.04
N_LAYERS = 3               # Ansatz depth (shallow for dry run)
SHOTS = 10_000             # Conservative for free tier / dry run
SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent


def compute_tfim_ground_energy_fast(n_qubits: int, j_coupling: float, h_field: float) -> float:
    """Compute ground-state energy using a sparse Hamiltonian + Lanczos (SciPy).

    This avoids building the full dense 2^n x 2^n matrix and is
    considerably faster and memory-efficient for n up to ~18.
    Falls back by raising ImportError if SciPy isn't available.
    """
    try:
        from scipy import sparse
        from scipy.sparse.linalg import eigsh
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("SciPy sparse eigensolver not available") from exc

    dim = 1 << n_qubits
    print(f"       Building sparse TFIM Hamiltonian (dim={dim})...")

    # Diagonal (ZZ) contribution computed per basis state
    diag = np.zeros(dim, dtype=float)

    rows = []
    cols = []
    data = []

    progress_step = max(1, dim // 20)
    t0 = time.time()

    for b in range(dim):
        # bits: lowest-order bit = qubit 0
        spins = [1 - 2 * ((b >> i) & 1) for i in range(n_qubits)]
        zz = 0.0
        for i in range(n_qubits - 1):
            zz -= j_coupling * spins[i] * spins[i + 1]
        diag[b] = zz

        # X operator: off-diagonal -h connects b <-> b ^ (1<<i)
        for i in range(n_qubits):
            b2 = b ^ (1 << i)
            rows.append(b)
            cols.append(b2)
            data.append(-h_field)

        if (b + 1) % progress_step == 0:
            pct = (b + 1) * 100.0 / dim
            print(f"         ... {pct:.0f}% built ({b+1}/{dim}) — elapsed {time.time()-t0:.1f}s")

    # Assemble sparse matrix (COO -> CSR)
    H_off = sparse.coo_matrix((data, (rows, cols)), shape=(dim, dim))
    H = H_off.tocsr() + sparse.diags(diag, format="csr")

    print("       Running Lanczos (sparse) eigensolver (k=1, which='SA')...")
    # Compute smallest algebraic eigenvalue
    vals, vecs = eigsh(H, k=1, which="SA")
    return float(vals[0])


# ═══════════════════════════════════════════════════════════════════════════
# Backend setup
# ═══════════════════════════════════════════════════════════════════════════

def get_aer_backend(noise: bool = True):
    """AerSimulator with IBM-calibre noise model.

    Models T1/T2 relaxation, depolarising gate errors, and measurement
    errors at levels representative of IBM Eagle/Heron processors.
    """
    from qiskit_aer import AerSimulator

    if not noise:
        return AerSimulator()

    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        thermal_relaxation_error,
    )

    model = NoiseModel()

    # IBM Heron-class noise parameters
    t1 = 300e3       # T1 ~ 300 µs (Heron r2)
    t2 = 150e3       # T2 ~ 150 µs
    gate_1q = 60     # Single-qubit gate time ~ 60 ns
    gate_2q = 660    # CX gate time ~ 660 ns
    gate_meas = 1200 # Measurement time ~ 1.2 µs

    # Single-qubit errors
    err_1q = thermal_relaxation_error(t1, t2, gate_1q)
    model.add_all_qubit_quantum_error(
        err_1q, ["rx", "ry", "rz", "h", "x", "z", "s", "sdg", "u"],
    )
    dep_1q = depolarizing_error(1e-3, 1)
    model.add_all_qubit_quantum_error(
        dep_1q, ["rx", "ry", "rz", "h", "x", "z", "s", "sdg", "u"],
    )

    # Two-qubit errors (CX error ~ 1%)
    err_2q = thermal_relaxation_error(t1, t2, gate_2q).expand(
        thermal_relaxation_error(t1, t2, gate_2q),
    )
    model.add_all_qubit_quantum_error(err_2q, ["cx"])
    dep_2q = depolarizing_error(1e-2, 2)
    model.add_all_qubit_quantum_error(dep_2q, ["cx"])

    # Measurement errors
    err_meas = thermal_relaxation_error(t1, t2, gate_meas)
    model.add_all_qubit_quantum_error(err_meas, ["measure"])

    return AerSimulator(noise_model=model)


def get_ibm_backend(
    token: str | None = None,
    backend_name: str | None = None,
    min_qubits: int = 16,
):
    """Connect to IBM Quantum and select a backend.

    Resolution order for token:
      1. Explicit --token argument
      2. IBMQ_TOKEN environment variable
      3. .secrets.json file in repo root
      4. Saved credentials from previous QiskitRuntimeService.save_account()
    """
    if not token:
        token = os.environ.get("IBMQ_TOKEN")
    if not token:
        secrets = ROOT / ".secrets.json"
        if secrets.is_file():
            with open(secrets) as f:
                token = json.load(f).get("ibmq_token")

    from qiskit_ibm_runtime import QiskitRuntimeService

    if token:
        try:
            QiskitRuntimeService.save_account(
                channel="ibm_quantum_platform",
                token=token,
                overwrite=True,
            )
        except Exception:
            pass
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform", token=token,
        )
    else:
        service = QiskitRuntimeService(channel="ibm_quantum_platform")

    if backend_name:
        backend = service.backend(backend_name)
        print(f"Using requested backend: {backend.name} ({backend.num_qubits} qubits)")
    else:
        print(f"Searching for least-busy backend with ≥{min_qubits} qubits...")
        backend = service.least_busy(
            min_num_qubits=min_qubits, simulator=False, operational=True,
        )
        print(f"Selected backend: {backend.name} ({backend.num_qubits} qubits)")

    return backend


# ═══════════════════════════════════════════════════════════════════════════
# Main dry-run experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_dryrun(backend, mode: str, n_qubits: int = N_QUBITS) -> dict:
    """Execute the 16-qubit TFIM dry run with qgate TSVF pipeline.

    Returns a results dict for programmatic inspection.
    """
    print("\n" + "=" * 70)
    print("  16-QUBIT TFIM CRITICAL-PHASE DRY RUN")
    print("  H = -J Σ Z_i Z_{i+1}  -  h Σ X_i")
    print(f"  J = {J_COUPLING},  h = {H_FIELD}  (h/J = {H_FIELD/J_COUPLING:.2f} — critical)")
    print(f"  Qubits: {n_qubits},  Layers: {N_LAYERS},  Shots: {SHOTS:,}")
    print(f"  Mode: {mode}")
    print("=" * 70)

    # ── Step 1: Compute exact ground-state energy (classical benchmark) ──
    print("\n[1/6] Computing exact ground-state energy (classical diag)...")
    t0 = time.time()
    # Prefer a sparse Lanczos-based solver (SciPy) to avoid dense OOM
    try:
        exact_energy = compute_tfim_ground_energy_fast(n_qubits, J_COUPLING, H_FIELD)
        dt_exact = time.time() - t0
        print(f"       E_exact = {exact_energy:.6f}  ({dt_exact:.2f}s) — sparse Lanczos")
    except Exception:
        print("       SciPy sparse eigensolver not available or failed — falling back to dense diagonalisation (may be very slow / OOM)")
        t0 = time.time()
        exact_energy = tfim_exact_ground_energy(n_qubits, J_COUPLING, H_FIELD)
        dt_exact = time.time() - t0
        print(f"       E_exact = {exact_energy:.6f}  ({dt_exact:.2f}s) — dense diag")
    print(f"       (This is the target — our filtered estimate should approach this)")

    # ── Step 2: Configure the TSVF adapter ──
    print("\n[2/6] Configuring VQETSVFAdapter with energy-probe post-selection...")
    tsvf_adapter = VQETSVFAdapter(
        backend=backend,
        algorithm_mode="tsvf",
        n_qubits=n_qubits,
        j_coupling=J_COUPLING,
        h_field=H_FIELD,
        seed=SEED,
        weak_angle_base=np.pi / 4,
        weak_angle_ramp=np.pi / 8,
        optimization_level=2,
    )
    print(f"       Algorithm: TSVF (chaotic ansatz + energy probe)")
    print(f"       Probe angles: base=π/4, ramp=π/8 per layer")
    print(f"       Optimisation level: 2 (ISA transpilation)")

    # Also build a standard adapter for comparison
    std_adapter = VQETSVFAdapter(
        backend=backend,
        algorithm_mode="standard",
        n_qubits=n_qubits,
        j_coupling=J_COUPLING,
        h_field=H_FIELD,
        seed=SEED,
        optimization_level=2,
    )

    # ── Step 3: Build circuits and check transpilation depth ──
    print("\n[3/6] Building circuits and checking ISA transpilation...")
    std_circuit = std_adapter.build_circuit(n_qubits, N_LAYERS)
    tsvf_circuit = tsvf_adapter.build_circuit(n_qubits, N_LAYERS)

    depth_original_std = std_circuit.depth()
    depth_original_tsvf = tsvf_circuit.depth()

    depth_transpiled_std = std_adapter.get_transpiled_depth(std_circuit)
    depth_transpiled_tsvf = tsvf_adapter.get_transpiled_depth(tsvf_circuit)

    ratio_std = depth_transpiled_std / max(depth_original_std, 1)
    ratio_tsvf = depth_transpiled_tsvf / max(depth_original_tsvf, 1)

    print(f"       Standard VQE:")
    print(f"         Original depth:    {depth_original_std}")
    print(f"         Transpiled depth:  {depth_transpiled_std}")
    print(f"         Blow-up ratio:     {ratio_std:.1f}×")
    print(f"       TSVF VQE:")
    print(f"         Original depth:    {depth_original_tsvf}")
    print(f"         Transpiled depth:  {depth_transpiled_tsvf}")
    print(f"         Blow-up ratio:     {ratio_tsvf:.1f}×")

    if ratio_tsvf > 100:
        print("       ⚠  WARNING: Depth blow-up >100× — re-routing needed!")
    elif ratio_tsvf > 20:
        print("       ⚡ Moderate blow-up — consider lowering opt_level or pre-routing")
    else:
        print("       ✅ Depth ratio in acceptable range for hardware execution")

    # ── Step 4: Execute standard VQE (baseline) ──
    print(f"\n[4/6] Executing standard VQE (no TSVF) — {SHOTS:,} shots...")
    t0 = time.time()
    std_raw = std_adapter.run(std_circuit, SHOTS)
    dt_std = time.time() - t0
    std_counts = std_adapter._extract_counts(std_raw)
    energy_std = estimate_energy_from_counts(std_counts, n_qubits, J_COUPLING)
    err_std = energy_error(energy_std, exact_energy)
    print(f"       E_standard = {energy_std:.6f}")
    print(f"       |Error|    = {err_std:.6f}")
    print(f"       ({dt_std:.1f}s execution)")

    # ── Step 5: Execute TSVF VQE with qgate Galton filtering ──
    print(f"\n[5/6] Executing TSVF VQE with qgate Score Fusion + Galton...")

    # Configure the trajectory filter
    tsvf_config = GateConfig(
        n_subsystems=n_qubits,
        n_cycles=N_LAYERS,
        shots=SHOTS,
        variant=ConditioningVariant.SCORE_FUSION,
        fusion=FusionConfig(
            alpha=0.8,            # Weight toward low-frequency stability
            threshold=0.5,        # Base accept threshold
        ),
        dynamic_threshold=DynamicThresholdConfig(
            mode="galton",
            target_acceptance=0.15,   # Relaxed for dry run
            min_window_size=50,       # Warmup on 50 shots
            window_size=2000,         # Rolling window
            use_quantile=True,        # Empirical quantile (recommended)
            min_threshold=0.15,       # Floor
            max_threshold=0.95,       # Ceiling
        ),
        adapter="mock",  # Unused — we pass the adapter directly
        metadata={
            "experiment": f"tfim_dryrun_{n_qubits}q",
            "h_field": H_FIELD,
            "j_coupling": J_COUPLING,
            "n_qubits": n_qubits,
            "n_layers": N_LAYERS,
        },
    )

    t0 = time.time()

    # Build & run the TSVF circuit
    tsvf_raw = tsvf_adapter.run(tsvf_circuit, SHOTS)
    dt_exec = time.time() - t0

    # Parse into ParityOutcome objects
    tsvf_outcomes = tsvf_adapter.parse_results(tsvf_raw, n_qubits, N_LAYERS)

    # Apply trajectory filtering
    tf = TrajectoryFilter(tsvf_config, tsvf_adapter)
    result = tf.filter(tsvf_outcomes)

    dt_total = time.time() - t0

    # Extract energy from post-selected counts
    tsvf_counts = tsvf_adapter._extract_counts(tsvf_raw)

    # Post-selected energy: filter to accepted trajectories
    # Use the ancilla bit to post-select
    accepted_counts = {}
    total_accepted = 0
    for bs, cnt in tsvf_counts.items():
        bs_str = str(bs).strip()
        if " " in bs_str:
            parts = bs_str.split()
            anc_bit = parts[0]
            search_bits = parts[-1]
        else:
            anc_bit = bs_str[0]
            search_bits = bs_str[1:]
        if anc_bit == "1":
            accepted_counts[search_bits] = accepted_counts.get(search_bits, 0) + cnt
            total_accepted += cnt

    if total_accepted > 0:
        energy_tsvf = estimate_energy_from_counts(
            accepted_counts, n_qubits, J_COUPLING,
        )
    else:
        energy_tsvf = energy_std  # Fallback
        print("       ⚠  No ancilla post-selection accepted — using unfiltered")

    err_tsvf = energy_error(energy_tsvf, exact_energy)

    # Galton telemetry
    galton_meta = result.metadata.get("galton", {})
    galton_threshold = galton_meta.get(
        "galton_effective_threshold",
        result.threshold_used,
    )
    galton_in_warmup = galton_meta.get("galton_in_warmup", None)
    galton_window = galton_meta.get("galton_window_size_current", None)
    galton_rolling_mean = galton_meta.get("galton_rolling_mean", None)

    print(f"       Execution time:      {dt_exec:.1f}s")
    print(f"       Parse + filter time: {dt_total - dt_exec:.1f}s")
    print(f"       E_tsvf_filtered   =  {energy_tsvf:.6f}")
    print(f"       |Error|           =  {err_tsvf:.6f}")

    # ── Step 6: Validation summary ──
    print("\n" + "=" * 70)
    print("  DRY RUN VALIDATION REPORT")
    print("=" * 70)

    improvement = (err_std - err_tsvf) / max(err_std, 1e-12) * 100

    print(f"\n  Problem:          {n_qubits}-qubit 1D TFIM at h/J = {H_FIELD/J_COUPLING:.2f}")
    print(f"  Exact GS energy:  {exact_energy:.6f}")
    print(f"  Backend:          {getattr(backend, 'name', 'AerSimulator')}")
    print(f"                    ({getattr(backend, 'num_qubits', n_qubits)} physical qubits)")

    print(f"\n  ┌─ TRANSPILATION ────────────────────────────────────────")
    print(f"  │  Standard depth:  {depth_original_std} → {depth_transpiled_std} ({ratio_std:.1f}×)")
    print(f"  │  TSVF depth:      {depth_original_tsvf} → {depth_transpiled_tsvf} ({ratio_tsvf:.1f}×)")
    status = "✅ PASS" if ratio_tsvf < 50 else "⚠  CHECK"
    print(f"  │  Status:          {status}")

    print(f"  │")
    print(f"  ├─ GALTON THRESHOLD ─────────────────────────────────────")
    print(f"  │  Effective θ:     {galton_threshold}")
    print(f"  │  In warmup:       {galton_in_warmup}")
    print(f"  │  Window size:     {galton_window}")
    print(f"  │  Rolling mean:    {galton_rolling_mean}")
    status = "✅ PASS" if galton_threshold is not None and not np.isnan(float(galton_threshold)) else "❌ FAIL"
    print(f"  │  Status:          {status}")

    print(f"  │")
    print(f"  ├─ ACCEPTANCE RATE ──────────────────────────────────────")
    print(f"  │  Total shots:     {result.total_shots:,}")
    print(f"  │  Accepted:        {result.accepted_shots:,}")
    print(f"  │  qgate accept %:  {result.acceptance_probability:.2%}")
    print(f"  │  Ancilla accept:  {total_accepted:,}/{sum(tsvf_counts.values()):,} ({total_accepted/max(sum(tsvf_counts.values()),1):.1%})")
    print(f"  │  TTS:             {result.tts:.2f}")
    ok = 0.01 <= result.acceptance_probability <= 0.50
    status = "✅ PASS" if ok else "⚠  REVIEW"
    print(f"  │  Status:          {status}")

    print(f"  │")
    print(f"  ├─ ENERGY ESTIMATION ────────────────────────────────────")
    print(f"  │  E_standard:      {energy_std:.6f}  (|err| = {err_std:.4f})")
    print(f"  │  E_tsvf:          {energy_tsvf:.6f}  (|err| = {err_tsvf:.4f})")
    print(f"  │  Improvement:     {improvement:+.1f}%")
    status = "✅ TSVF WINS" if err_tsvf < err_std else "⚠  STANDARD WINS"
    print(f"  │  Status:          {status}")

    print(f"  │")
    print(f"  └─ VERDICT ─────────────────────────────────────────────")
    all_pass = (
        ratio_tsvf < 50
        and galton_threshold is not None
        and not np.isnan(float(galton_threshold))
        and result.acceptance_probability > 0
    )
    if all_pass:
        print(f"     ✅ ALL CHECKS PASSED — Cleared for 127-qubit Torino run")
    else:
        print(f"     ⚠  REVIEW NEEDED — Check flagged items above")

    print()

    return {
        "exact_energy": exact_energy,
        "energy_standard": energy_std,
        "energy_tsvf": energy_tsvf,
        "error_standard": err_std,
        "error_tsvf": err_tsvf,
        "improvement_pct": improvement,
        "depth_original_tsvf": depth_original_tsvf,
        "depth_transpiled_tsvf": depth_transpiled_tsvf,
        "depth_ratio_tsvf": ratio_tsvf,
        "galton_threshold": galton_threshold,
        "galton_in_warmup": galton_in_warmup,
        "acceptance_probability": result.acceptance_probability,
        "accepted_shots": result.accepted_shots,
        "total_shots": result.total_shots,
        "ancilla_accepted": total_accepted,
        "tts": result.tts,
        "mode": mode,
        "backend": getattr(backend, "name", "AerSimulator"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="16-Qubit TFIM Critical-Phase Dry Run — qgate TSVF Pipeline Validation",
    )
    parser.add_argument(
        "--mode",
        choices=["aer", "ibm"],
        default="aer",
        help="Execution mode: 'aer' (local simulator) or 'ibm' (real hardware)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="IBM Quantum token (or set IBMQ_TOKEN env var)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Specific IBM backend name (e.g. ibm_brisbane, ibm_torino)",
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Use ideal (noiseless) Aer simulator",
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=N_QUBITS,
        help="Override the number of qubits for a smaller/faster dry-run",
    )
    args = parser.parse_args()

    if args.mode == "aer":
        backend = get_aer_backend(noise=not args.no_noise)
    else:
        backend = get_ibm_backend(
            token=args.token,
            backend_name=args.backend,
            min_qubits=args.n_qubits,
        )

    results = run_dryrun(backend, mode=args.mode, n_qubits=args.n_qubits)

    # Save results JSON
    out_dir = SCRIPT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dryrun_{results['backend']}_{args.n_qubits}q.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
