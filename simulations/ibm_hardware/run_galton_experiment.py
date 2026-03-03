#!/usr/bin/env python3
"""
run_galton_experiment.py — 3-mode threshold comparison on IBM Quantum hardware.

Compares fixed, rolling_z, and galton thresholding on the SAME hardware
outcomes for a fair apples-to-apples comparison.

Strategy:
  1. For each circuit config (N, W, D), build & execute ONCE on hardware
     with 10K shots.
  2. Apply the three threshold modes to the identical outcomes.
  3. Record acceptance probability, TTS, effective threshold, and galton
     telemetry for every (config, mode) combination.

This keeps QPU time minimal — we only pay for circuit execution once per
config, then post-process three ways.

Usage:
    # Dry-run with mock adapter (no hardware)
    python run_galton_experiment.py --mode mock

    # Local noisy Aer simulation
    python run_galton_experiment.py --mode aer

    # Real IBM hardware
    python run_galton_experiment.py --mode ibm --token <YOUR_TOKEN>

    # Customise
    python run_galton_experiment.py --mode ibm --shots 10000 \\
        --N 2 4 8 --W 2 4 --D 2 4

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Ensure qgate is importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "packages" / "qgate" / "src"))

from qgate.config import (
    ConditioningVariant,
    DynamicThresholdConfig,
    FusionConfig,
    GateConfig,
)
from qgate.conditioning import ParityOutcome
from qgate.filter import TrajectoryFilter
from qgate.scoring import score_batch
from qgate.threshold import GaltonAdaptiveThreshold

# Also need the circuit-building and sampler machinery
sys.path.insert(0, str(ROOT))
from qiskit_experiment.circuits import build_monitoring_circuit


# ═══════════════════════════════════════════════════════════════════════════
# Backend setup (reuses patterns from run_ibm_experiment.py)
# ═══════════════════════════════════════════════════════════════════════════

def get_aer_backend(noise: bool = True):
    from qiskit_aer import AerSimulator
    if not noise:
        return AerSimulator()
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(1e-3, 1),
                                      ["rx", "ry", "rz", "h", "x", "u"])
    model.add_all_qubit_quantum_error(depolarizing_error(1e-2, 2), ["cx"])
    model.add_all_qubit_quantum_error(depolarizing_error(2e-2, 1), ["measure"])
    return AerSimulator(noise_model=model)


def get_ibm_backend(token: str | None = None, min_qubits: int = 16):
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
                channel="ibm_quantum_platform", token=token, overwrite=True,
            )
        except Exception:
            pass
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform", token=token,
        )
    else:
        service = QiskitRuntimeService(channel="ibm_quantum_platform")

    print(f"Searching for backends with ≥{min_qubits} qubits...")
    backend = service.least_busy(
        min_num_qubits=min_qubits, simulator=False, operational=True,
    )
    print(f"Selected backend: {backend.name} ({backend.num_qubits} qubits)")
    return backend


# ═══════════════════════════════════════════════════════════════════════════
# Sampler helper
# ═══════════════════════════════════════════════════════════════════════════

def run_sampler(circuit, backend, shots: int) -> dict[str, int]:
    """Run circuit and return counts dict. Tries V2 Sampler first."""
    try:
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        from qiskit.transpiler.preset_passmanagers import (
            generate_preset_pass_manager,
        )
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa = pm.run(circuit)
        job = Sampler(mode=backend).run([isa], shots=shots)
        pub = job.result()[0]
        creg_name = circuit.cregs[0].name
        return pub.data[creg_name].get_counts()
    except Exception:
        pass
    # Fallback
    from qiskit import transpile
    t = transpile(circuit, backend=backend, optimization_level=1)
    job = backend.run(t, shots=shots)
    return job.result().get_counts(0)


# ═══════════════════════════════════════════════════════════════════════════
# Parse hardware counts → qgate ParityOutcome list
# ═══════════════════════════════════════════════════════════════════════════

def counts_to_outcomes(
    counts: dict[str, int], n_subsystems: int, n_cycles: int,
) -> list[ParityOutcome]:
    """Convert Qiskit count dict to a list of ParityOutcome objects."""
    outcomes: list[ParityOutcome] = []
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        bits_list = [int(b) for b in reversed(bits)]

        matrix = []
        for w in range(n_cycles):
            row = []
            for s in range(n_subsystems):
                idx = w * n_subsystems + s
                row.append(bits_list[idx] if idx < len(bits_list) else 1)
            matrix.append(row)

        base = np.array(matrix, dtype=np.int8)
        for _ in range(count):
            outcomes.append(ParityOutcome(
                n_subsystems=n_subsystems,
                n_cycles=n_cycles,
                parity_matrix=base.copy(),
            ))
    return outcomes


# ═══════════════════════════════════════════════════════════════════════════
# Threshold mode definitions
# ═══════════════════════════════════════════════════════════════════════════

THRESHOLD_MODES: dict[str, DynamicThresholdConfig] = {
    "fixed": DynamicThresholdConfig(
        mode="fixed",
        baseline=0.65,
    ),
    "rolling_z": DynamicThresholdConfig(
        mode="rolling_z",
        baseline=0.65,
        z_factor=1.0,
        window_size=20,
        min_threshold=0.3,
        max_threshold=0.95,
    ),
    "galton_quantile": DynamicThresholdConfig(
        mode="galton",
        baseline=0.65,
        window_size=5000,
        min_window_size=500,
        target_acceptance=0.10,
        use_quantile=True,
        robust_stats=True,
        min_threshold=0.3,
        max_threshold=0.99,
    ),
    "galton_zscore": DynamicThresholdConfig(
        mode="galton",
        baseline=0.65,
        window_size=5000,
        min_window_size=500,
        target_acceptance=0.10,
        use_quantile=False,
        robust_stats=True,
        z_sigma=1.282,  # ~10% one-sided tail for normal
        min_threshold=0.3,
        max_threshold=0.99,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Apply threshold modes to a shared set of outcomes
# ═══════════════════════════════════════════════════════════════════════════

def apply_modes(
    outcomes: list[ParityOutcome],
    n_sub: int, n_cyc: int, depth: int,
    alpha: float = 0.5,
) -> list[dict[str, Any]]:
    """Apply all threshold modes to the same outcomes and return rows."""
    from qgate.adapters.base import MockAdapter

    rows: list[dict[str, Any]] = []

    for mode_name, dt_cfg in THRESHOLD_MODES.items():
        config = GateConfig(
            n_subsystems=n_sub,
            n_cycles=n_cyc,
            shots=len(outcomes),
            variant=ConditioningVariant.SCORE_FUSION,
            fusion=FusionConfig(alpha=alpha, threshold=0.65),
            dynamic_threshold=dt_cfg,
        )
        # We create a filter but use .filter() with pre-built outcomes
        # (the adapter is unused — we already have hardware outcomes)
        adapter = MockAdapter(seed=0)
        tf = TrajectoryFilter(config, adapter)

        # For rolling_z we need to feed multiple batches to let it adapt.
        # Split outcomes into 10 mini-batches, filter sequentially.
        if dt_cfg.mode == "rolling_z":
            n_total = len(outcomes)
            batch_size = max(1, n_total // 10)
            all_accepted = 0
            all_scores: list[float] = []
            for i in range(0, n_total, batch_size):
                batch = outcomes[i : i + batch_size]
                r = tf.filter(batch)
                all_accepted += r.accepted_shots
                all_scores.extend(r.scores)
            acc_prob = all_accepted / n_total if n_total > 0 else 0.0
            tts = 1.0 / acc_prob if acc_prob > 0 else float("inf")
            mean_score = float(np.mean(all_scores)) if all_scores else None
            eff_threshold = tf.current_threshold
            galton_meta = {}
        else:
            r = tf.filter(outcomes)
            acc_prob = r.acceptance_probability
            tts = r.tts
            mean_score = r.mean_combined_score
            eff_threshold = r.threshold_used
            galton_meta = r.metadata.get("galton", {})

        row: dict[str, Any] = {
            "N": n_sub,
            "W": n_cyc,
            "D": depth,
            "alpha": alpha,
            "threshold_mode": mode_name,
            "total_shots": len(outcomes),
            "accepted_shots": int(acc_prob * len(outcomes)),
            "acceptance_probability": round(acc_prob, 6),
            "TTS": round(tts, 4),
            "mean_combined_score": round(mean_score, 6) if mean_score else None,
            "effective_threshold": round(eff_threshold, 6),
        }
        # Galton telemetry
        if galton_meta:
            for k, v in galton_meta.items():
                key = k.replace("galton_", "")
                row[key] = round(v, 6) if isinstance(v, float) else v

        rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3-mode threshold comparison (fixed / rolling_z / galton)"
                    " on IBM Quantum hardware.",
    )
    p.add_argument("--mode", choices=["mock", "aer", "ibm"], default="mock")
    p.add_argument("--token", type=str, default=None)
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--N", nargs="+", type=int, default=[2, 4, 8])
    p.add_argument("--W", nargs="+", type=int, default=[2, 4])
    p.add_argument("--D", nargs="+", type=int, default=[2, 4])
    p.add_argument("--alpha", nargs="+", type=float, default=[0.5])
    p.add_argument("--shots", type=int, default=10000)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Default output directory
    if args.output_dir is None:
        args.output_dir = str(
            Path(__file__).resolve().parent / "galton_experiment"
        )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("GALTON ADAPTIVE THRESHOLD — IBM HARDWARE COMPARISON")
    print("fixed  vs  rolling_z  vs  galton_quantile  vs  galton_zscore")
    print("Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915")
    print("=" * 72)
    print(f"Mode:     {args.mode}")
    print(f"N:        {args.N}")
    print(f"W:        {args.W}")
    print(f"D:        {args.D}")
    print(f"alpha:    {args.alpha}")
    print(f"Shots:    {args.shots}")
    print(f"Output:   {out_dir}")
    print(f"Circuits: {len(args.N) * len(args.W) * len(args.D)} configs "
          f"× {args.shots} shots each")
    print()

    # --- Backend ---
    t0 = time.time()
    backend = None
    if args.mode == "mock":
        print("Using MockAdapter (no Qiskit needed)")
    elif args.mode == "aer":
        backend = get_aer_backend(noise=not args.no_noise)
        noise_str = "noisy" if not args.no_noise else "ideal"
        print(f"Backend: AerSimulator ({noise_str})")
    else:
        n_max = max(args.N)
        backend = get_ibm_backend(token=args.token, min_qubits=3 * n_max)
    print()

    all_results: list[dict[str, Any]] = []
    total_configs = len(args.N) * len(args.W) * len(args.D)
    idx = 0

    for N in args.N:
        for W in args.W:
            for D in args.D:
                idx += 1
                tag = f"N={N} W={W} D={D}"
                elapsed = time.time() - t0
                print(f"[{idx}/{total_configs}] {tag}  "
                      f"(elapsed {elapsed:.0f}s)")

                # --- Get outcomes ---
                if args.mode == "mock":
                    # Use qgate MockAdapter to generate synthetic outcomes
                    from qgate.adapters.base import MockAdapter
                    adapter = MockAdapter(error_rate=0.08, seed=42 + idx)
                    outcomes = adapter.build_and_run(
                        n_subsystems=N, n_cycles=W, shots=args.shots,
                    )
                    print(f"  MockAdapter: {len(outcomes)} outcomes generated")
                else:
                    # Build circuit & run on hardware / Aer
                    print(f"  Building circuit "
                          f"({2*N} data + 1 ancilla qubits)...", end=" ")
                    circ = build_monitoring_circuit(N, D, W)
                    print(f"done ({circ.num_qubits} qubits, "
                          f"depth~{circ.depth()})")

                    print(f"  Executing {args.shots} shots...", end=" ",
                          flush=True)
                    counts = run_sampler(circ, backend, args.shots)
                    print(f"done ({len(counts)} unique bitstrings)")

                    outcomes = counts_to_outcomes(counts, N, W)
                    print(f"  Parsed {len(outcomes)} shot outcomes")

                # --- Apply all threshold modes ---
                for alpha in args.alpha:
                    rows = apply_modes(outcomes, N, W, D, alpha=alpha)
                    for row in rows:
                        all_results.append(row)
                        if not args.quiet:
                            m = row["threshold_mode"]
                            a = row["acceptance_probability"]
                            t = row["effective_threshold"]
                            print(f"    {m:18s}  P_acc={a:.4f}  "
                                  f"theta={t:.4f}")

    # --- Save CSV ---
    csv_path = out_dir / "galton_results.csv"
    if all_results:
        fieldnames = list(all_results[0].keys())
        # Gather all keys (galton rows have extra fields)
        for r in all_results:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

    # --- Save run log ---
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "backend": str(backend) if backend else "MockAdapter",
        "shots": args.shots,
        "N_values": args.N,
        "W_values": args.W,
        "D_values": args.D,
        "alpha_values": args.alpha,
        "threshold_modes": list(THRESHOLD_MODES.keys()),
        "total_circuit_configs": total_configs,
        "total_rows": len(all_results),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "run_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)

    # --- Print summary table ---
    print(f"\n{'=' * 72}")
    print("SUMMARY — Acceptance Probability by Mode")
    print(f"{'=' * 72}")
    print(f"{'Config':<16} {'fixed':>8} {'rolling_z':>10} "
          f"{'galton_q':>10} {'galton_z':>10}")
    print("-" * 56)

    # Group by (N, W, D)
    from itertools import groupby
    def config_key(r):
        return (r["N"], r["W"], r["D"])
    sorted_results = sorted(all_results, key=config_key)
    for key, group in groupby(sorted_results, key=config_key):
        group_list = list(group)
        vals = {}
        for r in group_list:
            vals[r["threshold_mode"]] = r["acceptance_probability"]
        label = f"N={key[0]} W={key[1]} D={key[2]}"
        print(f"{label:<16} "
              f"{vals.get('fixed', 0):>8.4f} "
              f"{vals.get('rolling_z', 0):>10.4f} "
              f"{vals.get('galton_quantile', 0):>10.4f} "
              f"{vals.get('galton_zscore', 0):>10.4f}")

    # --- Effective threshold comparison ---
    print(f"\n{'Config':<16} {'fixed':>8} {'rolling_z':>10} "
          f"{'galton_q':>10} {'galton_z':>10}")
    print("-" * 56)
    print("Effective Threshold (θ)")
    for key, group in groupby(
        sorted(all_results, key=config_key), key=config_key
    ):
        group_list = list(group)
        vals = {}
        for r in group_list:
            vals[r["threshold_mode"]] = r["effective_threshold"]
        label = f"N={key[0]} W={key[1]} D={key[2]}"
        print(f"{label:<16} "
              f"{vals.get('fixed', 0):>8.4f} "
              f"{vals.get('rolling_z', 0):>10.4f} "
              f"{vals.get('galton_quantile', 0):>10.4f} "
              f"{vals.get('galton_zscore', 0):>10.4f}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"Experiment complete!")
    print(f"Total time:  {elapsed:.1f}s")
    print(f"Results:     {csv_path} ({len(all_results)} rows)")
    print(f"Run log:     {out_dir / 'run_log.json'}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
