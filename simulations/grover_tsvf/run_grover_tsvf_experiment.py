#!/usr/bin/env python3
"""
run_grover_tsvf_experiment.py — Grover vs TSVF-Chaotic Grover comparison.

**Powered by the qgate package** (``pip install -e packages/qgate[qiskit]``).

Compares standard Grover's search with a time-symmetric / chaotic-ansatz
post-selected variant across increasing circuit depth (1–10 iterations),
demonstrating that post-selection stabilises search probability against
decoherence and amplitude overshoot.

Uses qgate's:
  • ``GroverTSVFAdapter`` — builds & runs circuits, maps to ParityOutcome.
  • ``TrajectoryFilter``  — scoring, Galton adaptive thresholding, conditioning.
  • ``GateConfig``        — declarative experiment configuration.
  • ``RunLogger``         — structured JSONL telemetry.
  • ``FilterResult``      — acceptance statistics & TTS.

Target state: |101⟩  (decimal 5 in a 3-qubit register)

Modes:
  --mode aer    Local AerSimulator with realistic noise model
  --mode ibm    Real IBM Quantum hardware (reads token from .secrets.json)

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# qgate imports  (the whole point — using our package!)
# ═══════════════════════════════════════════════════════════════════════════
from qgate import (
    ConditioningVariant,
    FilterResult,
    GateConfig,
    GroverTSVFAdapter,
    RunLogger,
    TrajectoryFilter,
)
from qgate.config import DynamicThresholdConfig, FusionConfig

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

# Target state
TARGET = "101"          # |101⟩ = decimal 5
N_SEARCH_QUBITS = 3    # 3-qubit search space
SHOTS = 8192            # per-config


# ═══════════════════════════════════════════════════════════════════════════
# Backend helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_aer_backend(noise: bool = True):
    """AerSimulator with a realistic noise model."""
    from qiskit_aer import AerSimulator
    if not noise:
        return AerSimulator()

    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        thermal_relaxation_error,
    )
    model = NoiseModel()

    t1 = 120e3
    t2 = 80e3
    gate_time_1q = 60
    gate_time_2q = 660
    gate_time_meas = 1200

    err_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
    model.add_all_qubit_quantum_error(
        err_1q, ["rx", "ry", "rz", "h", "x", "z", "s", "sdg", "u"],
    )
    dep_1q = depolarizing_error(1.5e-3, 1)
    model.add_all_qubit_quantum_error(
        dep_1q, ["rx", "ry", "rz", "h", "x", "z", "s", "sdg", "u"],
    )

    err_2q = thermal_relaxation_error(t1, t2, gate_time_2q).expand(
        thermal_relaxation_error(t1, t2, gate_time_2q),
    )
    model.add_all_qubit_quantum_error(err_2q, ["cx"])
    dep_2q = depolarizing_error(1.2e-2, 2)
    model.add_all_qubit_quantum_error(dep_2q, ["cx"])

    err_meas = thermal_relaxation_error(t1, t2, gate_time_meas)
    model.add_all_qubit_quantum_error(err_meas, ["measure"])

    return AerSimulator(noise_model=model)


def get_ibm_backend(token: str | None = None, min_qubits: int = 5):
    """Connect to IBM Quantum — least-busy backend with ≥ min_qubits."""
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
# Experiment runner — uses qgate GroverTSVFAdapter + TrajectoryFilter
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(
    backend,
    max_iterations: int = 10,
    shots: int = SHOTS,
    out_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run the full comparison experiment using the qgate pipeline.

    For each iteration count 1..max_iterations:
      1. Standard Grover via GroverTSVFAdapter(algorithm_mode="standard").
      2. TSVF Grover via GroverTSVFAdapter(algorithm_mode="tsvf")
         with Galton adaptive thresholding.
    """
    results: list[dict[str, Any]] = []

    run_logger: RunLogger | None = None
    if out_dir:
        run_logger = RunLogger(
            path=out_dir / "grover_tsvf_telemetry.jsonl",
            fmt="jsonl",
        )

    for n_iter in range(1, max_iterations + 1):
        print(f"\n{'='*60}")
        print(f"  Iterations: {n_iter}/{max_iterations}")
        print(f"{'='*60}")

        # ── Standard Grover ───────────────────────────────────────────
        std_adapter = GroverTSVFAdapter(
            backend=backend,
            algorithm_mode="standard",
            target_state=TARGET,
            optimization_level=1,
        )
        std_config = GateConfig(
            n_subsystems=N_SEARCH_QUBITS,
            n_cycles=n_iter,
            shots=shots,
            variant=ConditioningVariant.SCORE_FUSION,
            fusion=FusionConfig(alpha=0.5, threshold=0.5),
            dynamic_threshold=DynamicThresholdConfig(mode="fixed"),
            adapter="mock",
            metadata={
                "algorithm": "standard_grover",
                "iterations": n_iter,
                "target": TARGET,
            },
        )

        t0 = time.time()
        std_tf = TrajectoryFilter(std_config, std_adapter, logger=run_logger)
        std_outcomes = std_adapter.build_and_run(
            n_subsystems=N_SEARCH_QUBITS,
            n_cycles=n_iter,
            shots=shots,
        )
        std_result: FilterResult = std_tf.filter(std_outcomes)
        dt_std = time.time() - t0

        # Direct P(target) for plotting
        std_circuit = std_adapter.build_circuit(N_SEARCH_QUBITS, n_iter)
        std_raw = std_adapter.run(std_circuit, shots)
        p_std, n_std = std_adapter.extract_target_probability(
            std_raw, postselect=False,
        )
        depth_std = std_adapter.get_transpiled_depth(std_circuit)

        print(
            f"  Standard Grover:  P(|{TARGET}⟩) = {p_std:.4f}  "
            f"(depth={depth_std}, "
            f"qgate accept={std_result.acceptance_probability:.3f}, "
            f"TTS={std_result.tts:.2f}, {dt_std:.1f}s)",
        )

        # ── TSVF Chaotic Grover ──────────────────────────────────────
        tsvf_adapter = GroverTSVFAdapter(
            backend=backend,
            algorithm_mode="tsvf",
            target_state=TARGET,
            seed=42 + n_iter,
            optimization_level=1,
        )
        tsvf_config = GateConfig(
            n_subsystems=N_SEARCH_QUBITS,
            n_cycles=n_iter,
            shots=shots,
            variant=ConditioningVariant.SCORE_FUSION,
            fusion=FusionConfig(alpha=0.6, threshold=0.5),
            dynamic_threshold=DynamicThresholdConfig(
                mode="galton",
                target_acceptance=0.10,
                min_window_size=50,
                window_size=500,
                use_quantile=True,
                min_threshold=0.2,
                max_threshold=0.95,
            ),
            adapter="mock",
            metadata={
                "algorithm": "tsvf_chaotic_grover",
                "iterations": n_iter,
                "target": TARGET,
            },
        )

        t0 = time.time()
        tsvf_tf = TrajectoryFilter(
            tsvf_config, tsvf_adapter, logger=run_logger,
        )
        tsvf_outcomes = tsvf_adapter.build_and_run(
            n_subsystems=N_SEARCH_QUBITS,
            n_cycles=n_iter,
            shots=shots,
        )
        tsvf_result: FilterResult = tsvf_tf.filter(tsvf_outcomes)
        dt_tsvf = time.time() - t0

        # P(target) with post-selection
        tsvf_circuit = tsvf_adapter.build_circuit(
            N_SEARCH_QUBITS, n_iter, seed_offset=0,
        )
        tsvf_raw = tsvf_adapter.run(tsvf_circuit, shots)
        p_tsvf, n_accepted = tsvf_adapter.extract_target_probability(
            tsvf_raw, postselect=True,
        )
        depth_tsvf = tsvf_adapter.get_transpiled_depth(tsvf_circuit)
        accept_rate_raw = n_accepted / shots if shots > 0 else 0

        print(
            f"  TSVF Grover:      P(|{TARGET}⟩) = {p_tsvf:.4f}  "
            f"(depth={depth_tsvf}, "
            f"ancilla accept={n_accepted}/{shots} [{accept_rate_raw:.1%}], "
            f"qgate accept={tsvf_result.acceptance_probability:.3f}, "
            f"TTS={tsvf_result.tts:.2f}, "
            f"θ={tsvf_result.threshold_used:.3f}, {dt_tsvf:.1f}s)",
        )

        results.append({
            "iterations": n_iter,
            "p_standard": round(p_std, 6),
            "p_tsvf": round(p_tsvf, 6),
            "depth_standard": depth_std,
            "depth_tsvf": depth_tsvf,
            "shots_standard": n_std,
            "shots_tsvf_accepted": n_accepted,
            "accept_rate_tsvf": round(accept_rate_raw, 6),
            "qgate_accept_prob_std": round(
                std_result.acceptance_probability, 6,
            ),
            "qgate_accept_prob_tsvf": round(
                tsvf_result.acceptance_probability, 6,
            ),
            "qgate_tts_std": round(std_result.tts, 4),
            "qgate_tts_tsvf": round(tsvf_result.tts, 4),
            "qgate_mean_score_std": round(
                std_result.mean_combined_score or 0, 6,
            ),
            "qgate_mean_score_tsvf": round(
                tsvf_result.mean_combined_score or 0, 6,
            ),
            "qgate_threshold_tsvf": round(
                tsvf_result.threshold_used, 6,
            ),
            "qgate_dyn_threshold_tsvf": round(
                tsvf_result.dynamic_threshold_final or 0, 6,
            ),
            "time_standard_s": round(dt_std, 2),
            "time_tsvf_s": round(dt_tsvf, 2),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(results: list[dict], out_dir: Path, backend_name: str):
    """Generate patent-quality comparison plots."""
    iters = [r["iterations"] for r in results]
    p_std = [r["p_standard"] for r in results]
    p_tsvf = [r["p_tsvf"] for r in results]
    d_std = [r["depth_standard"] for r in results]
    d_tsvf = [r["depth_tsvf"] for r in results]

    patent_ref = "US 63/983,831 & 63/989,632 | IL 326915"

    # ── Plot 1: Probability of Success vs Iterations ──────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(
        iters, p_std, "ro-", linewidth=2, markersize=8,
        label="Standard Grover (baseline)",
    )
    ax1.plot(
        iters, p_tsvf, "b^-", linewidth=2, markersize=8,
        label="TSVF/Chaotic Grover (qgate post-selected)",
    )
    p_opt = math.sin(
        (2 * 1 + 1) * math.asin(1 / math.sqrt(2**N_SEARCH_QUBITS)),
    ) ** 2
    ax1.axhline(
        y=p_opt, color="green", linestyle="--", alpha=0.5,
        label=f"Theoretical optimum ({p_opt:.3f})",
    )
    ax1.set_xlabel("Number of Grover Iterations", fontsize=13)
    ax1.set_ylabel("Probability of Measuring |101⟩", fontsize=13)
    ax1.set_title(
        f"Probability of Success vs. Number of Iterations\n"
        f"3-qubit search · target |101⟩ · {backend_name} · qgate v0.5.0\n"
        f"{patent_ref}",
        fontsize=12,
    )
    ax1.legend(fontsize=11, loc="upper right")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xticks(iters)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    p1 = out_dir / "prob_vs_iterations.png"
    fig1.savefig(p1, dpi=200)
    fig1.savefig(out_dir / "prob_vs_iterations.svg")
    plt.close(fig1)
    print(f"  Saved: {p1}")

    # ── Plot 2: Circuit Depth vs Fidelity ─────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(
        d_std, p_std, "ro-", linewidth=2, markersize=8,
        label="Standard Grover",
    )
    ax2.plot(
        d_tsvf, p_tsvf, "b^-", linewidth=2, markersize=8,
        label="TSVF/Chaotic Grover (qgate post-selected)",
    )
    ax2.set_xlabel("Transpiled Circuit Depth", fontsize=13)
    ax2.set_ylabel("Fidelity (P(|101⟩) as proxy)", fontsize=13)
    ax2.set_title(
        f"Circuit Depth vs. Fidelity Under Noise\n"
        f"3-qubit search · target |101⟩ · {backend_name} · qgate v0.5.0\n"
        f"{patent_ref}",
        fontsize=12,
    )
    ax2.legend(fontsize=11, loc="upper right")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    p2 = out_dir / "depth_vs_fidelity.png"
    fig2.savefig(p2, dpi=200)
    fig2.savefig(out_dir / "depth_vs_fidelity.svg")
    plt.close(fig2)
    print(f"  Saved: {p2}")

    # ── Plot 3: Acceptance rates (ancilla vs qgate) ───────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    accept_raw = [r["accept_rate_tsvf"] for r in results]
    accept_qgate = [r["qgate_accept_prob_tsvf"] for r in results]
    x = np.arange(len(iters))
    width = 0.35
    ax3.bar(
        x - width / 2, accept_raw, width,
        color="steelblue", alpha=0.8, edgecolor="navy",
        label="Ancilla post-selection rate",
    )
    ax3.bar(
        x + width / 2, accept_qgate, width,
        color="coral", alpha=0.8, edgecolor="darkred",
        label="qgate score-fusion accept rate",
    )
    ax3.set_xlabel("Number of Grover Iterations", fontsize=13)
    ax3.set_ylabel("Acceptance Rate", fontsize=13)
    ax3.set_title(
        f"TSVF Post-Selection & qgate Acceptance Rates\n"
        f"{backend_name} · {patent_ref}",
        fontsize=12,
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels(iters)
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis="y")
    fig3.tight_layout()
    p3 = out_dir / "acceptance_rate.png"
    fig3.savefig(p3, dpi=200)
    plt.close(fig3)
    print(f"  Saved: {p3}")

    # ── Plot 4: Galton threshold adaptation ───────────────────────────
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    thresholds = [r["qgate_dyn_threshold_tsvf"] for r in results]
    scores_tsvf = [r["qgate_mean_score_tsvf"] for r in results]
    ax4.plot(
        iters, thresholds, "g^-", linewidth=2, markersize=8,
        label="Galton adaptive threshold (θ)",
    )
    ax4.plot(
        iters, scores_tsvf, "b.-", linewidth=2, markersize=8,
        label="Mean combined score (TSVF)",
    )
    ax4.set_xlabel("Number of Grover Iterations", fontsize=13)
    ax4.set_ylabel("Score / Threshold", fontsize=13)
    ax4.set_title(
        f"Galton Adaptive Threshold vs. Mean Score\n"
        f"{backend_name} · {patent_ref}",
        fontsize=12,
    )
    ax4.legend(fontsize=10)
    ax4.set_xticks(iters)
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, alpha=0.3)
    fig4.tight_layout()
    p4 = out_dir / "galton_threshold.png"
    fig4.savefig(p4, dpi=200)
    plt.close(fig4)
    print(f"  Saved: {p4}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Grover vs TSVF-Chaotic Grover comparison experiment — "
            "powered by the qgate trajectory filter package."
        ),
    )
    p.add_argument(
        "--mode", choices=["aer", "ibm"], default="aer",
        help="Backend: aer (noisy sim) or ibm (real hardware)",
    )
    p.add_argument(
        "--token", type=str, default=None,
        help="IBM Quantum token (fallback to .secrets.json)",
    )
    p.add_argument(
        "--max-iter", type=int, default=10,
        help="Maximum Grover iterations (default 10)",
    )
    p.add_argument(
        "--shots", type=int, default=SHOTS,
        help=f"Shots per circuit (default {SHOTS})",
    )
    p.add_argument(
        "--no-noise", action="store_true",
        help="Disable noise model (aer only)",
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: auto)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = SCRIPT_DIR / f"results_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("GROVER vs TSVF-CHAOTIC GROVER — qgate TRAJECTORY FILTER EXPERIMENT")
    print("Package: qgate v0.5.0  (GroverTSVFAdapter + Galton thresholding)")
    print("Patent ref: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915")
    print("=" * 72)

    if args.mode == "ibm":
        backend = get_ibm_backend(token=args.token, min_qubits=5)
        backend_name = f"{backend.name}"
    else:
        backend = get_aer_backend(noise=not args.no_noise)
        backend_name = (
            "AerSimulator (noisy)" if not args.no_noise
            else "AerSimulator (ideal)"
        )

    print(f"Backend:     {backend_name}")
    print(f"Target:      |{TARGET}⟩")
    print(f"Iterations:  1 – {args.max_iter}")
    print(f"Shots:       {args.shots}")
    print(f"Output:      {out_dir}")
    print()

    t_start = time.time()

    results = run_experiment(
        backend,
        max_iterations=args.max_iter,
        shots=args.shots,
        out_dir=out_dir,
    )

    elapsed = time.time() - t_start

    # Save CSV
    csv_path = out_dir / "grover_tsvf_results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV saved: {csv_path}")

    # Save run metadata
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": backend_name,
        "mode": args.mode,
        "target_state": TARGET,
        "n_search_qubits": N_SEARCH_QUBITS,
        "max_iterations": args.max_iter,
        "shots": args.shots,
        "elapsed_seconds": round(elapsed, 1),
        "qgate_version": "0.5.0",
        "adapter": "GroverTSVFAdapter",
        "patent_ref": "US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915",
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(results, out_dir, backend_name)

    # Print summary table
    print(f"\n{'='*72}")
    print("SUMMARY TABLE")
    print(f"{'='*72}")
    header = (
        f"{'Iter':>4} {'P_std':>8} {'P_tsvf':>8} {'D_std':>6} "
        f"{'D_tsvf':>7} {'Accept%':>8} {'qgateA%':>8} {'TTS':>7} {'θ':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['iterations']:>4} {r['p_standard']:>8.4f} "
            f"{r['p_tsvf']:>8.4f} {r['depth_standard']:>6} "
            f"{r['depth_tsvf']:>7} {r['accept_rate_tsvf']:>8.1%} "
            f"{r['qgate_accept_prob_tsvf']:>8.1%} "
            f"{r['qgate_tts_tsvf']:>7.2f} "
            f"{r['qgate_threshold_tsvf']:>6.3f}"
        )

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results:    {out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
