#!/usr/bin/env python3
"""
run_qpe_tsvf_experiment.py — QPE vs TSVF-QPE phase estimation comparison.

**Powered by the qgate package** (``pip install -e packages/qgate[qiskit]``).

Compares standard Quantum Phase Estimation with a time-symmetric /
chaotic-ansatz post-selected variant across increasing precision qubit
count (3–7), demonstrating that TSVF post-selection "anchors" the phase
—  keeping a sharp probability spike on the correct phase binary fraction
despite hardware noise.

Uses qgate's:
  • ``QPETSVFAdapter``    — builds & runs QPE circuits, maps to ParityOutcome.
  • ``TrajectoryFilter``  — scoring, Galton adaptive thresholding, conditioning.
  • ``GateConfig``        — declarative experiment configuration.
  • ``RunLogger``         — structured JSONL telemetry.
  • ``FilterResult``      — acceptance statistics & TTS.

Target unitary: U = diag(1, e^{2πiφ})  with eigenphase φ = 1/3
  (irrational in binary → good stress-test for precision)

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
    QPETSVFAdapter,
    RunLogger,
    TrajectoryFilter,
)
from qgate.adapters.qpe_adapter import (
    binary_fraction_to_phase,
    histogram_entropy,
    mean_phase_error,
    phase_error,
    phase_fidelity,
    phase_to_binary_fraction,
)
from qgate.config import DynamicThresholdConfig, FusionConfig

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

# Problem definition
EIGENPHASE = 1.0 / 3.0    # φ = 1/3 (irrational in binary)
SHOTS = 8192               # per-config


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


def get_ibm_backend(token: str | None = None, min_qubits: int = 10):
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
# Experiment runner — uses qgate QPETSVFAdapter + TrajectoryFilter
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(
    backend,
    eigenphase: float = EIGENPHASE,
    min_precision: int = 3,
    max_precision: int = 7,
    shots: int = SHOTS,
    out_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run the full QPE comparison experiment using the qgate pipeline.

    For each precision qubit count t = min_precision..max_precision:
      1. Standard QPE via QPETSVFAdapter(algorithm_mode="standard").
      2. TSVF QPE via QPETSVFAdapter(algorithm_mode="tsvf")
         with Galton adaptive thresholding.
    """
    results: list[dict[str, Any]] = []

    run_logger: RunLogger | None = None
    if out_dir:
        run_logger = RunLogger(
            path=out_dir / "qpe_tsvf_telemetry.jsonl",
            fmt="jsonl",
        )

    for t in range(min_precision, max_precision + 1):
        correct_bits = phase_to_binary_fraction(eigenphase, t)
        correct_phase = binary_fraction_to_phase(correct_bits)
        print(f"\n{'='*60}")
        print(f"  QPE Precision Qubits: t={t}/{max_precision}")
        print(f"  True phase: φ = {eigenphase:.6f}")
        print(f"  Best {t}-bit approx: 0.{correct_bits}₂ = {correct_phase:.6f}")
        print(f"  Ideal phase error: {phase_error(correct_phase, eigenphase):.6f}")
        print(f"{'='*60}")

        # ── Standard QPE ─────────────────────────────────────────────
        std_adapter = QPETSVFAdapter(
            backend=backend,
            algorithm_mode="standard",
            eigenphase=eigenphase,
            seed=42,
            optimization_level=1,
        )
        std_config = GateConfig(
            n_subsystems=t,
            n_cycles=1,
            shots=shots,
            variant=ConditioningVariant.SCORE_FUSION,
            fusion=FusionConfig(alpha=0.5, threshold=0.5),
            dynamic_threshold=DynamicThresholdConfig(mode="fixed"),
            adapter="mock",
            metadata={
                "experiment": "qpe_tsvf",
                "algorithm": "standard",
                "precision": t,
                "eigenphase": eigenphase,
            },
        )
        std_tf = TrajectoryFilter(std_config, std_adapter)

        print(f"\n  [Standard QPE] Building circuit (t={t})...")
        std_qc = std_adapter.build_circuit(n_subsystems=t, n_cycles=1)
        std_depth = std_adapter.get_transpiled_depth(std_qc)
        print(f"  [Standard QPE] Transpiled depth: {std_depth}")

        print(f"  [Standard QPE] Running {shots} shots...")
        t0 = time.time()
        std_raw = std_adapter.run(std_qc, shots=shots)
        std_time = time.time() - t0
        print(f"  [Standard QPE] Done in {std_time:.1f}s")

        std_metrics = std_adapter.extract_phase_metrics(
            std_raw, t, postselect=False,
        )
        std_best_bs, std_best_phase, std_best_count = (
            std_adapter.extract_best_phase(std_raw, t, postselect=False)
        )
        std_filter = std_tf.run()
        if run_logger:
            run_logger.log(std_filter)

        print(f"  [Standard QPE] Fidelity: {std_metrics['fidelity']:.4f}")
        print(f"  [Standard QPE] Mean phase error: {std_metrics['mean_phase_error']:.4f}")
        print(f"  [Standard QPE] Entropy: {std_metrics['entropy']:.4f} bits")
        print(f"  [Standard QPE] Best phase: 0.{std_best_bs}₂ = {std_best_phase:.6f}")

        # ── TSVF QPE ─────────────────────────────────────────────────
        tsvf_adapter = QPETSVFAdapter(
            backend=backend,
            algorithm_mode="tsvf",
            eigenphase=eigenphase,
            seed=42,
            weak_angle_base=math.pi / 4,
            weak_angle_ramp=math.pi / 8,
            optimization_level=1,
        )
        tsvf_config = GateConfig(
            n_subsystems=t,
            n_cycles=1,
            shots=shots,
            variant=ConditioningVariant.SCORE_FUSION,
            fusion=FusionConfig(alpha=0.5, threshold=0.5),
            dynamic_threshold=DynamicThresholdConfig(
                mode="galton",
                target_acceptance=0.6,
                window_size=200,
            ),
            adapter="mock",
            metadata={
                "experiment": "qpe_tsvf",
                "algorithm": "tsvf",
                "precision": t,
                "eigenphase": eigenphase,
            },
        )
        tsvf_tf = TrajectoryFilter(tsvf_config, tsvf_adapter)

        print(f"\n  [TSVF QPE] Building circuit (t={t})...")
        tsvf_qc = tsvf_adapter.build_circuit(n_subsystems=t, n_cycles=1)
        tsvf_depth = tsvf_adapter.get_transpiled_depth(tsvf_qc)
        print(f"  [TSVF QPE] Transpiled depth: {tsvf_depth}")

        print(f"  [TSVF QPE] Running {shots} shots...")
        t0 = time.time()
        tsvf_raw = tsvf_adapter.run(tsvf_qc, shots=shots)
        tsvf_time = time.time() - t0
        print(f"  [TSVF QPE] Done in {tsvf_time:.1f}s")

        tsvf_metrics = tsvf_adapter.extract_phase_metrics(
            tsvf_raw, t, postselect=True,
        )
        tsvf_best_bs, tsvf_best_phase, tsvf_best_count = (
            tsvf_adapter.extract_best_phase(tsvf_raw, t, postselect=True)
        )
        tsvf_filter = tsvf_tf.run()
        if run_logger:
            run_logger.log(tsvf_filter)

        print(f"  [TSVF QPE] Fidelity: {tsvf_metrics['fidelity']:.4f}")
        print(f"  [TSVF QPE] Mean phase error: {tsvf_metrics['mean_phase_error']:.4f}")
        print(f"  [TSVF QPE] Entropy: {tsvf_metrics['entropy']:.4f} bits")
        print(f"  [TSVF QPE] Acceptance rate: {tsvf_metrics['acceptance_rate']:.4f}")
        print(f"  [TSVF QPE] Best phase: 0.{tsvf_best_bs}₂ = {tsvf_best_phase:.6f}")

        # ── Record ────────────────────────────────────────────────────
        row = {
            "precision": t,
            "eigenphase": eigenphase,
            "correct_bits": correct_bits,
            # Standard
            "std_fidelity": std_metrics["fidelity"],
            "std_mean_error": std_metrics["mean_phase_error"],
            "std_entropy": std_metrics["entropy"],
            "std_measured_phase": std_metrics["measured_phase"],
            "std_best_bs": std_best_bs,
            "std_best_phase": std_best_phase,
            "std_depth": std_depth,
            "std_time": std_time,
            # TSVF
            "tsvf_fidelity": tsvf_metrics["fidelity"],
            "tsvf_mean_error": tsvf_metrics["mean_phase_error"],
            "tsvf_entropy": tsvf_metrics["entropy"],
            "tsvf_measured_phase": tsvf_metrics["measured_phase"],
            "tsvf_best_bs": tsvf_best_bs,
            "tsvf_best_phase": tsvf_best_phase,
            "tsvf_depth": tsvf_depth,
            "tsvf_acceptance": tsvf_metrics["acceptance_rate"],
            "tsvf_time": tsvf_time,
            # Deltas
            "fidelity_delta": tsvf_metrics["fidelity"] - std_metrics["fidelity"],
            "error_delta": std_metrics["mean_phase_error"] - tsvf_metrics["mean_phase_error"],
            "entropy_delta": std_metrics["entropy"] - tsvf_metrics["entropy"],
        }
        results.append(row)

        print(f"\n  ── Summary t={t} ──")
        print(f"  Fidelity:  std={std_metrics['fidelity']:.4f}  "
              f"tsvf={tsvf_metrics['fidelity']:.4f}  "
              f"Δ={row['fidelity_delta']:+.4f}")
        print(f"  Mean err:  std={std_metrics['mean_phase_error']:.4f}  "
              f"tsvf={tsvf_metrics['mean_phase_error']:.4f}  "
              f"Δ={row['error_delta']:+.4f}")
        print(f"  Entropy:   std={std_metrics['entropy']:.4f}  "
              f"tsvf={tsvf_metrics['entropy']:.4f}  "
              f"Δ={row['entropy_delta']:+.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Output — CSV, plots, metadata
# ═══════════════════════════════════════════════════════════════════════════

def save_results_csv(results: list[dict], out_dir: Path) -> Path:
    """Write results to CSV."""
    path = out_dir / "qpe_tsvf_results.csv"
    fields = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  CSV saved to {path}")
    return path


def plot_fidelity_vs_precision(results: list[dict], out_dir: Path):
    """Plot phase fidelity vs precision qubits."""
    prec = [r["precision"] for r in results]
    std_fid = [r["std_fidelity"] for r in results]
    tsvf_fid = [r["tsvf_fidelity"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(prec, std_fid, "o-", color="royalblue", linewidth=2,
            markersize=8, label="Standard QPE")
    ax.plot(prec, tsvf_fid, "s-", color="crimson", linewidth=2,
            markersize=8, label="TSVF QPE")
    ax.set_xlabel("Precision Qubits (t)", fontsize=12)
    ax.set_ylabel("Phase Fidelity P(correct)", fontsize=12)
    ax.set_title("QPE Phase Fidelity: Standard vs TSVF", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(prec)
    ax.set_ylim(-0.05, 1.05)

    for fmt in ("png", "svg"):
        fig.savefig(out_dir / f"fidelity_vs_precision.{fmt}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_phase_error_vs_precision(results: list[dict], out_dir: Path):
    """Plot mean phase error vs precision qubits."""
    prec = [r["precision"] for r in results]
    std_err = [r["std_mean_error"] for r in results]
    tsvf_err = [r["tsvf_mean_error"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(prec, std_err, "o-", color="royalblue", linewidth=2,
            markersize=8, label="Standard QPE")
    ax.plot(prec, tsvf_err, "s-", color="crimson", linewidth=2,
            markersize=8, label="TSVF QPE")
    ax.set_xlabel("Precision Qubits (t)", fontsize=12)
    ax.set_ylabel("Mean Phase Error", fontsize=12)
    ax.set_title("QPE Phase Error: Standard vs TSVF", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(prec)

    for fmt in ("png", "svg"):
        fig.savefig(out_dir / f"phase_error_vs_precision.{fmt}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_vs_precision(results: list[dict], out_dir: Path):
    """Plot histogram entropy vs precision qubits.

    Lower entropy → sharper distribution → more confident phase estimate.
    The dashed line shows the maximum entropy for uniform random (t bits).
    """
    prec = [r["precision"] for r in results]
    std_ent = [r["std_entropy"] for r in results]
    tsvf_ent = [r["tsvf_entropy"] for r in results]
    max_ent = [float(t) for t in prec]  # max entropy = t bits

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(prec, std_ent, "o-", color="royalblue", linewidth=2,
            markersize=8, label="Standard QPE")
    ax.plot(prec, tsvf_ent, "s-", color="crimson", linewidth=2,
            markersize=8, label="TSVF QPE")
    ax.plot(prec, max_ent, "--", color="gray", linewidth=1.5,
            label="Max entropy (uniform)")
    ax.set_xlabel("Precision Qubits (t)", fontsize=12)
    ax.set_ylabel("Shannon Entropy (bits)", fontsize=12)
    ax.set_title("QPE Histogram Sharpness: Standard vs TSVF", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(prec)

    for fmt in ("png", "svg"):
        fig.savefig(out_dir / f"entropy_vs_precision.{fmt}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_depth_vs_precision(results: list[dict], out_dir: Path):
    """Plot circuit depth vs precision qubits."""
    prec = [r["precision"] for r in results]
    std_d = [r["std_depth"] for r in results]
    tsvf_d = [r["tsvf_depth"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(prec, std_d, "o-", color="royalblue", linewidth=2,
            markersize=8, label="Standard QPE")
    ax.plot(prec, tsvf_d, "s-", color="crimson", linewidth=2,
            markersize=8, label="TSVF QPE")
    ax.set_xlabel("Precision Qubits (t)", fontsize=12)
    ax.set_ylabel("Transpiled Circuit Depth", fontsize=12)
    ax.set_title("QPE Circuit Depth: Standard vs TSVF", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(prec)

    for fmt in ("png", "svg"):
        fig.savefig(out_dir / f"depth_vs_precision.{fmt}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_acceptance_rate(results: list[dict], out_dir: Path):
    """Plot TSVF acceptance rate vs precision qubits."""
    prec = [r["precision"] for r in results]
    acc = [r["tsvf_acceptance"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(prec, acc, color="crimson", alpha=0.7, width=0.6)
    ax.set_xlabel("Precision Qubits (t)", fontsize=12)
    ax.set_ylabel("TSVF Acceptance Rate", fontsize=12)
    ax.set_title("TSVF Post-Selection Acceptance Rate", fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_xticks(prec)
    ax.grid(True, alpha=0.3, axis="y")

    for fmt in ("png", "svg"):
        fig.savefig(out_dir / f"acceptance_rate.{fmt}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_fidelity_ratio(results: list[dict], out_dir: Path):
    """Plot TSVF/Standard fidelity ratio vs precision qubits."""
    prec = [r["precision"] for r in results]
    ratio = []
    for r in results:
        if r["std_fidelity"] > 0:
            ratio.append(r["tsvf_fidelity"] / r["std_fidelity"])
        else:
            ratio.append(float("inf") if r["tsvf_fidelity"] > 0 else 1.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["forestgreen" if r >= 1.0 else "orange" for r in ratio]
    ax.bar(prec, ratio, color=colors, alpha=0.7, width=0.6)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5,
               label="Break-even")
    ax.set_xlabel("Precision Qubits (t)", fontsize=12)
    ax.set_ylabel("Fidelity Ratio (TSVF / Standard)", fontsize=12)
    ax.set_title("QPE Phase Anchoring: TSVF Advantage", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(prec)

    for fmt in ("png", "svg"):
        fig.savefig(out_dir / f"fidelity_ratio_vs_precision.{fmt}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metadata(
    results: list[dict],
    out_dir: Path,
    mode: str,
    backend_name: str,
    eigenphase: float = EIGENPHASE,
    shots: int = SHOTS,
):
    """Save experiment metadata JSON."""
    meta = {
        "experiment": "qpe_tsvf",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "backend": backend_name,
        "eigenphase": eigenphase,
        "shots": shots,
        "precision_range": [
            results[0]["precision"], results[-1]["precision"],
        ],
        "num_configs": len(results),
        "patent_refs": [
            "US App. No. 63/983,831 (Feb 16, 2026)",
            "US App. No. 63/989,632 (Feb 24, 2026)",
            "IL App. No. 326915",
        ],
    }
    path = out_dir / "experiment_metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to {path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="QPE vs TSVF-QPE phase estimation experiment",
    )
    parser.add_argument(
        "--mode", choices=["aer", "ibm"], default="aer",
        help="Backend mode (default: aer)",
    )
    parser.add_argument(
        "--min-precision", type=int, default=3,
        help="Minimum number of precision qubits (default: 3)",
    )
    parser.add_argument(
        "--max-precision", type=int, default=7,
        help="Maximum number of precision qubits (default: 7)",
    )
    parser.add_argument(
        "--shots", type=int, default=SHOTS,
        help=f"Shots per configuration (default: {SHOTS})",
    )
    parser.add_argument(
        "--eigenphase", type=float, default=EIGENPHASE,
        help=f"Target eigenphase φ ∈ [0,1) (default: {EIGENPHASE:.6f})",
    )
    parser.add_argument(
        "--no-noise", action="store_true",
        help="Disable noise model for Aer (ideal simulator)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Custom output directory (default: auto-timestamped)",
    )
    args = parser.parse_args()

    # ── Output dir ────────────────────────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = SCRIPT_DIR / f"results_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Backend ───────────────────────────────────────────────────────
    if args.mode == "aer":
        noise = not args.no_noise
        backend = get_aer_backend(noise=noise)
        backend_name = f"AerSimulator (noise={'on' if noise else 'off'})"
    else:
        backend = get_ibm_backend(min_qubits=args.max_precision + 2)
        backend_name = backend.name

    print(f"\n{'─'*60}")
    print(f"  QPE vs TSVF-QPE Phase Estimation Experiment")
    print(f"  Backend:    {backend_name}")
    print(f"  Eigenphase: φ = {args.eigenphase:.6f}")
    print(f"  Precision:  {args.min_precision} → {args.max_precision} qubits")
    print(f"  Shots:      {args.shots}")
    print(f"  Output:     {out_dir}")
    print(f"{'─'*60}")

    # ── Run ────────────────────────────────────────────────────────────
    results = run_experiment(
        backend,
        eigenphase=args.eigenphase,
        min_precision=args.min_precision,
        max_precision=args.max_precision,
        shots=args.shots,
        out_dir=out_dir,
    )

    # ── Save ───────────────────────────────────────────────────────────
    save_results_csv(results, out_dir)
    save_metadata(
        results, out_dir, args.mode, backend_name,
        eigenphase=args.eigenphase, shots=args.shots,
    )

    # ── Plots ──────────────────────────────────────────────────────────
    print("\n  Generating plots...")
    plot_fidelity_vs_precision(results, out_dir)
    plot_phase_error_vs_precision(results, out_dir)
    plot_entropy_vs_precision(results, out_dir)
    plot_depth_vs_precision(results, out_dir)
    plot_acceptance_rate(results, out_dir)
    plot_fidelity_ratio(results, out_dir)
    print(f"  6 plots saved to {out_dir}/")

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  RESULTS SUMMARY — QPE vs TSVF-QPE  (φ = {args.eigenphase:.6f})")
    print(f"{'='*90}")
    hdr = (
        f"{'t':>3} | {'Fid(std)':>8} {'Fid(tsvf)':>9} | "
        f"{'Err(std)':>8} {'Err(tsvf)':>9} | "
        f"{'Ent(std)':>8} {'Ent(tsvf)':>9} | "
        f"{'D(std)':>6} {'D(tsvf)':>7} | {'Accept%':>7} | "
        f"{'Best(std)':>10} {'Best(tsvf)':>11}"
    )
    print(hdr)
    print("─" * len(hdr))
    for r in results:
        print(
            f"{r['precision']:3d} | "
            f"{r['std_fidelity']:8.4f} {r['tsvf_fidelity']:9.4f} | "
            f"{r['std_mean_error']:8.4f} {r['tsvf_mean_error']:9.4f} | "
            f"{r['std_entropy']:8.4f} {r['tsvf_entropy']:9.4f} | "
            f"{r['std_depth']:6d} {r['tsvf_depth']:7d} | "
            f"{r['tsvf_acceptance']*100:6.1f}% | "
            f"0.{r['std_best_bs']:<9s} 0.{r['tsvf_best_bs']:<10s}"
        )
    print(f"{'='*90}")
    print(f"\n  ✅ Experiment complete. Results in: {out_dir}")


if __name__ == "__main__":
    main()
