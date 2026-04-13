#!/usr/bin/env python3
"""
run_vqe_tsvf_experiment.py — VQE vs TSVF-VQE ground-state energy comparison.

**Powered by the qgate package** (``pip install -e packages/qgate[qiskit]``).

Compares standard VQE for the Transverse-Field Ising Model (TFIM) with a
time-symmetric / chaotic-ansatz post-selected variant across increasing
circuit depth (1–L ansatz layers), demonstrating that post-selection
stabilises the energy estimate against decoherence on NISQ hardware.

Uses qgate's:
  • ``VQETSVFAdapter``    — builds & runs VQE circuits, maps to ParityOutcome.
  • ``TrajectoryFilter``  — scoring, Galton adaptive thresholding, conditioning.
  • ``GateConfig``        — declarative experiment configuration.
  • ``RunLogger``         — structured JSONL telemetry.
  • ``FilterResult``      — acceptance statistics & TTS.

Problem: Ground-state energy of 1D TFIM
  H = −J Σ Z_i Z_{i+1}  −  h Σ X_i

Modes:
  --mode aer    Local AerSimulator with realistic noise model
  --mode ibm    Real IBM Quantum hardware (reads token from .secrets.json)

Patent pending (see LICENSE)
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
    RunLogger,
    TrajectoryFilter,
    VQETSVFAdapter,
)
from qgate.adapters.vqe_adapter import (
    energy_error,
    energy_ratio,
    tfim_exact_ground_energy,
)
from qgate.config import DynamicThresholdConfig, FusionConfig

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

# Problem definition
N_QUBITS = 4          # Number of qubits
J_COUPLING = 1.0      # ZZ coupling strength
H_FIELD = 1.0         # Transverse field strength
SHOTS = 8192          # per-config


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
# Experiment runner — uses qgate VQETSVFAdapter + TrajectoryFilter
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(
    backend,
    exact_gs_energy: float,
    max_layers: int = 8,
    shots: int = SHOTS,
    out_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run the full VQE comparison experiment using the qgate pipeline.

    For each layer count L = 1..max_layers:
      1. Standard VQE via VQETSVFAdapter(algorithm_mode="standard").
      2. TSVF VQE via VQETSVFAdapter(algorithm_mode="tsvf")
         with Galton adaptive thresholding.
    """
    results: list[dict[str, Any]] = []

    run_logger: RunLogger | None = None
    if out_dir:
        run_logger = RunLogger(
            path=out_dir / "vqe_tsvf_telemetry.jsonl",
            fmt="jsonl",
        )

    for L in range(1, max_layers + 1):
        print(f"\n{'='*60}")
        print(f"  VQE Ansatz Layers: L={L}/{max_layers}")
        print(f"{'='*60}")

        # ── Standard VQE ──────────────────────────────────────────────
        std_adapter = VQETSVFAdapter(
            backend=backend,
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
            j_coupling=J_COUPLING,
            h_field=H_FIELD,
            seed=42,
            optimization_level=1,
        )
        std_config = GateConfig(
            n_subsystems=N_QUBITS,
            n_cycles=L,
            shots=shots,
            variant=ConditioningVariant.SCORE_FUSION,
            fusion=FusionConfig(alpha=0.5, threshold=0.5),
            dynamic_threshold=DynamicThresholdConfig(mode="fixed"),
            adapter="mock",
            metadata={
                "algorithm": "standard_vqe",
                "layers": L,
                "n_qubits": N_QUBITS,
                "j_coupling": J_COUPLING,
                "h_field": H_FIELD,
            },
        )

        t0 = time.time()
        std_tf = TrajectoryFilter(std_config, std_adapter, logger=run_logger)
        std_outcomes = std_adapter.build_and_run(
            n_subsystems=N_QUBITS, n_cycles=L, shots=shots,
        )
        std_result: FilterResult = std_tf.filter(std_outcomes)
        dt_std = time.time() - t0

        # Direct energy extraction
        std_circuit = std_adapter.build_circuit(N_QUBITS, L)
        std_raw = std_adapter.run(std_circuit, shots)
        std_energy, n_std = std_adapter.extract_energy(std_raw, postselect=False)
        std_ratio = energy_ratio(std_energy, exact_gs_energy)
        std_err = energy_error(std_energy, exact_gs_energy)
        depth_std = std_adapter.get_transpiled_depth(std_circuit)
        best_bs_std, best_e_std, _ = std_adapter.extract_best_bitstring(
            std_raw, postselect=False,
        )

        print(
            f"  Standard VQE:  E={std_energy:.4f}  ratio={std_ratio:.4f}  "
            f"err={std_err:.4f}  "
            f"(depth={depth_std}, "
            f"best={best_bs_std}[E={best_e_std:.2f}], "
            f"qgate accept={std_result.acceptance_probability:.3f}, "
            f"TTS={std_result.tts:.2f}, {dt_std:.1f}s)",
        )

        # ── TSVF VQE ─────────────────────────────────────────────────
        tsvf_adapter = VQETSVFAdapter(
            backend=backend,
            algorithm_mode="tsvf",
            n_qubits=N_QUBITS,
            j_coupling=J_COUPLING,
            h_field=H_FIELD,
            seed=42 + L,
            optimization_level=1,
        )
        tsvf_config = GateConfig(
            n_subsystems=N_QUBITS,
            n_cycles=L,
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
                "algorithm": "tsvf_vqe",
                "layers": L,
                "n_qubits": N_QUBITS,
                "j_coupling": J_COUPLING,
                "h_field": H_FIELD,
            },
        )

        t0 = time.time()
        tsvf_tf = TrajectoryFilter(
            tsvf_config, tsvf_adapter, logger=run_logger,
        )
        tsvf_outcomes = tsvf_adapter.build_and_run(
            n_subsystems=N_QUBITS, n_cycles=L, shots=shots,
        )
        tsvf_result: FilterResult = tsvf_tf.filter(tsvf_outcomes)
        dt_tsvf = time.time() - t0

        # Energy with post-selection
        tsvf_circuit = tsvf_adapter.build_circuit(N_QUBITS, L, seed_offset=0)
        tsvf_raw = tsvf_adapter.run(tsvf_circuit, shots)
        tsvf_energy, n_accepted = tsvf_adapter.extract_energy(
            tsvf_raw, postselect=True,
        )
        tsvf_ratio = energy_ratio(tsvf_energy, exact_gs_energy)
        tsvf_err = energy_error(tsvf_energy, exact_gs_energy)
        depth_tsvf = tsvf_adapter.get_transpiled_depth(tsvf_circuit)
        accept_rate_raw = n_accepted / shots if shots > 0 else 0
        best_bs_tsvf, best_e_tsvf, _ = tsvf_adapter.extract_best_bitstring(
            tsvf_raw, postselect=True,
        )

        print(
            f"  TSVF VQE:      E={tsvf_energy:.4f}  ratio={tsvf_ratio:.4f}  "
            f"err={tsvf_err:.4f}  "
            f"(depth={depth_tsvf}, "
            f"best={best_bs_tsvf}[E={best_e_tsvf:.2f}], "
            f"ancilla accept={n_accepted}/{shots} [{accept_rate_raw:.1%}], "
            f"qgate accept={tsvf_result.acceptance_probability:.3f}, "
            f"TTS={tsvf_result.tts:.2f}, "
            f"θ={tsvf_result.threshold_used:.3f}, {dt_tsvf:.1f}s)",
        )

        results.append({
            "layers": L,
            "energy_std": round(std_energy, 6),
            "energy_tsvf": round(tsvf_energy, 6),
            "energy_ratio_std": round(std_ratio, 6),
            "energy_ratio_tsvf": round(tsvf_ratio, 6),
            "energy_error_std": round(std_err, 6),
            "energy_error_tsvf": round(tsvf_err, 6),
            "depth_standard": depth_std,
            "depth_tsvf": depth_tsvf,
            "best_energy_std": round(best_e_std, 4),
            "best_energy_tsvf": round(best_e_tsvf, 4),
            "best_bs_std": best_bs_std,
            "best_bs_tsvf": best_bs_tsvf,
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

def generate_plots(
    results: list[dict],
    out_dir: Path,
    backend_name: str,
    exact_gs_energy: float,
):
    """Generate patent-quality VQE comparison plots."""
    layers = [r["layers"] for r in results]
    e_std = [r["energy_std"] for r in results]
    e_tsvf = [r["energy_tsvf"] for r in results]
    err_std = [r["energy_error_std"] for r in results]
    err_tsvf = [r["energy_error_tsvf"] for r in results]
    ratio_std = [r["energy_ratio_std"] for r in results]
    ratio_tsvf = [r["energy_ratio_tsvf"] for r in results]
    d_std = [r["depth_standard"] for r in results]
    d_tsvf = [r["depth_tsvf"] for r in results]

    patent_ref = "Patent pending"

    # ── Plot 1: Energy vs Layers ──────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(
        layers, e_std, "ro-", linewidth=2, markersize=8,
        label="Standard VQE (baseline)",
    )
    ax1.plot(
        layers, e_tsvf, "b^-", linewidth=2, markersize=8,
        label="TSVF VQE (qgate post-selected)",
    )
    ax1.axhline(
        y=exact_gs_energy, color="green", linestyle="--", alpha=0.7,
        linewidth=2, label=f"Exact GS energy ({exact_gs_energy:.4f})",
    )
    ax1.set_xlabel("Number of Ansatz Layers (L)", fontsize=13)
    ax1.set_ylabel("Estimated Energy ⟨H⟩", fontsize=13)
    ax1.set_title(
        f"VQE Energy Estimate vs. Ansatz Depth\n"
        f"{N_QUBITS}-qubit TFIM · J={J_COUPLING} · h={H_FIELD} · "
        f"{backend_name} · qgate v0.5.0\n{patent_ref}",
        fontsize=11,
    )
    ax1.legend(fontsize=11, loc="upper right")
    ax1.set_xticks(layers)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    p1 = out_dir / "energy_vs_layers.png"
    fig1.savefig(p1, dpi=200)
    fig1.savefig(out_dir / "energy_vs_layers.svg")
    plt.close(fig1)
    print(f"  Saved: {p1}")

    # ── Plot 2: Energy Error vs Layers ────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(
        layers, err_std, "ro-", linewidth=2, markersize=8,
        label="Standard VQE",
    )
    ax2.plot(
        layers, err_tsvf, "b^-", linewidth=2, markersize=8,
        label="TSVF VQE (qgate post-selected)",
    )
    ax2.axhline(
        y=0.0, color="green", linestyle="--", alpha=0.5,
        label="Zero error (exact)",
    )
    ax2.set_xlabel("Number of Ansatz Layers (L)", fontsize=13)
    ax2.set_ylabel("Absolute Energy Error |E_est − E_exact|", fontsize=13)
    ax2.set_title(
        f"VQE Energy Error vs. Ansatz Depth\n"
        f"{N_QUBITS}-qubit TFIM · {backend_name} · qgate v0.5.0\n{patent_ref}",
        fontsize=11,
    )
    ax2.legend(fontsize=11, loc="upper right")
    ax2.set_xticks(layers)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    p2 = out_dir / "energy_error_vs_layers.png"
    fig2.savefig(p2, dpi=200)
    fig2.savefig(out_dir / "energy_error_vs_layers.svg")
    plt.close(fig2)
    print(f"  Saved: {p2}")

    # ── Plot 3: Circuit Depth vs Energy ───────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(
        d_std, e_std, "ro-", linewidth=2, markersize=8,
        label="Standard VQE",
    )
    ax3.plot(
        d_tsvf, e_tsvf, "b^-", linewidth=2, markersize=8,
        label="TSVF VQE (qgate post-selected)",
    )
    ax3.axhline(
        y=exact_gs_energy, color="green", linestyle="--", alpha=0.7,
        linewidth=2, label=f"Exact GS ({exact_gs_energy:.4f})",
    )
    ax3.set_xlabel("Transpiled Circuit Depth", fontsize=13)
    ax3.set_ylabel("Estimated Energy ⟨H⟩", fontsize=13)
    ax3.set_title(
        f"Circuit Depth vs. Energy Under Noise\n"
        f"{N_QUBITS}-qubit TFIM · {backend_name} · qgate v0.5.0\n{patent_ref}",
        fontsize=11,
    )
    ax3.legend(fontsize=11, loc="upper right")
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    p3 = out_dir / "depth_vs_energy.png"
    fig3.savefig(p3, dpi=200)
    fig3.savefig(out_dir / "depth_vs_energy.svg")
    plt.close(fig3)
    print(f"  Saved: {p3}")

    # ── Plot 4: Acceptance rates ──────────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    accept_raw = [r["accept_rate_tsvf"] for r in results]
    accept_qgate = [r["qgate_accept_prob_tsvf"] for r in results]
    x = np.arange(len(layers))
    width = 0.35
    ax4.bar(
        x - width / 2, accept_raw, width,
        color="steelblue", alpha=0.8, edgecolor="navy",
        label="Ancilla post-selection rate",
    )
    ax4.bar(
        x + width / 2, accept_qgate, width,
        color="coral", alpha=0.8, edgecolor="darkred",
        label="qgate score-fusion accept rate",
    )
    ax4.set_xlabel("Number of Ansatz Layers (L)", fontsize=13)
    ax4.set_ylabel("Acceptance Rate", fontsize=13)
    ax4.set_title(
        f"TSVF Post-Selection & qgate Acceptance Rates\n"
        f"{backend_name} · {patent_ref}",
        fontsize=12,
    )
    ax4.set_xticks(x)
    ax4.set_xticklabels(layers)
    ax4.set_ylim(0, 1.05)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis="y")
    fig4.tight_layout()
    p4 = out_dir / "acceptance_rate.png"
    fig4.savefig(p4, dpi=200)
    plt.close(fig4)
    print(f"  Saved: {p4}")

    # ── Plot 5: Galton threshold adaptation ───────────────────────────
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    thresholds = [r["qgate_dyn_threshold_tsvf"] for r in results]
    scores_tsvf = [r["qgate_mean_score_tsvf"] for r in results]
    ax5.plot(
        layers, thresholds, "g^-", linewidth=2, markersize=8,
        label="Galton adaptive threshold (θ)",
    )
    ax5.plot(
        layers, scores_tsvf, "b.-", linewidth=2, markersize=8,
        label="Mean combined score (TSVF)",
    )
    ax5.set_xlabel("Number of Ansatz Layers (L)", fontsize=13)
    ax5.set_ylabel("Score / Threshold", fontsize=13)
    ax5.set_title(
        f"Galton Adaptive Threshold vs. Mean Score\n"
        f"{backend_name} · {patent_ref}",
        fontsize=12,
    )
    ax5.legend(fontsize=10)
    ax5.set_xticks(layers)
    ax5.set_ylim(0, 1.05)
    ax5.grid(True, alpha=0.3)
    fig5.tight_layout()
    p5 = out_dir / "galton_threshold.png"
    fig5.savefig(p5, dpi=200)
    plt.close(fig5)
    print(f"  Saved: {p5}")

    # ── Plot 6: Energy Ratio vs Layers ────────────────────────────────
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    ax6.plot(
        layers, ratio_std, "ro-", linewidth=2, markersize=8,
        label="Standard VQE",
    )
    ax6.plot(
        layers, ratio_tsvf, "b^-", linewidth=2, markersize=8,
        label="TSVF VQE (qgate post-selected)",
    )
    ax6.axhline(
        y=1.0, color="green", linestyle="--", alpha=0.5,
        label="Exact (ratio=1.0)",
    )
    ax6.set_xlabel("Number of Ansatz Layers (L)", fontsize=13)
    ax6.set_ylabel("Energy Ratio (E_est / E_exact)", fontsize=13)
    ax6.set_title(
        f"Energy Ratio vs. Ansatz Depth\n"
        f"{N_QUBITS}-qubit TFIM · {backend_name} · qgate v0.5.0\n{patent_ref}",
        fontsize=11,
    )
    ax6.legend(fontsize=11, loc="lower right")
    ax6.set_xticks(layers)
    ax6.grid(True, alpha=0.3)
    fig6.tight_layout()
    p6 = out_dir / "energy_ratio_vs_layers.png"
    fig6.savefig(p6, dpi=200)
    plt.close(fig6)
    print(f"  Saved: {p6}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "VQE vs TSVF-VQE TFIM ground-state energy comparison — "
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
        "--max-layers", type=int, default=8,
        help="Maximum ansatz layers L (default 8)",
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

    # Compute exact ground-state energy
    exact_gs_energy = tfim_exact_ground_energy(N_QUBITS, J_COUPLING, H_FIELD)

    print("=" * 72)
    print("VQE vs TSVF-VQE — qgate TRAJECTORY FILTER EXPERIMENT (TFIM)")
    print("Package: qgate v0.5.0  (VQETSVFAdapter + Galton thresholding)")
    print("Patent pending (see LICENSE)")
    print("=" * 72)
    print(f"Qubits:      {N_QUBITS}")
    print(f"J coupling:  {J_COUPLING}")
    print(f"h field:     {H_FIELD}")
    print(f"Exact GS E:  {exact_gs_energy:.6f}")

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
    print(f"Layers:      1 – {args.max_layers}")
    print(f"Shots:       {args.shots}")
    print(f"Output:      {out_dir}")
    print()

    t_start = time.time()

    results = run_experiment(
        backend,
        exact_gs_energy=exact_gs_energy,
        max_layers=args.max_layers,
        shots=args.shots,
        out_dir=out_dir,
    )

    elapsed = time.time() - t_start

    # Save CSV
    csv_path = out_dir / "vqe_tsvf_results.csv"
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
        "n_qubits": N_QUBITS,
        "j_coupling": J_COUPLING,
        "h_field": H_FIELD,
        "exact_ground_state_energy": exact_gs_energy,
        "max_layers": args.max_layers,
        "shots": args.shots,
        "elapsed_seconds": round(elapsed, 1),
        "qgate_version": "0.5.0",
        "adapter": "VQETSVFAdapter",
        "patent_ref": "Patent pending",
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(results, out_dir, backend_name, exact_gs_energy)

    # Print summary table
    print(f"\n{'='*72}")
    print("SUMMARY TABLE")
    print(f"{'='*72}")
    print(f"  Exact ground-state energy: {exact_gs_energy:.6f}")
    header = (
        f"{'L':>3} {'E_std':>9} {'E_tsvf':>9} {'Err_std':>8} "
        f"{'Err_tsvf':>9} {'D_std':>6} {'D_tsvf':>7} "
        f"{'Accept%':>8} {'qgateA%':>8} {'TTS':>7} {'θ':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['layers']:>3} {r['energy_std']:>9.4f} "
            f"{r['energy_tsvf']:>9.4f} {r['energy_error_std']:>8.4f} "
            f"{r['energy_error_tsvf']:>9.4f} {r['depth_standard']:>6} "
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
