"""
sweep.py – Parameter sweep engine.

Sweeps over:
    N  ∈ {1, 2, 4, 8}       subsystems (Bell pairs)
    W  ∈ {2, 4}              monitoring cycles
    D  ∈ {2, 4, 8}           scramble depth
    k  ∈ {0.8, 0.9}          hierarchical fraction
    α  ∈ {0.3, 0.7}          fusion weight
    θ_c ∈ {0.65}             fusion combined threshold (fixed)

For each (N, D, W), one circuit is built and sampled once (5000 shots).
The same counts are then post-processed with every conditioning variant.

Batch-level abort: a short W=1 probe is run first.  If the probe pass-rate
< θ_batch (0.65), the full circuit is skipped and marked as aborted.

Patent pending (see LICENSE)
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .circuits import build_monitoring_circuit, build_probe_circuit
from .conditioning import (
    ConditioningResult,
    apply_conditioning,
    evaluate_probe_batch,
)


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

DEFAULT_SWEEP = {
    "N_values": [1, 2, 4, 8],
    "W_values": [2, 4],
    "D_values": [2, 4, 8],
    "k_values": [0.8, 0.9],
    "alpha_values": [0.3, 0.7],
    "threshold_combined": 0.65,
    "shots": 5000,
    "theta_batch": 0.65,        # batch-level abort threshold
    "probe_shots": 1000,        # shots for probe circuit
}


@dataclass
class SweepConfig:
    N_values: list[int]
    W_values: list[int]
    D_values: list[int]
    k_values: list[float]
    alpha_values: list[float]
    threshold_combined: float
    shots: int
    theta_batch: float
    probe_shots: int

    @classmethod
    def default(cls) -> "SweepConfig":
        return cls(**DEFAULT_SWEEP)

    def total_circuit_configs(self) -> int:
        return len(self.N_values) * len(self.W_values) * len(self.D_values)

    def total_conditioning_configs(self) -> int:
        """Total rows in the output CSV."""
        n_circ = self.total_circuit_configs()
        # per circuit: 1 global + len(k) hierarchical + len(alpha) fusion
        n_variants = 1 + len(self.k_values) + len(self.alpha_values)
        return n_circ * n_variants


# ---------------------------------------------------------------------------
# Execution helper (works for both Aer and real backends)
# ---------------------------------------------------------------------------

def _run_sampler(circuit, backend, shots: int) -> dict[str, int]:
    """Run a circuit with the Sampler primitive and return counts.

    Supports both qiskit-aer (AerSimulator) and IBM Runtime backends.
    Falls back to backend.run() + counts if Sampler is not available.
    """
    try:
        # Try V2 Sampler (qiskit-ibm-runtime or qiskit-aer >= 0.13)
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa_circuit = pm.run(circuit)

        sampler = Sampler(mode=backend)
        job = sampler.run([isa_circuit], shots=shots)
        result = job.result()

        # V2 Sampler returns PubResult with data
        pub_result = result[0]
        creg_name = circuit.cregs[0].name
        bitarray = pub_result.data[creg_name]
        counts = bitarray.get_counts()
        return counts

    except (ImportError, Exception):
        pass

    try:
        # Fallback: direct backend.run() (Aer or BasicProvider)
        from qiskit import transpile

        transpiled = transpile(circuit, backend=backend, optimization_level=1)
        job = backend.run(transpiled, shots=shots)
        result = job.result()
        return result.get_counts(0)

    except Exception as e:
        raise RuntimeError(
            f"Failed to run circuit on backend {backend}: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(
    backend,
    config: Optional[SweepConfig] = None,
    output_dir: str = "patent_appendix",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Execute the full parameter sweep.

    Args:
        backend:    Qiskit backend (AerSimulator or IBM Runtime backend).
        config:     SweepConfig (defaults to DEFAULT_SWEEP).
        output_dir: directory for CSVs and figures.
        verbose:    print progress.

    Returns:
        list of result dicts (also saved to CSV).
    """
    if config is None:
        config = SweepConfig.default()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    total = config.total_circuit_configs()
    idx = 0
    t0 = time.time()

    if verbose:
        print("=" * 72)
        print("QISKIT DYNAMIC-CIRCUIT POST-SELECTION SWEEP")
        print(f"Backend: {backend}")
        print(f"Circuit configs: {total}")
        print(f"Conditioning variants per circuit: "
              f"1 global + {len(config.k_values)} hierarchical + "
              f"{len(config.alpha_values)} fusion")
        print(f"Shots per circuit: {config.shots}")
        print(f"Output: {out.resolve()}")
        print("=" * 72)

    for N in config.N_values:
        for W in config.W_values:
            for D in config.D_values:
                idx += 1
                tag = f"N={N} W={W} D={D}"

                if verbose:
                    elapsed = time.time() - t0
                    print(f"\n[{idx}/{total}] {tag}  "
                          f"(elapsed {elapsed:.0f}s)")

                # --- Batch-level abort (probe circuit) ---
                probe_circ = build_probe_circuit(N, D)
                if verbose:
                    print(f"  Probe (W=1, {config.probe_shots} shots)...",
                          end=" ", flush=True)

                probe_counts = _run_sampler(probe_circ, backend,
                                            config.probe_shots)
                proceed, probe_rate = evaluate_probe_batch(
                    probe_counts, N, theta=config.theta_batch
                )
                if verbose:
                    status = "PROCEED" if proceed else "ABORT"
                    print(f"pass_rate={probe_rate:.3f} → {status}")

                if not proceed:
                    # Record aborted results for all variants
                    for variant_info in _variant_list(config):
                        res = ConditioningResult(
                            variant=variant_info["variant"],
                            n_subsystems=N, depth=D, n_cycles=W,
                            k_fraction=variant_info.get("k"),
                            alpha=variant_info.get("alpha"),
                            threshold_combined=variant_info.get("tc"),
                            probe_pass_rate=probe_rate,
                            batch_aborted=True,
                        )
                        all_results.append(res.as_dict())
                    continue

                # --- Full circuit ---
                circ = build_monitoring_circuit(N, D, W)
                if verbose:
                    print(f"  Full circuit (W={W}, {config.shots} shots)...",
                          end=" ", flush=True)

                counts = _run_sampler(circ, backend, config.shots)
                if verbose:
                    print(f"done ({len(counts)} unique bitstrings)")

                # --- Apply conditioning variants ---
                # 1) Global
                res_g = apply_conditioning(counts, N, W, D, "global")
                res_g.probe_pass_rate = probe_rate
                all_results.append(res_g.as_dict())
                if verbose:
                    print(f"    Global: accept={res_g.acceptance_probability:.3f}")

                # 2) Hierarchical k-of-N
                for k in config.k_values:
                    res_h = apply_conditioning(counts, N, W, D,
                                               "hierarchical",
                                               k_fraction=k)
                    res_h.probe_pass_rate = probe_rate
                    all_results.append(res_h.as_dict())
                    if verbose:
                        print(f"    Hierarchical k={k}: "
                              f"accept={res_h.acceptance_probability:.3f}")

                # 3) Score fusion
                for alpha in config.alpha_values:
                    res_f = apply_conditioning(
                        counts, N, W, D, "score_fusion",
                        alpha=alpha,
                        threshold_combined=config.threshold_combined,
                    )
                    res_f.probe_pass_rate = probe_rate
                    all_results.append(res_f.as_dict())
                    if verbose:
                        print(f"    Fusion α={alpha}: "
                              f"accept={res_f.acceptance_probability:.3f} "
                              f"mean_score="
                              f"{np.mean(res_f.scores) if res_f.scores else 0:.3f}")

    # --- Save CSV ---
    csv_path = out / "results.csv"
    _save_csv(all_results, csv_path)

    # --- Save run log ---
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": str(backend),
        "config": {
            "N_values": config.N_values,
            "W_values": config.W_values,
            "D_values": config.D_values,
            "k_values": config.k_values,
            "alpha_values": config.alpha_values,
            "threshold_combined": config.threshold_combined,
            "shots": config.shots,
            "theta_batch": config.theta_batch,
            "probe_shots": config.probe_shots,
        },
        "total_circuit_configs": total,
        "total_rows": len(all_results),
        "elapsed_seconds": time.time() - t0,
    }
    with open(out / "run_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)

    if verbose:
        print(f"\n{'=' * 72}")
        print(f"Sweep complete: {len(all_results)} rows → {csv_path}")
        print(f"Elapsed: {time.time() - t0:.1f}s")

    return all_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _variant_list(config: SweepConfig) -> list[dict]:
    """List of all conditioning variants for record-keeping."""
    variants: list[dict] = [{"variant": "global"}]
    for k in config.k_values:
        variants.append({"variant": "hierarchical", "k": k})
    for alpha in config.alpha_values:
        variants.append({
            "variant": "score_fusion",
            "alpha": alpha,
            "tc": config.threshold_combined,
        })
    return variants


def _save_csv(results: list[dict], path: Path) -> None:
    """Write results to CSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
