#!/usr/bin/env python3
"""
run_ibm_experiment.py – Main entry point for IBM Quantum dynamic-circuit
post-selection conditioning experiment.

Usage
-----
    # Local noisy simulation (AerSimulator)
    python qiskit_experiment/run_ibm_experiment.py --mode aer

    # Real IBM Quantum backend (needs IBMQ token)
    python qiskit_experiment/run_ibm_experiment.py --mode ibm --token <YOUR_TOKEN>

    # Custom sweep parameters
    python qiskit_experiment/run_ibm_experiment.py --mode aer \\
        --shots 2000 --N 1 2 4 --D 2 4 --W 2

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure package imports work when run as a script
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from qiskit_experiment.sweep import SweepConfig, run_sweep
from qiskit_experiment.plots import generate_all_plots


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def get_aer_backend(noise_model: bool = True):
    """Return a local AerSimulator, optionally with a basic noise model.

    Uses a simple depolarising noise model to approximate IBM hardware
    noise without requiring an IBM account.
    """
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        raise ImportError(
            "qiskit-aer is required for local simulation.\n"
            "Install with: pip install qiskit-aer"
        )

    if noise_model:
        try:
            from qiskit_aer.noise import NoiseModel, depolarizing_error
            model = NoiseModel()
            # Single-qubit gate error ~ 0.1%
            error_1q = depolarizing_error(1e-3, 1)
            model.add_all_qubit_quantum_error(error_1q, ["rx", "ry", "rz", "h", "x"])
            # Two-qubit gate error ~ 1%
            error_2q = depolarizing_error(1e-2, 2)
            model.add_all_qubit_quantum_error(error_2q, ["cx"])
            # Measurement error ~ 2%
            error_meas = depolarizing_error(2e-2, 1)
            model.add_all_qubit_quantum_error(error_meas, ["measure"])
            return AerSimulator(noise_model=model)
        except Exception as e:
            print(f"Warning: Could not build noise model ({e}); "
                  f"using ideal simulator.")
            return AerSimulator()
    else:
        return AerSimulator()


def get_ibm_backend(token: str | None = None,
                    instance: str = "ibm-q/open/main",
                    min_qubits: int = 16):
    """Connect to IBM Quantum and select the least-busy dynamic-capable backend.

    Args:
        token:      IBM Quantum API token.  Falls back to env var
                    ``IBMQ_TOKEN`` or saved credentials.
        instance:   IBM Quantum instance (hub/group/project).
        min_qubits: minimum number of qubits required (default 16 for N=8).

    Returns:
        An IBM Runtime backend.
    """
    # Resolution order: explicit arg → env var → .secrets.json → saved creds
    if not token:
        token = os.environ.get("IBMQ_TOKEN")
    if not token:
        secrets_path = Path(__file__).resolve().parent.parent / ".secrets.json"
        if secrets_path.is_file():
            try:
                with open(secrets_path) as f:
                    token = json.load(f).get("ibmq_token")
            except Exception:
                pass

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        raise ImportError(
            "qiskit-ibm-runtime is required for real hardware.\n"
            "Install with: pip install qiskit-ibm-runtime"
        )

    if token:
        # Save credentials for future use
        try:
            QiskitRuntimeService.save_account(
                channel="ibm_quantum_platform",
                token=token,
                overwrite=True,
            )
        except Exception:
            pass  # may already be saved
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform", token=token,
        )
    else:
        # Try loading saved credentials
        try:
            service = QiskitRuntimeService(channel="ibm_quantum_platform")
        except Exception as e:
            raise RuntimeError(
                "No IBM Quantum token found. Provide --token <TOKEN> or set "
                "the IBMQ_TOKEN environment variable.\n"
                f"Error: {e}"
            )

    # Find least-busy backend with enough qubits & dynamic-circuit support
    print(f"Searching for backends with ≥{min_qubits} qubits...")
    backends = service.backends(
        min_num_qubits=min_qubits,
        simulator=False,
        operational=True,
    )

    if not backends:
        raise RuntimeError(
            f"No operational backends found with ≥{min_qubits} qubits."
        )

    # Pick least busy – use the backend objects directly
    backend = service.least_busy(
        min_num_qubits=min_qubits,
        simulator=False,
        operational=True,
    )
    print(f"Selected backend: {backend.name}  "
          f"({backend.num_qubits} qubits)")
    return backend


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IBM Quantum dynamic-circuit post-selection experiment. "
                    "Patent ref: US 63/983,831 & 63/989,632 | IL 326915",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local noisy simulation
  python qiskit_experiment/run_ibm_experiment.py --mode aer

  # Ideal (noiseless) simulation
  python qiskit_experiment/run_ibm_experiment.py --mode aer --no-noise

  # Real IBM hardware
  python qiskit_experiment/run_ibm_experiment.py --mode ibm --token YOUR_TOKEN

  # Custom sweep
  python qiskit_experiment/run_ibm_experiment.py --mode aer \\
      --N 1 2 4 --D 2 4 --W 2 4 --shots 2000
        """,
    )
    p.add_argument("--mode", choices=["aer", "ibm"], default="aer",
                   help="Backend mode: 'aer' for local simulation, "
                        "'ibm' for real hardware (default: aer)")
    p.add_argument("--token", type=str, default=None,
                   help="IBM Quantum API token (or set IBMQ_TOKEN env var)")
    p.add_argument("--instance", type=str, default="ibm-q/open/main",
                   help="IBM Quantum instance (default: ibm-q/open/main)")
    p.add_argument("--no-noise", action="store_true",
                   help="Use ideal (noiseless) AerSimulator")

    # Sweep parameters
    p.add_argument("--N", nargs="+", type=int, default=[1, 2, 4, 8],
                   help="System sizes (default: 1 2 4 8)")
    p.add_argument("--W", nargs="+", type=int, default=[2, 4],
                   help="Monitoring cycles (default: 2 4)")
    p.add_argument("--D", nargs="+", type=int, default=[2, 4, 8],
                   help="Scramble depths (default: 2 4 8)")
    p.add_argument("--k", nargs="+", type=float, default=[0.8, 0.9],
                   help="Hierarchical fractions (default: 0.8 0.9)")
    p.add_argument("--alpha", nargs="+", type=float, default=[0.3, 0.7],
                   help="Fusion weights (default: 0.3 0.7)")
    p.add_argument("--threshold", type=float, default=0.65,
                   help="Score-fusion combined threshold (default: 0.65)")
    p.add_argument("--shots", type=int, default=5000,
                   help="Shots per circuit (default: 5000)")
    p.add_argument("--probe-shots", type=int, default=1000,
                   help="Shots for probe circuit (default: 1000)")
    p.add_argument("--theta-batch", type=float, default=0.65,
                   help="Batch-abort threshold (default: 0.65)")

    # Output
    p.add_argument("--output-dir", type=str, default="patent_appendix",
                   help="Output directory (default: patent_appendix)")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip plot generation")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress output")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 72)
    print("QUANTUM ERROR SUPPRESSION VIA POST-SELECTION CONDITIONING")
    print("Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915")
    print("=" * 72)
    print(f"Mode:    {args.mode}")
    print(f"Output:  {args.output_dir}")
    print(f"N:       {args.N}")
    print(f"W:       {args.W}")
    print(f"D:       {args.D}")
    print(f"Shots:   {args.shots}")
    print()

    # --- Select backend ---
    t0 = time.time()
    if args.mode == "aer":
        backend = get_aer_backend(noise_model=not args.no_noise)
        noise_str = "noisy" if not args.no_noise else "ideal"
        print(f"Backend: AerSimulator ({noise_str})")
    else:
        n_max = max(args.N)
        min_qubits = 2 * n_max + 1  # data + ancilla
        backend = get_ibm_backend(
            token=args.token,
            instance=args.instance,
            min_qubits=min_qubits,
        )
    print()

    # --- Build sweep config ---
    config = SweepConfig(
        N_values=args.N,
        W_values=args.W,
        D_values=args.D,
        k_values=args.k,
        alpha_values=args.alpha,
        threshold_combined=args.threshold,
        shots=args.shots,
        theta_batch=args.theta_batch,
        probe_shots=args.probe_shots,
    )

    # --- Run sweep ---
    results = run_sweep(
        backend=backend,
        config=config,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )

    # --- Generate plots ---
    if not args.no_plots and results:
        print("\nGenerating patent appendix figures...")
        saved = generate_all_plots(results, output_dir=args.output_dir)
        for p in saved:
            print(f"  ✓ {p}")

    # --- Summary ---
    elapsed = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"Experiment complete!")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Results:    {args.output_dir}/results.csv ({len(results)} rows)")
    print(f"Figures:    {args.output_dir}/*.png")
    print(f"Run log:    {args.output_dir}/run_log.json")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
