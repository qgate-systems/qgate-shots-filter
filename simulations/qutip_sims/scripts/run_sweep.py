#!/usr/bin/env python3
"""
Sweep runner for driven-qubit dephasing simulations with windowed acceptance.

Supports multiple acceptance modes:
- final: accept if fidelity[-1] >= threshold
- window_max: accept if max(fidelity in [T-Δ, T]) >= threshold
- window_mean: accept if mean(fidelity in [T-Δ, T]) >= threshold

Runs a grid of (gamma_phi, threshold, accept_window) combinations with repeats.

Outputs:
- sweep_config.json: full configuration
- results.parquet / results.csv: one row per run
- summary_stats.json: aggregate statistics
- figures/: heatmaps per accept_window
- runs/: per-run artifacts
- README.md: summary of sweep

Usage:
    uv run python scripts/run_sweep.py
    uv run python scripts/run_sweep.py --accept-mode window_max --accept-windows 0.5 1.0 --repeats 10
    uv run python scripts/run_sweep.py --threshold-min 0.60 --threshold-max 0.80 --threshold-steps 11
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sim import run_simulation


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def parse_float_list(s: str) -> list[float]:
    """Parse a comma-separated string into a list of floats."""
    if not s.strip():
        raise ValueError("Empty list string")
    parts = [p.strip() for p in s.split(",")]
    return [float(p) for p in parts if p]


def safe_write_json(path: Path, data: dict) -> None:
    """Write dict to JSON file, handling non-serializable types."""
    def default_serializer(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=default_serializer)


def get_git_commit_hash() -> str | None:
    """Get current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def make_sweep_dir(base: str = "runs", prefix: str = "window_acceptance") -> tuple[Path, str]:
    """Create a unique sweep directory with timestamp and short ID."""
    sweep_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
    ts = time.strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(base) / f"{prefix}_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "figures").mkdir(exist_ok=True)
    (sweep_dir / "runs").mkdir(exist_ok=True)
    return sweep_dir, sweep_id


def make_run_key(
    gamma_phi: float,
    threshold: float,
    accept_window: float,
    repeat: int
) -> str:
    """Generate a stable run key string."""
    return f"gphi={gamma_phi:.4f}_thr={threshold:.2f}_win={accept_window:.1f}_rep={repeat:03d}"


def flatten_summary(
    summary: dict,
    sweep_params: dict,
    extra_fields: dict
) -> dict:
    """
    Flatten summary dict into a row for the results DataFrame.
    
    Priority: extra_fields > sweep_params > summary
    """
    row = {}
    
    for k, v in summary.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            row[k] = v
        elif isinstance(v, (np.integer,)):
            row[k] = int(v)
        elif isinstance(v, (np.floating,)):
            row[k] = float(v)
        elif isinstance(v, (np.bool_,)):
            row[k] = bool(v)
    
    row.update(sweep_params)
    row.update(extra_fields)
    
    return row


# -----------------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------------

def plot_heatmap(
    data: np.ndarray,
    gamma_phis: list[float],
    thresholds: list[float],
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = ".2f",
    xlabel: str = "Threshold",
    ylabel: str = "γ_φ (dephasing rate)"
) -> plt.Figure:
    """Create a heatmap-like plot using imshow."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    im = ax.imshow(
        data,
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        origin="lower"
    )
    
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45, ha="right")
    ax.set_yticks(range(len(gamma_phis)))
    ax.set_yticklabels([f"{g:.4f}" for g in gamma_phis])
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    cbar = fig.colorbar(im, ax=ax)
    
    # Annotate cells
    for i in range(len(gamma_phis)):
        for j in range(len(thresholds)):
            val = data[i, j]
            if np.isnan(val):
                text = "N/A"
            else:
                text = f"{val:{fmt}}"
            
            # Choose text color
            if np.isnan(val):
                text_color = "gray"
            else:
                vmin_eff = vmin if vmin is not None else np.nanmin(data)
                vmax_eff = vmax if vmax is not None else np.nanmax(data)
                bg_val = (val - vmin_eff) / (vmax_eff - vmin_eff + 1e-9)
                text_color = "white" if bg_val > 0.5 else "black"
            
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=7)
    
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path_stem: Path) -> None:
    """Save figure as both SVG and PNG."""
    fig.savefig(path_stem.with_suffix(".svg"))
    fig.savefig(path_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Sweep logic
# -----------------------------------------------------------------------------

def run_single(
    gamma_phi: float,
    threshold: float,
    accept_window: float,
    accept_mode: str,
    drive_amp: float,
    drive_freq: float,
    t_max: float,
    n_steps: int,
    seed: int,
    repeat: int,
    sweep_dir: Path,
    sweep_id: str,
    save_timeseries: bool
) -> dict:
    """Run a single simulation and return result row."""
    run_key = make_run_key(gamma_phi, threshold, accept_window, repeat)
    run_dir = sweep_dir / "runs" / run_key
    run_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    
    sweep_params = {
        "sweep_id": sweep_id,
        "run_id": f"{sweep_id}_{run_key}",
        "run_key": run_key,
        "timestamp": timestamp,
        "seed": seed,
        "repeat": repeat,
        "gamma_phi": gamma_phi,
        "accept_threshold": threshold,
        "accept_mode": accept_mode,
        "accept_window": accept_window,
        "drive_amp": drive_amp,
        "drive_freq": drive_freq,
        "t_max": t_max,
        "n_steps": n_steps,
    }
    
    try:
        df, summary = run_simulation(
            drive_amp=drive_amp,
            drive_freq=drive_freq,
            gamma_phi=gamma_phi,
            threshold=threshold,
            t_max=t_max,
            n_steps=n_steps,
            accept_mode=accept_mode,
            accept_window=accept_window,
            seed=seed
        )
        
        # Rename for consistency
        summary["final_fidelity"] = summary.pop("fidelity_final")
        summary["peak_fidelity"] = summary.pop("fidelity_max")
        
        # Save per-run summary
        summary["run_key"] = run_key
        safe_write_json(run_dir / "summary.json", summary)
        
        if save_timeseries:
            df.to_parquet(run_dir / "timeseries.parquet")
        
        row = flatten_summary(
            summary,
            sweep_params,
            {"status": "ok", "error": None}
        )
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        row = {
            **sweep_params,
            "status": "error",
            "error": error_msg,
            "accepted": None,
            "final_fidelity": None,
            "peak_fidelity": None,
        }
        safe_write_json(run_dir / "summary.json", {
            "error": error_msg,
            "traceback": traceback.format_exc()
        })
    
    return row


def compute_summary_stats(df: pd.DataFrame) -> dict:
    """Compute comprehensive aggregate statistics."""
    ok_df = df[df["status"] == "ok"].copy()
    
    total_runs = len(df)
    ok_runs = len(ok_df)
    error_runs = total_runs - ok_runs
    
    stats = {
        "total_runs": total_runs,
        "ok_runs": ok_runs,
        "error_runs": error_runs,
    }
    
    if ok_runs > 0:
        # Overall acceptance
        accepted_df = ok_df[ok_df["accepted"] == True]
        stats["acceptance_rate_overall"] = float(len(accepted_df) / ok_runs)
        
        # Fidelity stats (all runs)
        stats["final_fidelity"] = {
            "mean": float(ok_df["final_fidelity"].mean()),
            "median": float(ok_df["final_fidelity"].median()),
            "std": float(ok_df["final_fidelity"].std()),
            "min": float(ok_df["final_fidelity"].min()),
            "max": float(ok_df["final_fidelity"].max()),
        }
        stats["peak_fidelity"] = {
            "mean": float(ok_df["peak_fidelity"].mean()),
            "median": float(ok_df["peak_fidelity"].median()),
            "std": float(ok_df["peak_fidelity"].std()),
            "min": float(ok_df["peak_fidelity"].min()),
            "max": float(ok_df["peak_fidelity"].max()),
        }
        
        # Accepted-only stats
        if len(accepted_df) > 0:
            stats["accepted_final_fidelity"] = {
                "mean": float(accepted_df["final_fidelity"].mean()),
                "median": float(accepted_df["final_fidelity"].median()),
                "std": float(accepted_df["final_fidelity"].std()),
            }
            stats["accepted_peak_fidelity"] = {
                "mean": float(accepted_df["peak_fidelity"].mean()),
                "median": float(accepted_df["peak_fidelity"].median()),
                "std": float(accepted_df["peak_fidelity"].std()),
            }
        else:
            stats["accepted_final_fidelity"] = None
            stats["accepted_peak_fidelity"] = None
        
        # Per (gamma_phi, threshold, window) stats
        group_stats = []
        for (gphi, thr, win), group in ok_df.groupby(["gamma_phi", "accept_threshold", "accept_window"]):
            acc_rate = group["accepted"].mean()
            group_stats.append({
                "gamma_phi": gphi,
                "accept_threshold": thr,
                "accept_window": win,
                "acceptance_rate": float(acc_rate),
                "mean_final_fidelity": float(group["final_fidelity"].mean()),
                "mean_peak_fidelity": float(group["peak_fidelity"].mean()),
                "n_runs": len(group),
            })
        stats["by_grid_point"] = group_stats
        
    return stats


def create_aggregated_heatmap_data(
    df: pd.DataFrame,
    gamma_phis: list[float],
    thresholds: list[float],
    accept_window: float,
    value_col: str,
    agg_func: str = "mean"
) -> np.ndarray:
    """Create 2D array for heatmap, aggregating over repeats."""
    data = np.full((len(gamma_phis), len(thresholds)), np.nan)
    
    # Filter to this window
    window_df = df[(np.abs(df["accept_window"] - accept_window) < 1e-9) & (df["status"] == "ok")]
    
    for i, gphi in enumerate(gamma_phis):
        for j, thr in enumerate(thresholds):
            mask = (
                (np.abs(window_df["gamma_phi"] - gphi) < 1e-9) &
                (np.abs(window_df["accept_threshold"] - thr) < 1e-9)
            )
            vals = window_df.loc[mask, value_col].dropna()
            if len(vals) > 0:
                if agg_func == "mean":
                    data[i, j] = float(vals.mean())
                elif agg_func == "sum":
                    data[i, j] = float(vals.sum())
                elif agg_func == "fraction":
                    data[i, j] = float(vals.sum() / len(vals))
    
    return data


def create_accepted_mean_heatmap_data(
    df: pd.DataFrame,
    gamma_phis: list[float],
    thresholds: list[float],
    accept_window: float,
    value_col: str
) -> np.ndarray:
    """Create 2D array for accepted-only mean (NaN if none accepted)."""
    data = np.full((len(gamma_phis), len(thresholds)), np.nan)
    
    window_df = df[(np.abs(df["accept_window"] - accept_window) < 1e-9) & (df["status"] == "ok")]
    
    for i, gphi in enumerate(gamma_phis):
        for j, thr in enumerate(thresholds):
            mask = (
                (np.abs(window_df["gamma_phi"] - gphi) < 1e-9) &
                (np.abs(window_df["accept_threshold"] - thr) < 1e-9) &
                (window_df["accepted"] == True)
            )
            vals = window_df.loc[mask, value_col].dropna()
            if len(vals) > 0:
                data[i, j] = float(vals.mean())
    
    return data


def generate_heatmaps(
    df: pd.DataFrame,
    gamma_phis: list[float],
    thresholds: list[float],
    accept_windows: list[float],
    figures_dir: Path,
    accept_mode: str
) -> None:
    """Generate all heatmaps for each accept_window."""
    
    for win in accept_windows:
        win_dir = figures_dir / f"window_{win:.1f}"
        win_dir.mkdir(exist_ok=True)
        
        # 1. Acceptance fraction heatmap
        acc_data = create_aggregated_heatmap_data(
            df, gamma_phis, thresholds, win, "accepted", "fraction"
        )
        fig = plot_heatmap(
            acc_data, gamma_phis, thresholds,
            title=f"Acceptance Fraction ({accept_mode}, Δ={win})",
            cmap="RdYlGn", vmin=0, vmax=1, fmt=".2f"
        )
        save_figure(fig, win_dir / "acceptance_fraction_heatmap")
        
        # 2. Final fidelity heatmap (mean)
        fid_final_data = create_aggregated_heatmap_data(
            df, gamma_phis, thresholds, win, "final_fidelity", "mean"
        )
        fig = plot_heatmap(
            fid_final_data, gamma_phis, thresholds,
            title=f"Mean Final Fidelity ({accept_mode}, Δ={win})",
            cmap="viridis", vmin=0, vmax=1, fmt=".2f"
        )
        save_figure(fig, win_dir / "final_fidelity_heatmap")
        
        # 3. Peak fidelity heatmap (mean)
        fid_peak_data = create_aggregated_heatmap_data(
            df, gamma_phis, thresholds, win, "peak_fidelity", "mean"
        )
        fig = plot_heatmap(
            fid_peak_data, gamma_phis, thresholds,
            title=f"Mean Peak Fidelity ({accept_mode}, Δ={win})",
            cmap="plasma", vmin=0, vmax=1, fmt=".2f"
        )
        save_figure(fig, win_dir / "peak_fidelity_heatmap")
        
        # 4. Accepted-only mean final fidelity
        acc_fid_data = create_accepted_mean_heatmap_data(
            df, gamma_phis, thresholds, win, "final_fidelity"
        )
        fig = plot_heatmap(
            acc_fid_data, gamma_phis, thresholds,
            title=f"Accepted Mean Final Fidelity ({accept_mode}, Δ={win})",
            cmap="coolwarm", vmin=0, vmax=1, fmt=".2f"
        )
        save_figure(fig, win_dir / "accepted_mean_final_fidelity_heatmap")


def write_readme(sweep_dir: Path, config: dict, stats: dict) -> None:
    """Write a README.md summarizing the sweep."""
    thresholds_preview = ", ".join(f"{t:.2f}" for t in config['thresholds'][:3])
    gamma_phis_preview = ", ".join(f"{g:.4f}" for g in config['gamma_phis'][:3])
    
    acceptance_rate = stats.get('acceptance_rate_overall')
    if acceptance_rate is not None:
        acceptance_str = f"{acceptance_rate:.2%}"
    else:
        acceptance_str = "N/A"
    
    fid_final = stats.get('final_fidelity', {})
    fid_peak = stats.get('peak_fidelity', {})
    
    readme = f"""# Sweep: {config['sweep_id']}

## Configuration

- **Timestamp**: {config['timestamp']}
- **Accept Mode**: {config['accept_mode']}
- **Accept Windows**: {config['accept_windows']}
- **Thresholds**: {thresholds_preview}... ({len(config['thresholds'])} values)
- **Gamma Phis**: {gamma_phis_preview}... ({len(config['gamma_phis'])} values)
- **Repeats per grid point**: {config['repeats']}
- **Total runs**: {config['total_runs']}
- **Git commit**: {config.get('git_commit') or 'N/A'}

## Simulation Parameters

- **t_max**: {config['t_max']}
- **n_steps**: {config['n_steps']}
- **drive_amp**: {config['drive_amp']}
- **drive_freq**: {config['drive_freq']}

## Summary Statistics

- **Total runs**: {stats['total_runs']}
- **OK runs**: {stats['ok_runs']}
- **Error runs**: {stats['error_runs']}
- **Overall acceptance rate**: {acceptance_str}

### Fidelity Statistics (all runs)

| Metric | Mean | Median | Std |
|--------|------|--------|-----|
| Final Fidelity | {fid_final.get('mean', 'N/A'):.4f} | {fid_final.get('median', 'N/A'):.4f} | {fid_final.get('std', 'N/A'):.4f} |
| Peak Fidelity | {fid_peak.get('mean', 'N/A'):.4f} | {fid_peak.get('median', 'N/A'):.4f} | {fid_peak.get('std', 'N/A'):.4f} |

## Files

- `sweep_config.json`: Full configuration
- `results.parquet`: All run results (one row per run)
- `results.csv`: Same as above (CSV format)
- `summary_stats.json`: Aggregate statistics
- `figures/`: Heatmaps organized by accept_window
  - `window_X.X/acceptance_fraction_heatmap.png/.svg`
  - `window_X.X/final_fidelity_heatmap.png/.svg`
  - `window_X.X/peak_fidelity_heatmap.png/.svg`
  - `window_X.X/accepted_mean_final_fidelity_heatmap.png/.svg`
- `runs/`: Per-run artifacts (summary.json, optional timeseries.parquet)
- `README.md`: This file
"""
    
    with open(sweep_dir / "README.md", "w") as f:
        f.write(readme)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run parameter sweep with windowed acceptance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Output configuration
    parser.add_argument("--out-dir", type=str, default="runs",
                        help="Base directory for sweep outputs")
    
    # Simulation parameters
    parser.add_argument("--t-max", type=float, default=12.0,
                        help="Maximum simulation time")
    parser.add_argument("--n-steps", type=int, default=500,
                        help="Number of time steps")
    parser.add_argument("--drive-amp", type=float, default=1.0,
                        help="Drive amplitude")
    parser.add_argument("--drive-freq", type=float, default=1.0,
                        help="Drive frequency")
    
    # Acceptance mode
    parser.add_argument("--accept-mode", type=str, default="final",
                        choices=["final", "window_max", "window_mean"],
                        help="Acceptance mode")
    
    # Accept windows (can specify multiple)
    parser.add_argument("--accept-windows", type=float, nargs="+", default=[1.0],
                        help="Accept window sizes (Δ)")
    parser.add_argument("--accept-window", type=float, default=None,
                        help="Single accept window (deprecated, use --accept-windows)")
    
    # Threshold sweep parameters
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Comma-separated list of thresholds (overrides min/max/steps)")
    parser.add_argument("--threshold-min", type=float, default=0.60,
                        help="Minimum threshold")
    parser.add_argument("--threshold-max", type=float, default=0.80,
                        help="Maximum threshold")
    parser.add_argument("--threshold-steps", type=int, default=11,
                        help="Number of threshold steps")
    
    # Gamma phi sweep parameters
    parser.add_argument("--gamma-phis", type=str, default=None,
                        help="Comma-separated list of gamma_phis (overrides min/max)")
    parser.add_argument("--gamma-phi-min", type=float, default=0.005,
                        help="Minimum gamma_phi")
    parser.add_argument("--gamma-phi-max", type=float, default=0.10,
                        help="Maximum gamma_phi")
    parser.add_argument("--gamma-phi-steps", type=int, default=5,
                        help="Number of gamma_phi steps")
    
    # Repeats
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of repeats per grid point")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed")
    
    # Execution options
    parser.add_argument("--save-timeseries", action="store_true",
                        help="Save per-run time series parquet files")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Max parallel workers (reserved for future use)")
    
    args = parser.parse_args()
    
    # Build threshold list
    if args.thresholds:
        thresholds = sorted(parse_float_list(args.thresholds))
    else:
        thresholds = list(np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps))
    
    # Build gamma_phi list
    if args.gamma_phis:
        gamma_phis = sorted(parse_float_list(args.gamma_phis))
    else:
        gamma_phis = list(np.linspace(args.gamma_phi_min, args.gamma_phi_max, args.gamma_phi_steps))
    
    # Build accept_windows list
    if args.accept_window is not None:
        accept_windows = [args.accept_window]
    else:
        accept_windows = sorted(args.accept_windows)
    
    # For "final" mode, window doesn't matter but we still iterate
    if args.accept_mode == "final":
        accept_windows = [0.0]  # Placeholder
    
    # Create sweep directory
    sweep_dir, sweep_id = make_sweep_dir(args.out_dir, "window_acceptance")
    
    # Compute total runs
    total_runs = len(gamma_phis) * len(thresholds) * len(accept_windows) * args.repeats
    
    print(f"Sweep directory: {sweep_dir}")
    print(f"Sweep ID: {sweep_id}")
    print(f"Accept mode: {args.accept_mode}")
    print(f"Accept windows: {accept_windows}")
    print(f"Thresholds: {len(thresholds)} values from {min(thresholds):.2f} to {max(thresholds):.2f}")
    print(f"Gamma phis: {len(gamma_phis)} values from {min(gamma_phis):.4f} to {max(gamma_phis):.4f}")
    print(f"Repeats: {args.repeats}")
    print(f"Total runs: {total_runs}")
    print()
    
    # Save sweep configuration
    config = {
        "sweep_id": sweep_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit_hash(),
        "accept_mode": args.accept_mode,
        "accept_windows": accept_windows,
        "thresholds": thresholds,
        "gamma_phis": gamma_phis,
        "repeats": args.repeats,
        "base_seed": args.seed,
        "t_max": args.t_max,
        "n_steps": args.n_steps,
        "drive_amp": args.drive_amp,
        "drive_freq": args.drive_freq,
        "total_runs": total_runs,
        "args": {k: v for k, v in vars(args).items()},
    }
    safe_write_json(sweep_dir / "sweep_config.json", config)
    
    # Build grid
    grid = list(product(gamma_phis, thresholds, accept_windows, range(args.repeats)))
    
    # Run sweep
    results = []
    run_index = 0
    
    for gphi, thr, win, rep in tqdm(grid, desc="Running sweep", unit="run"):
        seed = args.seed + run_index
        run_index += 1
        
        row = run_single(
            gamma_phi=gphi,
            threshold=thr,
            accept_window=win,
            accept_mode=args.accept_mode,
            drive_amp=args.drive_amp,
            drive_freq=args.drive_freq,
            t_max=args.t_max,
            n_steps=args.n_steps,
            seed=seed,
            repeat=rep,
            sweep_dir=sweep_dir,
            sweep_id=sweep_id,
            save_timeseries=args.save_timeseries
        )
        results.append(row)
    
    print()
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns
    priority_cols = [
        "run_id", "timestamp", "seed", "gamma_phi", "accept_threshold",
        "accept_mode", "accept_window", "accepted", "final_fidelity", "peak_fidelity",
        "t_final", "window_start", "window_end", "status", "error"
    ]
    other_cols = [c for c in df.columns if c not in priority_cols]
    col_order = [c for c in priority_cols if c in df.columns] + other_cols
    df = df[col_order]
    
    # Save results
    df.to_parquet(sweep_dir / "results.parquet", index=False)
    df.to_csv(sweep_dir / "results.csv", index=False)
    print(f"Saved results.parquet and results.csv ({len(df)} rows)")
    
    # Compute and save summary statistics
    stats = compute_summary_stats(df)
    safe_write_json(sweep_dir / "summary_stats.json", stats)
    print(f"Saved summary_stats.json")
    
    # Print summary
    print()
    print("=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)
    print(f"Total runs:    {stats['total_runs']}")
    print(f"OK runs:       {stats['ok_runs']}")
    print(f"Error runs:    {stats['error_runs']}")
    if stats.get("acceptance_rate_overall") is not None:
        print(f"Accept rate:   {stats['acceptance_rate_overall']:.1%}")
        print(f"Mean final fidelity: {stats['final_fidelity']['mean']:.4f}")
        print(f"Mean peak fidelity:  {stats['peak_fidelity']['mean']:.4f}")
    print()
    
    # Generate heatmaps
    print("Generating heatmaps...")
    generate_heatmaps(
        df, gamma_phis, thresholds, accept_windows,
        sweep_dir / "figures", args.accept_mode
    )
    print(f"Saved heatmaps to {sweep_dir / 'figures'}")
    
    # Write README
    write_readme(sweep_dir, config, stats)
    print(f"Saved README.md")
    
    print()
    print(f"Sweep complete: {sweep_dir}")


if __name__ == "__main__":
    main()
