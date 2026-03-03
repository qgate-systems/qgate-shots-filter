#!/usr/bin/env python3
"""
Incremental Sweep Runner - Extended Validation

This script runs simulations incrementally by:
1. Loading existing results from a prior sweep
2. Identifying missing (N, gamma, threshold, window, mode, hier_rule, k_fraction) configurations
3. Running ONLY the missing configurations
4. Merging new results into existing CSV/JSON files
5. Regenerating plots with annotations for patent appendix

Key Features:
- Deterministic seeding per configuration (reproducibility)
- Logs skipped configurations for audit trail
- Annotates plots with "Extended validation" and "Data reused from prior sweep"
- Emphasizes scalability trends for patent documentation

Usage:
    # Extend to N=64 with k_fraction variants:
    uv run python scripts/run_incremental.py \
        --base-results runs/followup_20260127_133350 \
        --system-sizes 64 \
        --gamma-values 2.0,5.0 \
        --thresholds 0.68 \
        --windows 0.5 \
        --k-fractions 0.8,0.9 \
        --trials 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sim import run_simulation, compute_acceptance


# =============================================================================
# Configuration Types
# =============================================================================

@dataclass
class ConfigKey:
    """Unique key for a configuration (used to identify duplicates)."""
    N: int
    gamma: float
    threshold: float
    window: float
    mode: str
    hier_rule: str
    k_fraction: float
    
    def __hash__(self):
        # Use rounded values for float comparison
        return hash((
            self.N, 
            round(self.gamma, 6), 
            round(self.threshold, 6), 
            round(self.window, 6), 
            self.mode, 
            self.hier_rule,
            round(self.k_fraction, 6) if not math.isnan(self.k_fraction) else "nan"
        ))
    
    def __eq__(self, other):
        if not isinstance(other, ConfigKey):
            return False
        return (
            self.N == other.N and
            abs(self.gamma - other.gamma) < 1e-6 and
            abs(self.threshold - other.threshold) < 1e-6 and
            abs(self.window - other.window) < 1e-6 and
            self.mode == other.mode and
            self.hier_rule == other.hier_rule and
            (
                (math.isnan(self.k_fraction) and math.isnan(other.k_fraction)) or
                (not math.isnan(self.k_fraction) and not math.isnan(other.k_fraction) and 
                 abs(self.k_fraction - other.k_fraction) < 1e-6)
            )
        )


@dataclass
class SimulationConfig:
    """Physics simulation parameters."""
    drive_amp: float = 1.0
    drive_freq: float = 1.0
    t_max: float = 12.0
    n_steps: int = 500


# =============================================================================
# Deterministic Seeding
# =============================================================================

def compute_config_seed(base_seed: int, N: int, gamma: float, threshold: float, 
                        window: float, mode: str, hier_rule: str, k_fraction: float) -> int:
    """
    Compute a deterministic seed from configuration parameters.
    """
    k_str = f"{k_fraction:.6f}" if not math.isnan(k_fraction) else "nan"
    config_str = f"{base_seed}:{N}:{gamma:.6f}:{threshold:.6f}:{window:.6f}:{mode}:{hier_rule}:{k_str}"
    hash_bytes = hashlib.sha256(config_str.encode()).digest()
    seed = int.from_bytes(hash_bytes[:4], byteorder='big')
    return seed


# =============================================================================
# Simulation Core
# =============================================================================

def simulate_n_qubit_trajectory(
    N: int,
    gamma_phi: float,
    config: SimulationConfig,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Simulate N independent qubit trajectories under dephasing noise."""
    subsystem_fidelities = []
    tlist = None
    
    for i in range(N):
        subsystem_seed = seed + i * 10000
        
        df, _ = run_simulation(
            drive_amp=config.drive_amp,
            drive_freq=config.drive_freq,
            gamma_phi=gamma_phi,
            threshold=0.0,
            t_max=config.t_max,
            n_steps=config.n_steps,
            accept_mode="final",
            accept_window=0.0,
            seed=subsystem_seed
        )
        
        if tlist is None:
            tlist = df["t"].values
        
        subsystem_fidelities.append(df["fidelity"].values)
    
    # Global (product) fidelity
    global_fidelity = np.ones_like(tlist)
    for fid in subsystem_fidelities:
        global_fidelity = global_fidelity * fid
    
    return tlist, global_fidelity, subsystem_fidelities


def apply_conditioning(
    tlist: np.ndarray,
    global_fidelity: np.ndarray,
    subsystem_fidelities: List[np.ndarray],
    mode: str,
    threshold: float,
    accept_mode: str,
    accept_window: float,
    hier_rule: str,
    k_fraction: float,
) -> Tuple[bool, float, float, int]:
    """Apply conditioning strategy and return acceptance + metrics."""
    N = len(subsystem_fidelities)
    
    if mode == "global":
        accepted, _, _, _ = compute_acceptance(
            tlist, global_fidelity, threshold, accept_mode, accept_window
        )
        n_passing = N if accepted else 0
    else:
        # HIERARCHICAL: count passing subsystems
        n_passing = 0
        for fid in subsystem_fidelities:
            sub_accepted, _, _, _ = compute_acceptance(
                tlist, fid, threshold, accept_mode, accept_window
            )
            if sub_accepted:
                n_passing += 1
        
        # Compute required passing count
        if hier_rule == "all":
            required = N
        else:  # k_of_n
            required = int(math.ceil(k_fraction * N))
        
        accepted = n_passing >= required
    
    final_fidelity = float(global_fidelity[-1])
    peak_fidelity = float(np.max(global_fidelity))
    
    return accepted, final_fidelity, peak_fidelity, n_passing


def run_trials_for_config(
    N: int,
    gamma: float,
    threshold: float,
    window: float,
    mode: str,
    hier_rule: str,
    k_fraction: float,
    n_trials: int,
    base_seed: int,
    sim_config: SimulationConfig,
) -> Dict[str, Any]:
    """Run trials for a single configuration."""
    
    seed = compute_config_seed(base_seed, N, gamma, threshold, window, mode, hier_rule, k_fraction)
    
    accepted_count = 0
    final_fidelities = []
    passing_counts = []
    
    for trial in range(n_trials):
        trial_seed = seed + trial
        
        tlist, global_fidelity, subsystem_fidelities = simulate_n_qubit_trajectory(
            N=N,
            gamma_phi=gamma,
            config=sim_config,
            seed=trial_seed,
        )
        
        accepted, final_fid, _, n_passing = apply_conditioning(
            tlist, global_fidelity, subsystem_fidelities,
            mode, threshold, "window_max", window,
            hier_rule, k_fraction,
        )
        
        passing_counts.append(n_passing)
        
        if accepted:
            accepted_count += 1
            final_fidelities.append(final_fid)
    
    # Compute statistics
    acceptance_prob = accepted_count / n_trials if n_trials > 0 else 0.0
    
    if accepted_count > 0:
        mean_fid = float(np.mean(final_fidelities))
        median_fid = float(np.median(final_fidelities))
        p10_fid = float(np.percentile(final_fidelities, 10)) if accepted_count >= 10 else float(min(final_fidelities))
    else:
        mean_fid = float('nan')
        median_fid = float('nan')
        p10_fid = float('nan')
    
    tts = 1.0 / acceptance_prob if acceptance_prob > 0 else float('inf')
    mean_passing = float(np.mean(passing_counts))
    
    return {
        "N": N,
        "gamma": gamma,
        "threshold": threshold,
        "window": window,
        "mode": mode,
        "hier_rule": hier_rule,
        "k_fraction": k_fraction if mode == "hierarchical" else float('nan'),
        "n_trials": n_trials,
        "n_accepted": accepted_count,
        "acceptance_probability": acceptance_prob,
        "mean_accepted_final_fidelity": mean_fid,
        "median_accepted_final_fidelity": median_fid,
        "p10_accepted_final_fidelity": p10_fid,
        "time_to_solution": tts,
        "mean_passing_subsystems": mean_passing,
        "undersampled": False,
        "source": "incremental",  # Mark as newly computed
    }


# =============================================================================
# Incremental Loading and Merging
# =============================================================================

def load_existing_results(base_dir: Path) -> Tuple[pd.DataFrame, Set[ConfigKey]]:
    """Load existing results and extract configuration keys."""
    results_path = base_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    df = pd.read_csv(results_path)
    
    # Handle Infinity values
    for col in ["time_to_solution"]:
        if col in df.columns:
            df[col] = df[col].replace("Infinity", float('inf'))
            df[col] = df[col].replace("-Infinity", float('-inf'))
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Add source column if missing
    if "source" not in df.columns:
        df["source"] = "prior_sweep"
    
    # Extract config keys
    existing_keys: Set[ConfigKey] = set()
    for _, row in df.iterrows():
        k_frac = row.get("k_fraction", float('nan'))
        if pd.isna(k_frac):
            k_frac = float('nan')
        
        hier_rule = row.get("hier_rule", "N/A")
        if pd.isna(hier_rule):
            hier_rule = "N/A"
            
        key = ConfigKey(
            N=int(row["N"]),
            gamma=float(row["gamma"]),
            threshold=float(row["threshold"]),
            window=float(row["window"]),
            mode=str(row["mode"]),
            hier_rule=str(hier_rule),
            k_fraction=float(k_frac),
        )
        existing_keys.add(key)
    
    return df, existing_keys


def generate_requested_configs(
    system_sizes: List[int],
    gamma_values: List[float],
    thresholds: List[float],
    windows: List[float],
    k_fractions: List[float],
) -> List[ConfigKey]:
    """Generate all requested configuration keys."""
    configs = []
    
    for N, gamma, threshold, window in product(system_sizes, gamma_values, thresholds, windows):
        # Global mode
        configs.append(ConfigKey(
            N=N, gamma=gamma, threshold=threshold, window=window,
            mode="global", hier_rule="N/A", k_fraction=float('nan')
        ))
        
        # Hierarchical mode with each k_fraction
        for k_frac in k_fractions:
            configs.append(ConfigKey(
                N=N, gamma=gamma, threshold=threshold, window=window,
                mode="hierarchical", hier_rule="k_of_n", k_fraction=k_frac
            ))
    
    return configs


def find_missing_configs(
    requested: List[ConfigKey], 
    existing: Set[ConfigKey]
) -> List[ConfigKey]:
    """Find configurations that are not in the existing set."""
    missing = []
    for cfg in requested:
        if cfg not in existing:
            missing.append(cfg)
    return missing


# =============================================================================
# Plotting with Annotations
# =============================================================================

N_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 4: '#2ca02c', 8: '#d62728', 
            16: '#9467bd', 32: '#8c564b', 64: '#e377c2', 128: '#7f7f7f'}


def plot_scalability_extended(
    df: pd.DataFrame, 
    figures_dir: Path,
    gamma_values: List[float],
    new_system_sizes: List[int],
    annotation_text: str,
) -> None:
    """
    Create scalability plots with annotations showing extended validation.
    Emphasizes scalability trends for patent documentation.
    """
    
    # Get unique k_fractions
    hier_df = df[df["mode"] == "hierarchical"]
    k_fractions = sorted(hier_df["k_fraction"].dropna().unique())
    
    for gamma in gamma_values:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        subset = df[df["gamma"] == gamma]
        if len(subset) == 0:
            continue
        
        # Global conditioning
        g_df = subset[subset["mode"] == "global"].sort_values("N")
        if len(g_df) > 0:
            accept = g_df["acceptance_probability"].replace(0, 1e-7)
            # Mark new vs prior data
            prior_mask = g_df["source"] == "prior_sweep"
            new_mask = g_df["source"] == "incremental"
            
            ax.semilogy(g_df["N"], accept, 'o--', color="#d62728", 
                       linewidth=2, markersize=10, alpha=0.8)
            
            # Highlight new data points
            if new_mask.any():
                ax.scatter(g_df.loc[new_mask, "N"], 
                          accept.loc[new_mask].replace(1e-7, 1e-7),
                          s=200, facecolors='none', edgecolors='#d62728', 
                          linewidths=3, zorder=5, label="Global (new)")
            ax.plot([], [], 'o--', color="#d62728", markersize=10, label="Global")
        
        # Hierarchical with each k_fraction
        colors = ["#2ca02c", "#17becf", "#bcbd22"]
        for idx, k_frac in enumerate(k_fractions):
            h_df = subset[(subset["mode"] == "hierarchical") & 
                         (subset["k_fraction"].round(2) == round(k_frac, 2))].sort_values("N")
            
            if len(h_df) > 0:
                accept = h_df["acceptance_probability"].replace(0, 1e-7)
                color = colors[idx % len(colors)]
                
                ax.semilogy(h_df["N"], accept, 's-', color=color,
                           linewidth=2, markersize=10, alpha=0.8)
                
                # Highlight new data points
                new_mask = h_df["source"] == "incremental"
                if new_mask.any():
                    ax.scatter(h_df.loc[new_mask, "N"], 
                              accept.loc[new_mask],
                              s=200, facecolors='none', edgecolors=color, 
                              linewidths=3, zorder=5)
                
                ax.plot([], [], 's-', color=color, markersize=10, 
                       label=f"Hierarchical (k={k_frac:.1f})")
        
        # Practical threshold line
        ax.axhline(y=1e-3, color='gray', linestyle=':', alpha=0.5)
        ax.text(max(df["N"].unique()) * 0.95, 1.5e-3, "Practical limit", 
               fontsize=9, color='gray', ha='right')
        
        # Annotations for new data
        if new_system_sizes:
            for N in new_system_sizes:
                if N in df["N"].values:
                    ax.axvline(x=N, color='orange', linestyle='--', alpha=0.3)
            ax.text(0.02, 0.98, "⬤ = Extended validation (new data)", 
                   transform=ax.transAxes, fontsize=9, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel("System Size N", fontsize=12)
        ax.set_ylabel("Acceptance Probability (log)", fontsize=12)
        ax.set_title(f"Scalability at γ={gamma}\n{annotation_text}", fontsize=12)
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(df["N"].unique()))
        ax.set_ylim(1e-8, 2)
        
        fig.tight_layout()
        fig.savefig(figures_dir / f"scalability_extended_gamma_{gamma}.png", dpi=300)
        fig.savefig(figures_dir / f"scalability_extended_gamma_{gamma}.svg")
        plt.close(fig)


def plot_summary_table_figure(
    df: pd.DataFrame,
    figures_dir: Path,
    annotation_text: str,
) -> None:
    """Create a summary table as a figure for patent appendix."""
    
    # Filter to main comparison settings
    thresholds = sorted(df["threshold"].unique())
    target_thr = thresholds[len(thresholds)//2] if thresholds else 0.68
    
    windows = sorted(df["window"].unique())
    target_win = 0.5 if 0.5 in windows else windows[0]
    
    subset = df[(df["threshold"] == target_thr) & (df["window"] == target_win)]
    
    # Build comparison table
    rows = []
    for N in sorted(subset["N"].unique()):
        for gamma in sorted(subset["gamma"].unique()):
            row = {"N": N, "γ": gamma}
            
            # Global
            g_df = subset[(subset["N"] == N) & (subset["gamma"] == gamma) & 
                         (subset["mode"] == "global")]
            if len(g_df) > 0:
                row["Global"] = f"{g_df['acceptance_probability'].iloc[0]:.4f}"
                row["G_src"] = g_df["source"].iloc[0]
            else:
                row["Global"] = "N/A"
                row["G_src"] = "N/A"
            
            # Hierarchical variants
            h_df = subset[(subset["N"] == N) & (subset["gamma"] == gamma) & 
                         (subset["mode"] == "hierarchical")]
            for _, h_row in h_df.iterrows():
                k = h_row["k_fraction"]
                if not math.isnan(k):
                    col_name = f"Hier(k={k:.1f})"
                    row[col_name] = f"{h_row['acceptance_probability']:.4f}"
                    row[f"H{k:.1f}_src"] = h_row["source"]
            
            rows.append(row)
    
    table_df = pd.DataFrame(rows)
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=(14, len(rows) * 0.4 + 2))
    ax.axis('off')
    
    # Prepare table data
    display_cols = ["N", "γ", "Global"] + [c for c in table_df.columns 
                                           if c.startswith("Hier(")]
    table_data = table_df[display_cols].values
    
    table = ax.table(
        cellText=table_data,
        colLabels=display_cols,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header
    for j, col in enumerate(display_cols):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', weight='bold')
    
    # Highlight rows with new data
    for i, row in enumerate(rows):
        is_new = row.get("G_src") == "incremental" or any(
            row.get(f"H{k:.1f}_src") == "incremental" 
            for k in [0.8, 0.9, 1.0]
        )
        if is_new:
            for j in range(len(display_cols)):
                table[(i + 1, j)].set_facecolor('#FFF2CC')
    
    ax.set_title(f"Acceptance Probability Summary\n{annotation_text}\n"
                f"(threshold={target_thr}, window={target_win})\n"
                "Yellow = New data from incremental run", 
                fontsize=12, pad=20)
    
    fig.tight_layout()
    fig.savefig(figures_dir / "summary_table.png", dpi=300, bbox_inches='tight')
    fig.savefig(figures_dir / "summary_table.svg", bbox_inches='tight')
    plt.close(fig)


def plot_patent_scalability_figure(
    df: pd.DataFrame,
    figures_dir: Path,
    annotation_text: str,
) -> None:
    """
    Create the KEY patent figure showing scalability advantage.
    This is designed for inclusion in patent appendix.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Acceptance vs N for all gammas (stacked)
    ax = axes[0]
    
    k_fractions = sorted(df[df["mode"] == "hierarchical"]["k_fraction"].dropna().unique())
    primary_k = k_fractions[-1] if k_fractions else 0.9  # Use highest k_fraction
    
    gamma_values = sorted(df["gamma"].unique())
    gamma_colors = {g: c for g, c in zip(gamma_values, plt.cm.viridis(np.linspace(0.2, 0.8, len(gamma_values))))}
    
    for gamma in gamma_values:
        g_df = df[(df["gamma"] == gamma) & (df["mode"] == "global")].sort_values("N")
        h_df = df[(df["gamma"] == gamma) & (df["mode"] == "hierarchical") & 
                 (df["k_fraction"].round(2) == round(primary_k, 2))].sort_values("N")
        
        color = gamma_colors[gamma]
        
        if len(g_df) > 0:
            accept = g_df["acceptance_probability"].replace(0, 1e-8)
            ax.semilogy(g_df["N"], accept, 'o--', color=color, alpha=0.6, markersize=6)
        
        if len(h_df) > 0:
            accept = h_df["acceptance_probability"].replace(0, 1e-8)
            ax.semilogy(h_df["N"], accept, 's-', color=color, markersize=8,
                       label=f"γ={gamma}")
    
    ax.set_xlabel("System Size N", fontsize=12)
    ax.set_ylabel("Acceptance Probability", fontsize=12)
    ax.set_title(f"(a) Scalability: Global (○) vs Hierarchical (□)\n"
                f"Hierarchical rule: k_of_n (k={primary_k})", fontsize=11)
    ax.legend(title="Noise Rate", fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(df["N"].unique()))
    
    # Right: Advantage ratio (Hier/Global) vs N
    ax = axes[1]
    
    for gamma in gamma_values:
        ratios = []
        Ns = []
        for N in sorted(df["N"].unique()):
            g_df = df[(df["N"] == N) & (df["gamma"] == gamma) & (df["mode"] == "global")]
            h_df = df[(df["N"] == N) & (df["gamma"] == gamma) & (df["mode"] == "hierarchical") &
                     (df["k_fraction"].round(2) == round(primary_k, 2))]
            
            if len(g_df) > 0 and len(h_df) > 0:
                g_acc = g_df["acceptance_probability"].iloc[0]
                h_acc = h_df["acceptance_probability"].iloc[0]
                
                if g_acc > 0:
                    ratios.append(h_acc / g_acc)
                elif h_acc > 0:
                    ratios.append(1e6)  # Infinite advantage
                else:
                    continue
                Ns.append(N)
        
        if ratios:
            ax.semilogy(Ns, ratios, 'o-', color=gamma_colors[gamma], 
                       markersize=8, linewidth=2, label=f"γ={gamma}")
    
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.text(max(df["N"].unique()) * 0.95, 1.3, "Equal performance", 
           fontsize=9, color='gray', ha='right')
    
    ax.set_xlabel("System Size N", fontsize=12)
    ax.set_ylabel("Advantage Ratio (Hier/Global)", fontsize=12)
    ax.set_title("(b) Hierarchical Advantage vs System Size\n"
                "(Values > 1 favor hierarchical)", fontsize=11)
    ax.legend(title="Noise Rate", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(df["N"].unique()))
    
    # Add annotation box
    fig.text(0.5, 0.02, annotation_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(figures_dir / "patent_scalability_figure.png", dpi=300)
    fig.savefig(figures_dir / "patent_scalability_figure.svg")
    plt.close(fig)


# =============================================================================
# Summary Statistics
# =============================================================================

def compute_merged_summary(
    df: pd.DataFrame, 
    n_skipped: int,
    n_new: int,
    skipped_configs: List[str],
) -> Dict[str, Any]:
    """Compute summary statistics for merged results."""
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "merge_info": {
            "configurations_reused": n_skipped,
            "configurations_added": n_new,
            "total_configurations": len(df),
            "skipped_config_keys": skipped_configs[:20],  # First 20 for brevity
        },
        "total_trials_run": int(df["n_trials"].sum()),
    }
    
    # Per-N statistics
    per_n_stats = {}
    for N in sorted(df["N"].unique()):
        n_df = df[df["N"] == N]
        per_n_stats[str(N)] = {
            "global_mean_acceptance": float(n_df[n_df["mode"] == "global"]["acceptance_probability"].mean()),
            "hier_mean_acceptance": float(n_df[n_df["mode"] == "hierarchical"]["acceptance_probability"].mean()),
            "configs_count": len(n_df),
        }
    
    summary["per_N_statistics"] = per_n_stats
    
    # Mode comparison
    global_df = df[df["mode"] == "global"]
    hier_df = df[df["mode"] == "hierarchical"]
    
    summary["mode_comparison"] = {
        "global": {
            "mean_acceptance": float(global_df["acceptance_probability"].mean()) if len(global_df) > 0 else 0,
            "configs_with_zero_acceptance": int((global_df["acceptance_probability"] == 0).sum()),
        },
        "hierarchical": {
            "mean_acceptance": float(hier_df["acceptance_probability"].mean()) if len(hier_df) > 0 else 0,
            "configs_with_zero_acceptance": int((hier_df["acceptance_probability"] == 0).sum()),
            "k_fractions_tested": sorted(hier_df["k_fraction"].dropna().unique().tolist()),
        }
    }
    
    return summary


# =============================================================================
# JSON/CSV Helpers
# =============================================================================

def safe_json_serialize(obj):
    """Convert non-JSON-serializable types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return "NaN"
        if np.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
    return obj


def save_json(path: Path, data: Dict) -> None:
    """Save dictionary to JSON."""
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return safe_json_serialize(obj)
    
    with open(path, "w") as f:
        json.dump(convert(data), f, indent=2)


def save_csv(path: Path, df: pd.DataFrame) -> None:
    """Save DataFrame to CSV."""
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=[np.floating]).columns:
        df_copy[col] = df_copy[col].apply(
            lambda x: "Infinity" if (isinstance(x, float) and math.isinf(x) and x > 0)
            else "-Infinity" if (isinstance(x, float) and math.isinf(x) and x < 0)
            else x
        )
    df_copy.to_csv(path, index=False)


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run incremental sweep - only compute missing configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--base-results", type=str, required=True,
                        help="Path to directory with existing results.csv")
    parser.add_argument("--system-sizes", type=str, required=True,
                        help="Comma-separated system sizes to extend to")
    parser.add_argument("--gamma-values", type=str, default="2.0,5.0",
                        help="Comma-separated gamma values")
    parser.add_argument("--thresholds", type=str, default="0.68",
                        help="Comma-separated thresholds")
    parser.add_argument("--windows", type=str, default="0.5",
                        help="Comma-separated windows")
    parser.add_argument("--k-fractions", type=str, default="0.8,0.9",
                        help="Comma-separated k_fraction values for hierarchical rule")
    parser.add_argument("--trials", type=int, default=200,
                        help="Number of trials per configuration")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: extend base-results)")
    
    args = parser.parse_args()
    
    # Parse parameters
    system_sizes = [int(x.strip()) for x in args.system_sizes.split(",")]
    gamma_values = [float(x.strip()) for x in args.gamma_values.split(",")]
    thresholds = [float(x.strip()) for x in args.thresholds.split(",")]
    windows = [float(x.strip()) for x in args.windows.split(",")]
    k_fractions = [float(x.strip()) for x in args.k_fractions.split(",")]
    
    base_dir = Path(args.base_results)
    
    print("=" * 80)
    print("INCREMENTAL SWEEP - EXTENDED VALIDATION")
    print("=" * 80)
    print(f"Base results: {base_dir}")
    print(f"Extending to: N ∈ {system_sizes}")
    print(f"Gamma values: {gamma_values}")
    print(f"Thresholds: {thresholds}")
    print(f"Windows: {windows}")
    print(f"k_fractions: {k_fractions}")
    print(f"Trials per config: {args.trials}")
    print()
    
    # Load existing results
    print("Loading existing results...")
    existing_df, existing_keys = load_existing_results(base_dir)
    print(f"  Found {len(existing_df)} existing configurations")
    print(f"  Unique config keys: {len(existing_keys)}")
    
    # Generate requested configs
    requested = generate_requested_configs(
        system_sizes, gamma_values, thresholds, windows, k_fractions
    )
    print(f"\nRequested configurations: {len(requested)}")
    
    # Find missing
    missing = find_missing_configs(requested, existing_keys)
    print(f"Missing (to compute): {len(missing)}")
    
    # Log skipped configs
    skipped = [cfg for cfg in requested if cfg in existing_keys]
    skipped_strs = [f"N={c.N},γ={c.gamma},θ={c.threshold},w={c.window},{c.mode},k={c.k_fraction}" 
                   for c in skipped]
    
    if skipped:
        print(f"\nSkipping {len(skipped)} configurations (already exist):")
        for s in skipped_strs[:5]:
            print(f"  - {s}")
        if len(skipped_strs) > 5:
            print(f"  ... and {len(skipped_strs) - 5} more")
    
    # Setup output
    if args.out_dir:
        output_dir = Path(args.out_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = base_dir / f"incremental_{ts}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Run missing configurations
    if missing:
        print(f"\n{'=' * 60}")
        print("RUNNING MISSING CONFIGURATIONS")
        print('=' * 60)
        
        sim_config = SimulationConfig()
        new_results = []
        
        for cfg in tqdm(missing, desc="Computing", unit="config"):
            result = run_trials_for_config(
                N=cfg.N,
                gamma=cfg.gamma,
                threshold=cfg.threshold,
                window=cfg.window,
                mode=cfg.mode,
                hier_rule=cfg.hier_rule,
                k_fraction=cfg.k_fraction,
                n_trials=args.trials,
                base_seed=args.seed,
                sim_config=sim_config,
            )
            new_results.append(result)
        
        new_df = pd.DataFrame(new_results)
        print(f"\nComputed {len(new_df)} new configurations")
        
        # Merge with existing
        merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        print("\nNo missing configurations - using existing data only")
        merged_df = existing_df.copy()
        new_df = pd.DataFrame()
    
    # Save merged results
    save_csv(output_dir / "results_merged.csv", merged_df)
    print(f"Saved results_merged.csv ({len(merged_df)} configurations)")
    
    # Save summary
    summary = compute_merged_summary(
        merged_df, 
        n_skipped=len(skipped),
        n_new=len(missing),
        skipped_configs=skipped_strs,
    )
    save_json(output_dir / "summary_stats.json", summary)
    print("Saved summary_stats.json")
    
    # Save log of what was done
    run_log = {
        "timestamp": datetime.now().isoformat(),
        "base_results": str(base_dir),
        "parameters": {
            "system_sizes": system_sizes,
            "gamma_values": gamma_values,
            "thresholds": thresholds,
            "windows": windows,
            "k_fractions": k_fractions,
            "trials": args.trials,
            "seed": args.seed,
        },
        "existing_configs": len(existing_keys),
        "requested_configs": len(requested),
        "skipped_configs": len(skipped),
        "computed_configs": len(missing),
        "total_merged": len(merged_df),
    }
    save_json(output_dir / "run_log.json", run_log)
    
    # Generate annotated plots
    print("\nGenerating annotated plots...")
    
    annotation = f"Extended validation (incremental run)\nData reused from prior sweep: {len(skipped)} configs"
    
    plot_scalability_extended(
        merged_df, figures_dir, gamma_values, system_sizes, annotation
    )
    print("  - scalability_extended_gamma_*.png/.svg")
    
    plot_summary_table_figure(merged_df, figures_dir, annotation)
    print("  - summary_table.png/.svg")
    
    plot_patent_scalability_figure(merged_df, figures_dir, annotation)
    print("  - patent_scalability_figure.png/.svg")
    
    # Print key findings
    print("\n" + "=" * 90)
    print("KEY FINDINGS - EXTENDED VALIDATION")
    print("=" * 90)
    
    for N in sorted(merged_df["N"].unique()):
        n_df = merged_df[merged_df["N"] == N]
        g_accept = n_df[n_df["mode"] == "global"]["acceptance_probability"].mean()
        h_accept = n_df[n_df["mode"] == "hierarchical"]["acceptance_probability"].mean()
        
        source_tag = ""
        if N in system_sizes and len(new_df) > 0:
            source_tag = " [NEW]"
        
        print(f"N={N:3d}{source_tag}: Global={g_accept:.4f}, Hierarchical={h_accept:.4f}")
    
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nFigures suitable for patent appendix:")
    print(f"  - {figures_dir / 'patent_scalability_figure.png'}")
    print(f"  - {figures_dir / 'summary_table.png'}")


if __name__ == "__main__":
    main()
