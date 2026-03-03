#!/usr/bin/env python3
"""
Comparison Analysis Script

Compares baseline overnight sweep results with follow-up sweep results
to demonstrate the effectiveness of hierarchical conditioning strategies.

This script:
1. Auto-detects the most recent baseline and follow-up sweep folders
2. Loads results from both
3. Produces comparison plots showing global vs hierarchical (all) vs hierarchical (k_of_n)
4. Saves publication-quality figures in PNG and SVG formats

Usage:
    uv run python scripts/analyze_compare.py
    uv run python scripts/analyze_compare.py --baseline runs/high_noise_20260126_233522 --followup runs/followup_XXXXX
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# =============================================================================
# Directory Discovery
# =============================================================================

def find_latest_run(runs_dir: Path, prefix: str) -> Optional[Path]:
    """
    Find the most recent run directory matching a prefix.
    Directories are named like: prefix_YYYYMMDD_HHMMSS
    """
    pattern = re.compile(rf"^{prefix}_(\d{{8}}_\d{{6}})$")
    
    matching_dirs = []
    for d in runs_dir.iterdir():
        if d.is_dir():
            match = pattern.match(d.name)
            if match:
                timestamp = match.group(1)
                matching_dirs.append((timestamp, d))
    
    if not matching_dirs:
        return None
    
    # Sort by timestamp descending
    matching_dirs.sort(key=lambda x: x[0], reverse=True)
    return matching_dirs[0][1]


def load_results(run_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load results.csv and summary_stats.json from a run directory."""
    results_path = run_dir / "results.csv"
    stats_path = run_dir / "summary_stats.json"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    # Load CSV, handling Infinity properly
    df = pd.read_csv(results_path)
    
    # Convert "Infinity" strings back to float inf
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace("Infinity", float('inf'))
            df[col] = df[col].replace("-Infinity", float('-inf'))
            df[col] = df[col].replace("NaN", float('nan'))
    
    # Load stats if available
    stats = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    
    return df, stats


# =============================================================================
# Comparison Plots
# =============================================================================

N_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 4: '#2ca02c', 8: '#d62728', 
            16: '#9467bd', 32: '#8c564b', 64: '#e377c2'}


def plot_acceptance_comparison(
    baseline_df: pd.DataFrame,
    followup_df: pd.DataFrame,
    output_dir: Path,
    gamma: float = 2.0,
    threshold: float = 0.9,
    window: float = 0.5,
) -> None:
    """
    Plot acceptance vs N comparing global, hierarchical (all), hierarchical (k_of_n).
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Extract baseline data (hierarchical "all" rule)
    baseline_subset = baseline_df[
        (baseline_df["gamma"] == gamma) &
        (baseline_df["threshold"] == threshold) &
        (baseline_df["window"] == window)
    ]
    
    # Extract followup data
    followup_subset = followup_df[
        (followup_df["gamma"] == gamma) &
        (followup_df["threshold"] == threshold) &
        (followup_df["window"] == window)
    ]
    
    # Determine hier_rule in followup
    followup_hier_rule = "k_of_n"  # Default assumption
    if "hier_rule" in followup_df.columns:
        rules = followup_df[followup_df["mode"] == "hierarchical"]["hier_rule"].unique()
        if len(rules) > 0 and rules[0] != "N/A":
            followup_hier_rule = rules[0]
    
    # Plot Global from baseline
    global_df = baseline_subset[baseline_subset["mode"] == "global"].sort_values("N")
    if len(global_df) > 0:
        accept = global_df["acceptance_probability"].replace(0, 1e-6)
        ax.semilogy(global_df["N"], accept, 'o-', color='#d62728',
                   linewidth=2, markersize=10, label="Global")
    
    # Plot Hierarchical (all) from baseline
    hier_all_df = baseline_subset[baseline_subset["mode"] == "hierarchical"].sort_values("N")
    if len(hier_all_df) > 0:
        accept = hier_all_df["acceptance_probability"].replace(0, 1e-6)
        ax.semilogy(hier_all_df["N"], accept, 's-', color='#2ca02c',
                   linewidth=2, markersize=10, label="Hierarchical (all)")
    
    # Plot Hierarchical (k_of_n) from followup
    hier_kofn_df = followup_subset[followup_subset["mode"] == "hierarchical"].sort_values("N")
    if len(hier_kofn_df) > 0:
        accept = hier_kofn_df["acceptance_probability"].replace(0, 1e-6)
        
        # Get k_fraction if available
        k_frac = 0.9
        if "k_fraction" in hier_kofn_df.columns:
            k_fracs = hier_kofn_df["k_fraction"].dropna().unique()
            if len(k_fracs) > 0:
                k_frac = k_fracs[0]
        
        label = f"Hierarchical ({followup_hier_rule}, k={k_frac})"
        ax.semilogy(hier_kofn_df["N"], accept, '^--', color='#1f77b4',
                   linewidth=2, markersize=10, label=label)
    
    ax.axhline(y=1e-3, color='gray', linestyle=':', alpha=0.5, 
               label="Practical threshold")
    
    ax.set_xlabel("System Size N", fontsize=12)
    ax.set_ylabel("Acceptance Probability (log)", fontsize=12)
    ax.set_title(f"Acceptance Comparison at γ={gamma}, threshold={threshold}\n"
                 f"Global vs Hierarchical (all) vs Hierarchical (k_of_n)",
                 fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks to all N values present
    all_N = sorted(set(global_df["N"].tolist() if len(global_df) > 0 else []) | 
                   set(hier_kofn_df["N"].tolist() if len(hier_kofn_df) > 0 else []))
    if all_N:
        ax.set_xticks(all_N)
    ax.set_ylim(1e-6, 2)
    
    fig.tight_layout()
    fig.savefig(output_dir / f"comparison_acceptance_gamma_{gamma}.png", dpi=300)
    fig.savefig(output_dir / f"comparison_acceptance_gamma_{gamma}.svg")
    plt.close(fig)
    print(f"  - comparison_acceptance_gamma_{gamma}.png/.svg")


def plot_tts_comparison(
    baseline_df: pd.DataFrame,
    followup_df: pd.DataFrame,
    output_dir: Path,
    gamma: float = 2.0,
    threshold: float = 0.9,
    window: float = 0.5,
) -> None:
    """
    Plot TTS vs N comparing strategies.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    MAX_TTS_DISPLAY = 1e6
    
    baseline_subset = baseline_df[
        (baseline_df["gamma"] == gamma) &
        (baseline_df["threshold"] == threshold) &
        (baseline_df["window"] == window)
    ]
    
    followup_subset = followup_df[
        (followup_df["gamma"] == gamma) &
        (followup_df["threshold"] == threshold) &
        (followup_df["window"] == window)
    ]
    
    followup_hier_rule = "k_of_n"
    if "hier_rule" in followup_df.columns:
        rules = followup_df[followup_df["mode"] == "hierarchical"]["hier_rule"].unique()
        if len(rules) > 0 and rules[0] != "N/A":
            followup_hier_rule = rules[0]
    
    # Plot Global
    global_df = baseline_subset[baseline_subset["mode"] == "global"].sort_values("N")
    if len(global_df) > 0:
        tts = global_df["time_to_solution"].clip(upper=MAX_TTS_DISPLAY)
        ax.semilogy(global_df["N"], tts, 'o-', color='#d62728',
                   linewidth=2, markersize=10, label="Global")
        
        # Mark infinite points
        inf_mask = np.isinf(global_df["time_to_solution"])
        if inf_mask.any():
            inf_N = global_df["N"].values[inf_mask]
            ax.scatter(inf_N, [MAX_TTS_DISPLAY] * len(inf_N), 
                      marker='x', s=150, color='#d62728', zorder=5)
    
    # Plot Hierarchical (all)
    hier_all_df = baseline_subset[baseline_subset["mode"] == "hierarchical"].sort_values("N")
    if len(hier_all_df) > 0:
        tts = hier_all_df["time_to_solution"].clip(upper=MAX_TTS_DISPLAY)
        ax.semilogy(hier_all_df["N"], tts, 's-', color='#2ca02c',
                   linewidth=2, markersize=10, label="Hierarchical (all)")
        
        inf_mask = np.isinf(hier_all_df["time_to_solution"])
        if inf_mask.any():
            inf_N = hier_all_df["N"].values[inf_mask]
            ax.scatter(inf_N, [MAX_TTS_DISPLAY] * len(inf_N), 
                      marker='x', s=150, color='#2ca02c', zorder=5)
    
    # Plot Hierarchical (k_of_n)
    hier_kofn_df = followup_subset[followup_subset["mode"] == "hierarchical"].sort_values("N")
    if len(hier_kofn_df) > 0:
        tts = hier_kofn_df["time_to_solution"].clip(upper=MAX_TTS_DISPLAY)
        
        k_frac = 0.9
        if "k_fraction" in hier_kofn_df.columns:
            k_fracs = hier_kofn_df["k_fraction"].dropna().unique()
            if len(k_fracs) > 0:
                k_frac = k_fracs[0]
        
        label = f"Hierarchical ({followup_hier_rule}, k={k_frac})"
        ax.semilogy(hier_kofn_df["N"], tts, '^--', color='#1f77b4',
                   linewidth=2, markersize=10, label=label)
        
        inf_mask = np.isinf(hier_kofn_df["time_to_solution"])
        if inf_mask.any():
            inf_N = hier_kofn_df["N"].values[inf_mask]
            ax.scatter(inf_N, [MAX_TTS_DISPLAY] * len(inf_N), 
                      marker='x', s=150, color='#1f77b4', zorder=5)
    
    ax.axhline(y=MAX_TTS_DISPLAY, color='gray', linestyle='--', alpha=0.3)
    ax.text(ax.get_xlim()[1] * 0.95, MAX_TTS_DISPLAY * 1.2, "TTS → ∞",
           fontsize=10, color='gray', ha='right')
    
    ax.set_xlabel("System Size N", fontsize=12)
    ax.set_ylabel("Time-to-Solution (log)", fontsize=12)
    ax.set_title(f"TTS Comparison at γ={gamma}, threshold={threshold}\n"
                 f"Lower is better; × marks infinite TTS",
                 fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    all_N = sorted(set(global_df["N"].tolist() if len(global_df) > 0 else []) | 
                   set(hier_kofn_df["N"].tolist() if len(hier_kofn_df) > 0 else []))
    if all_N:
        ax.set_xticks(all_N)
    
    fig.tight_layout()
    fig.savefig(output_dir / f"comparison_tts_gamma_{gamma}.png", dpi=300)
    fig.savefig(output_dir / f"comparison_tts_gamma_{gamma}.svg")
    plt.close(fig)
    print(f"  - comparison_tts_gamma_{gamma}.png/.svg")


def plot_multi_gamma_comparison(
    baseline_df: pd.DataFrame,
    followup_df: pd.DataFrame,
    output_dir: Path,
    threshold: float = 0.9,
    window: float = 0.5,
) -> None:
    """
    Create a 2x2 plot comparing acceptance and TTS for multiple gamma values.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    gammas = [2.0, 5.0]
    MAX_TTS_DISPLAY = 1e6
    
    for col, gamma in enumerate(gammas):
        baseline_subset = baseline_df[
            (baseline_df["gamma"] == gamma) &
            (baseline_df["threshold"] == threshold) &
            (baseline_df["window"] == window)
        ]
        
        followup_subset = followup_df[
            (followup_df["gamma"] == gamma) &
            (followup_df["threshold"] == threshold) &
            (followup_df["window"] == window)
        ]
        
        followup_hier_rule = "k_of_n"
        k_frac = 0.9
        if "hier_rule" in followup_df.columns:
            rules = followup_df[followup_df["mode"] == "hierarchical"]["hier_rule"].unique()
            if len(rules) > 0 and rules[0] != "N/A":
                followup_hier_rule = rules[0]
        if "k_fraction" in followup_df.columns:
            k_fracs = followup_df[followup_df["mode"] == "hierarchical"]["k_fraction"].dropna().unique()
            if len(k_fracs) > 0:
                k_frac = k_fracs[0]
        
        # Row 0: Acceptance
        ax = axes[0, col]
        
        global_df = baseline_subset[baseline_subset["mode"] == "global"].sort_values("N")
        if len(global_df) > 0:
            accept = global_df["acceptance_probability"].replace(0, 1e-6)
            ax.semilogy(global_df["N"], accept, 'o-', color='#d62728',
                       linewidth=2, markersize=8, label="Global")
        
        hier_all_df = baseline_subset[baseline_subset["mode"] == "hierarchical"].sort_values("N")
        if len(hier_all_df) > 0:
            accept = hier_all_df["acceptance_probability"].replace(0, 1e-6)
            ax.semilogy(hier_all_df["N"], accept, 's-', color='#2ca02c',
                       linewidth=2, markersize=8, label="Hierarchical (all)")
        
        hier_kofn_df = followup_subset[followup_subset["mode"] == "hierarchical"].sort_values("N")
        if len(hier_kofn_df) > 0:
            accept = hier_kofn_df["acceptance_probability"].replace(0, 1e-6)
            ax.semilogy(hier_kofn_df["N"], accept, '^--', color='#1f77b4',
                       linewidth=2, markersize=8, label=f"Hier ({followup_hier_rule})")
        
        ax.set_xlabel("System Size N")
        ax.set_ylabel("Acceptance Probability (log)")
        ax.set_title(f"Acceptance at γ={gamma}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1e-6, 2)
        
        # Row 1: TTS
        ax = axes[1, col]
        
        if len(global_df) > 0:
            tts = global_df["time_to_solution"].clip(upper=MAX_TTS_DISPLAY)
            ax.semilogy(global_df["N"], tts, 'o-', color='#d62728',
                       linewidth=2, markersize=8, label="Global")
        
        if len(hier_all_df) > 0:
            tts = hier_all_df["time_to_solution"].clip(upper=MAX_TTS_DISPLAY)
            ax.semilogy(hier_all_df["N"], tts, 's-', color='#2ca02c',
                       linewidth=2, markersize=8, label="Hierarchical (all)")
        
        if len(hier_kofn_df) > 0:
            tts = hier_kofn_df["time_to_solution"].clip(upper=MAX_TTS_DISPLAY)
            ax.semilogy(hier_kofn_df["N"], tts, '^--', color='#1f77b4',
                       linewidth=2, markersize=8, label=f"Hier ({followup_hier_rule})")
        
        ax.axhline(y=MAX_TTS_DISPLAY, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel("System Size N")
        ax.set_ylabel("Time-to-Solution (log)")
        ax.set_title(f"TTS at γ={gamma}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Baseline vs Follow-up Comparison (threshold={threshold}, window={window})",
                fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_multi_gamma.png", dpi=300)
    fig.savefig(output_dir / "comparison_multi_gamma.svg")
    plt.close(fig)
    print("  - comparison_multi_gamma.png/.svg")


def print_comparison_table(
    baseline_df: pd.DataFrame,
    followup_df: pd.DataFrame,
    threshold: float = 0.9,
    window: float = 0.5,
) -> None:
    """Print comparison table to console."""
    
    print("\n" + "=" * 100)
    print("COMPARISON TABLE: Global vs Hierarchical (all) vs Hierarchical (k_of_n)")
    print(f"(threshold={threshold}, window={window})")
    print("=" * 100)
    
    # Get k_fraction from followup
    k_frac = 0.9
    if "k_fraction" in followup_df.columns:
        k_fracs = followup_df[followup_df["mode"] == "hierarchical"]["k_fraction"].dropna().unique()
        if len(k_fracs) > 0:
            k_frac = k_fracs[0]
    
    print(f"\n{'N':>4} | {'γ':>5} | {'Global':>10} | {'Hier(all)':>10} | {'Hier(k={:.1f})':>12} | {'Adv(all)':>10} | {'Adv(k_of_n)':>12}".format(k_frac))
    print("-" * 100)
    
    gammas = sorted(set(baseline_df["gamma"].unique()) & set(followup_df["gamma"].unique()))
    all_N = sorted(set(baseline_df["N"].unique()) | set(followup_df["N"].unique()))
    
    for gamma in gammas:
        for N in all_N:
            # Get baseline global
            g_base = baseline_df[
                (baseline_df["N"] == N) & 
                (baseline_df["gamma"] == gamma) & 
                (baseline_df["threshold"] == threshold) &
                (baseline_df["window"] == window) &
                (baseline_df["mode"] == "global")
            ]
            
            # Get baseline hierarchical (all)
            h_all = baseline_df[
                (baseline_df["N"] == N) & 
                (baseline_df["gamma"] == gamma) & 
                (baseline_df["threshold"] == threshold) &
                (baseline_df["window"] == window) &
                (baseline_df["mode"] == "hierarchical")
            ]
            
            # Get followup hierarchical (k_of_n)
            h_kofn = followup_df[
                (followup_df["N"] == N) & 
                (followup_df["gamma"] == gamma) & 
                (followup_df["threshold"] == threshold) &
                (followup_df["window"] == window) &
                (followup_df["mode"] == "hierarchical")
            ]
            
            g_acc = g_base["acceptance_probability"].iloc[0] if len(g_base) > 0 else float('nan')
            h_all_acc = h_all["acceptance_probability"].iloc[0] if len(h_all) > 0 else float('nan')
            h_kofn_acc = h_kofn["acceptance_probability"].iloc[0] if len(h_kofn) > 0 else float('nan')
            
            # Format acceptance
            g_str = f"{g_acc:.4f}" if not math.isnan(g_acc) else "N/A"
            h_all_str = f"{h_all_acc:.4f}" if not math.isnan(h_all_acc) else "N/A"
            h_kofn_str = f"{h_kofn_acc:.4f}" if not math.isnan(h_kofn_acc) else "N/A"
            
            # Compute advantages
            if g_acc > 0 and h_all_acc > 0:
                adv_all = f"{h_all_acc / g_acc:.1f}×"
            elif g_acc == 0 and h_all_acc > 0:
                adv_all = "∞"
            else:
                adv_all = "N/A"
            
            if g_acc > 0 and h_kofn_acc > 0:
                adv_kofn = f"{h_kofn_acc / g_acc:.1f}×"
            elif g_acc == 0 and h_kofn_acc > 0:
                adv_kofn = "∞"
            else:
                adv_kofn = "N/A"
            
            print(f"{N:>4} | {gamma:>5.1f} | {g_str:>10} | {h_all_str:>10} | {h_kofn_str:>12} | {adv_all:>10} | {adv_kofn:>12}")
        
        print("-" * 100)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline and follow-up sweep results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to baseline run directory (auto-detected if not provided)")
    parser.add_argument("--followup", type=str, default=None,
                        help="Path to follow-up run directory (auto-detected if not provided)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for comparison plots")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Threshold to use for comparison plots")
    parser.add_argument("--window", type=float, default=0.5,
                        help="Window to use for comparison plots")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Primary gamma value for single-gamma plots")
    
    args = parser.parse_args()
    
    runs_dir = Path("runs")
    
    # Find baseline
    if args.baseline:
        baseline_dir = Path(args.baseline)
    else:
        baseline_dir = find_latest_run(runs_dir, "high_noise")
        if not baseline_dir:
            print("ERROR: Could not find baseline run directory (high_noise_*)")
            sys.exit(1)
    
    # Find followup
    if args.followup:
        followup_dir = Path(args.followup)
    else:
        followup_dir = find_latest_run(runs_dir, "followup")
        if not followup_dir:
            print("ERROR: Could not find follow-up run directory (followup_*)")
            sys.exit(1)
    
    print("=" * 80)
    print("COMPARISON ANALYSIS")
    print("=" * 80)
    print(f"Baseline: {baseline_dir}")
    print(f"Followup: {followup_dir}")
    print()
    
    # Load data
    print("Loading baseline results...")
    baseline_df, baseline_stats = load_results(baseline_dir)
    print(f"  - {len(baseline_df)} configurations")
    
    print("Loading follow-up results...")
    followup_df, followup_stats = load_results(followup_dir)
    print(f"  - {len(followup_df)} configurations")
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = followup_dir / "comparison"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    
    # Find common gammas
    common_gammas = sorted(set(baseline_df["gamma"].unique()) & set(followup_df["gamma"].unique()))
    
    for gamma in common_gammas:
        plot_acceptance_comparison(
            baseline_df, followup_df, output_dir,
            gamma=gamma, threshold=args.threshold, window=args.window
        )
        plot_tts_comparison(
            baseline_df, followup_df, output_dir,
            gamma=gamma, threshold=args.threshold, window=args.window
        )
    
    # Multi-gamma comparison
    plot_multi_gamma_comparison(
        baseline_df, followup_df, output_dir,
        threshold=args.threshold, window=args.window
    )
    
    # Print comparison table
    print_comparison_table(
        baseline_df, followup_df,
        threshold=args.threshold, window=args.window
    )
    
    print(f"\nAll comparison plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
