#!/usr/bin/env python3
"""
High-Noise Regime Simulation Runner

This script extends the scaling simulations into extreme noise regimes,
with adaptive trial counts to ensure statistical significance even at
low acceptance probabilities.

Key Features:
1. JSON-configurable parameters for reproducibility
2. Adaptive trials: start with N trials, increase if acceptance < threshold
3. Deterministic seeding via hash of configuration
4. Comprehensive outputs: CSV, JSON, publication-quality figures
5. Handles infinite TTS gracefully in all outputs

Patent Relevance:
- Demonstrates hierarchical advantage persists under extreme decoherence
- Quantifies the noise level at which global conditioning becomes infeasible
- Provides reproducible evidence for scalability claims

Usage:
    uv run python scripts/run_high_noise.py
    uv run python scripts/run_high_noise.py --config configs/high_noise_config.json
    uv run python scripts/run_high_noise.py --gamma-values 1.0,2.0,5.0 --n-trials 500
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sim import run_simulation, compute_acceptance


# =============================================================================
# Configuration Management
# =============================================================================

@dataclass
class AdaptiveTrialsConfig:
    """Configuration for adaptive trial adjustment."""
    enabled: bool = True
    initial_trials: int = 200
    max_trials: int = 2000
    low_acceptance_threshold: float = 0.02
    step_multiplier: int = 5


@dataclass
class SimulationConfig:
    """Physics simulation parameters."""
    drive_amp: float = 1.0
    drive_freq: float = 1.0
    t_max: float = 12.0
    n_steps: int = 500


@dataclass
class HighNoiseConfig:
    """Complete configuration for high-noise sweep."""
    
    system_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    gamma_values: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0, 10.0])
    thresholds: List[float] = field(default_factory=lambda: [0.6, 0.7, 0.8])
    windows: List[float] = field(default_factory=lambda: [0.5, 1.0])
    acceptance_mode: str = "window_max"
    
    adaptive_trials: AdaptiveTrialsConfig = field(default_factory=AdaptiveTrialsConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    
    seed: int = 42
    output_base_dir: str = "runs"
    
    @classmethod
    def from_json(cls, path: Path) -> "HighNoiseConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        # Parse nested configs
        adaptive = AdaptiveTrialsConfig(**data.get("adaptive_trials", {}))
        sim = SimulationConfig(**data.get("simulation", {}))
        
        return cls(
            system_sizes=data.get("system_sizes", [1, 2, 4, 8, 16]),
            gamma_values=data.get("gamma_values", [0.5, 1.0, 2.0, 5.0, 10.0]),
            thresholds=data.get("thresholds", [0.6, 0.7, 0.8]),
            windows=data.get("windows", [0.5, 1.0]),
            acceptance_mode=data.get("acceptance_mode", "window_max"),
            adaptive_trials=adaptive,
            simulation=sim,
            seed=data.get("seed", 42),
            output_base_dir=data.get("output", {}).get("base_dir", "runs"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "system_sizes": self.system_sizes,
            "gamma_values": self.gamma_values,
            "thresholds": self.thresholds,
            "windows": self.windows,
            "acceptance_mode": self.acceptance_mode,
            "adaptive_trials": asdict(self.adaptive_trials),
            "simulation": asdict(self.simulation),
            "seed": self.seed,
            "output_base_dir": self.output_base_dir,
        }


# =============================================================================
# Deterministic Seeding
# =============================================================================

def compute_config_seed(base_seed: int, N: int, gamma: float, threshold: float, 
                        window: float, mode: str) -> int:
    """
    Compute a deterministic seed from configuration parameters.
    
    Uses SHA-256 hash to ensure:
    1. Different configs get different seeds
    2. Same config always gets same seed (reproducibility)
    3. Seeds are well-distributed
    """
    config_str = f"{base_seed}:{N}:{gamma:.6f}:{threshold:.6f}:{window:.6f}:{mode}"
    hash_bytes = hashlib.sha256(config_str.encode()).digest()
    # Use first 4 bytes as seed (gives 32-bit int)
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
    """
    Simulate N independent qubit trajectories under dephasing noise.
    
    HIGH-NOISE REGIME PHYSICS:
    At high γ (>> drive amplitude), the qubit rapidly decoheres before
    completing coherent oscillations. This leads to:
    - Suppressed peak fidelity
    - Faster decay to mixed state
    - More pronounced effect of product fidelity collapse for global conditioning
    
    The hierarchical approach benefits because:
    - Individual subsystem tests remain feasible even if global product fails
    - Retry overhead grows polynomially, not exponentially
    """
    subsystem_fidelities = []
    tlist = None
    
    for i in range(N):
        # Each subsystem gets unique but deterministic seed
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
) -> Tuple[bool, float, float]:
    """
    Apply conditioning strategy and return acceptance + fidelity metrics.
    
    Returns:
        (accepted, final_fidelity, peak_fidelity)
    """
    if mode == "global":
        # GLOBAL CONDITIONING:
        # Test the product fidelity F_1 × F_2 × ... × F_N against threshold.
        # This fails catastrophically at high noise because:
        # - Each F_i drops faster under high γ
        # - Product decays as exp(-N × f(γ)) approximately
        # - For N=16, γ=5.0: product ≈ 10^-10 or less
        accepted, _, _, _ = compute_acceptance(
            tlist, global_fidelity, threshold, accept_mode, accept_window
        )
    else:
        # HIERARCHICAL CONDITIONING:
        # Test each subsystem independently against threshold.
        # This maintains viability because:
        # - If single-qubit acceptance is p, overall is p^N
        # - Even at low p (e.g., 0.3), for N=16: 0.3^16 ≈ 4×10^-9
        # - But for moderate p (e.g., 0.7), for N=16: 0.7^16 ≈ 0.003
        # - Key insight: p remains NON-ZERO for much higher γ than product threshold
        accepted = True
        for fid in subsystem_fidelities:
            sub_accepted, _, _, _ = compute_acceptance(
                tlist, fid, threshold, accept_mode, accept_window
            )
            if not sub_accepted:
                accepted = False
                break
    
    final_fidelity = float(global_fidelity[-1])
    peak_fidelity = float(np.max(global_fidelity))
    
    return accepted, final_fidelity, peak_fidelity


def run_trials_for_config(
    N: int,
    gamma: float,
    threshold: float,
    window: float,
    mode: str,
    config: HighNoiseConfig,
) -> Dict[str, Any]:
    """
    Run trials for a single configuration with adaptive trial count.
    
    Adaptive Logic:
    1. Start with initial_trials
    2. If acceptance < low_acceptance_threshold, increase trials
    3. Repeat until max_trials or acceptance becomes estimable
    """
    base_seed = compute_config_seed(config.seed, N, gamma, threshold, window, mode)
    
    trials_run = 0
    accepted_count = 0
    final_fidelities = []
    peak_fidelities = []
    
    adaptive = config.adaptive_trials
    target_trials = adaptive.initial_trials
    
    undersampled = False
    
    while trials_run < target_trials:
        trial_seed = base_seed + trials_run
        
        tlist, global_fidelity, subsystem_fidelities = simulate_n_qubit_trajectory(
            N=N,
            gamma_phi=gamma,
            config=config.simulation,
            seed=trial_seed,
        )
        
        accepted, final_fid, peak_fid = apply_conditioning(
            tlist, global_fidelity, subsystem_fidelities,
            mode, threshold, config.acceptance_mode, window
        )
        
        trials_run += 1
        if accepted:
            accepted_count += 1
            final_fidelities.append(final_fid)
            peak_fidelities.append(peak_fid)
        
        # Check if we need more trials (adaptive)
        if adaptive.enabled and trials_run == target_trials:
            current_accept_rate = accepted_count / trials_run
            if current_accept_rate < adaptive.low_acceptance_threshold:
                if target_trials < adaptive.max_trials:
                    target_trials = min(
                        target_trials * adaptive.step_multiplier,
                        adaptive.max_trials
                    )
    
    # Compute statistics
    n_accepted = accepted_count
    acceptance_prob = n_accepted / trials_run if trials_run > 0 else 0.0
    
    if n_accepted > 0:
        mean_fid = float(np.mean(final_fidelities))
        median_fid = float(np.median(final_fidelities))
        p10_fid = float(np.percentile(final_fidelities, 10)) if n_accepted >= 10 else float(min(final_fidelities))
    else:
        mean_fid = float('nan')
        median_fid = float('nan')
        p10_fid = float('nan')
    
    # Time-to-solution
    if acceptance_prob > 0:
        tts = 1.0 / acceptance_prob
    else:
        tts = float('inf')
    
    # Flag undersampled if 0 accepted with low trials
    if n_accepted == 0 and trials_run < adaptive.max_trials:
        undersampled = True
    
    return {
        "N": N,
        "gamma": gamma,
        "threshold": threshold,
        "window": window,
        "mode": mode,
        "n_trials": trials_run,
        "n_accepted": n_accepted,
        "acceptance_probability": acceptance_prob,
        "mean_accepted_final_fidelity": mean_fid,
        "median_accepted_final_fidelity": median_fid,
        "p10_accepted_final_fidelity": p10_fid,
        "time_to_solution": tts,
        "undersampled": undersampled,
    }


# =============================================================================
# Main Sweep
# =============================================================================

def run_high_noise_sweep(config: HighNoiseConfig) -> pd.DataFrame:
    """Execute the full high-noise sweep with adaptive trials."""
    
    # Build parameter grid
    grid = list(product(
        config.system_sizes,
        config.gamma_values,
        config.thresholds,
        config.windows,
        ["global", "hierarchical"],
    ))
    
    print(f"High-Noise Sweep Configuration")
    print("=" * 60)
    print(f"System sizes: {config.system_sizes}")
    print(f"Gamma values: {config.gamma_values}")
    print(f"Thresholds: {config.thresholds}")
    print(f"Windows: {config.windows}")
    print(f"Adaptive trials: {config.adaptive_trials.initial_trials} → {config.adaptive_trials.max_trials}")
    print(f"Total configurations: {len(grid)}")
    print()
    
    results = []
    
    for N, gamma, threshold, window, mode in tqdm(grid, desc="Sweeping", unit="config"):
        result = run_trials_for_config(N, gamma, threshold, window, mode, config)
        results.append(result)
    
    return pd.DataFrame(results)


# =============================================================================
# Summary Statistics
# =============================================================================

def compute_summary_stats(df: pd.DataFrame, config: HighNoiseConfig) -> Dict[str, Any]:
    """
    Compute comprehensive summary statistics.
    
    Includes:
    - Per-N best settings for each mode
    - Acceptance vs N analysis
    - Undersampled flags
    """
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": config.to_dict(),
        "total_configurations": len(df),
        "total_trials_run": int(df["n_trials"].sum()),
    }
    
    # Per-N best settings for each mode
    best_settings = {}
    for mode in ["global", "hierarchical"]:
        mode_df = df[df["mode"] == mode]
        best_settings[mode] = {}
        
        for N in config.system_sizes:
            n_df = mode_df[mode_df["N"] == N]
            if len(n_df) > 0:
                best_row = n_df.loc[n_df["acceptance_probability"].idxmax()]
                best_settings[mode][str(N)] = {
                    "best_gamma": float(best_row["gamma"]),
                    "best_threshold": float(best_row["threshold"]),
                    "best_window": float(best_row["window"]),
                    "max_acceptance": float(best_row["acceptance_probability"]),
                    "tts_at_best": float(best_row["time_to_solution"]) if not math.isinf(best_row["time_to_solution"]) else "Infinity",
                }
    
    summary["best_settings_per_N"] = best_settings
    
    # Acceptance vs N slope analysis (for fixed gamma/threshold/window)
    slope_analysis = []
    for gamma, threshold, window in product(config.gamma_values, config.thresholds, config.windows):
        for mode in ["global", "hierarchical"]:
            subset = df[
                (df["gamma"] == gamma) & 
                (df["threshold"] == threshold) & 
                (df["window"] == window) &
                (df["mode"] == mode)
            ].sort_values("N")
            
            if len(subset) >= 3:
                # Log-linear fit: log(acceptance) vs N
                valid = subset[subset["acceptance_probability"] > 0]
                if len(valid) >= 2:
                    log_accept = np.log10(valid["acceptance_probability"].values + 1e-10)
                    N_vals = valid["N"].values
                    
                    # Simple linear regression
                    slope, intercept = np.polyfit(N_vals, log_accept, 1)
                    
                    slope_analysis.append({
                        "gamma": gamma,
                        "threshold": threshold,
                        "window": window,
                        "mode": mode,
                        "slope_log_acceptance_vs_N": float(slope),
                        "interpretation": "exponential_decay" if slope < -0.1 else "stable" if slope > -0.01 else "moderate_decay"
                    })
    
    summary["acceptance_vs_N_analysis"] = slope_analysis
    
    # Undersampled configurations
    undersampled = df[df["undersampled"] == True][
        ["N", "gamma", "threshold", "window", "mode", "n_trials", "n_accepted"]
    ].to_dict(orient="records")
    
    summary["undersampled_configurations"] = undersampled
    summary["n_undersampled"] = len(undersampled)
    
    # Mode comparison summary
    global_df = df[df["mode"] == "global"]
    hier_df = df[df["mode"] == "hierarchical"]
    
    summary["mode_comparison"] = {
        "global": {
            "mean_acceptance": float(global_df["acceptance_probability"].mean()),
            "configs_with_zero_acceptance": int((global_df["acceptance_probability"] == 0).sum()),
            "configs_with_finite_tts": int((global_df["time_to_solution"] < float('inf')).sum()),
        },
        "hierarchical": {
            "mean_acceptance": float(hier_df["acceptance_probability"].mean()),
            "configs_with_zero_acceptance": int((hier_df["acceptance_probability"] == 0).sum()),
            "configs_with_finite_tts": int((hier_df["time_to_solution"] < float('inf')).sum()),
        }
    }
    
    return summary


# =============================================================================
# Plotting Functions
# =============================================================================

# Consistent color scheme
N_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 4: '#2ca02c', 8: '#d62728', 16: '#9467bd', 32: '#8c564b'}
GAMMA_COLORS = {0.5: '#1f77b4', 1.0: '#ff7f0e', 2.0: '#2ca02c', 5.0: '#d62728', 10.0: '#9467bd'}


def plot_acceptance_vs_N_per_gamma(df: pd.DataFrame, figures_dir: Path, config: HighNoiseConfig) -> None:
    """
    Create separate acceptance vs N plots for each gamma value.
    Overlays global vs hierarchical for direct comparison.
    """
    # Use first threshold/window as default
    default_thr = config.thresholds[1] if len(config.thresholds) > 1 else config.thresholds[0]
    default_win = config.windows[-1] if len(config.windows) > 0 else 1.0
    
    for gamma in config.gamma_values:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        subset = df[
            (df["gamma"] == gamma) & 
            (df["threshold"] == default_thr) & 
            (df["window"] == default_win)
        ]
        
        for mode, color, marker, label in [
            ("global", "#d62728", "o", "Global"),
            ("hierarchical", "#2ca02c", "s", "Hierarchical")
        ]:
            m_df = subset[subset["mode"] == mode].sort_values("N")
            # Replace 0 with small value for log plot
            accept = m_df["acceptance_probability"].replace(0, 1e-4)
            ax.semilogy(m_df["N"], accept, f"{marker}-", color=color, 
                       linewidth=2, markersize=10, label=label)
        
        ax.axhline(y=1e-3, color='gray', linestyle=':', alpha=0.5, 
                   label="Practical threshold")
        
        ax.set_xlabel("System Size N", fontsize=12)
        ax.set_ylabel("Acceptance Probability (log)", fontsize=12)
        ax.set_title(f"Acceptance vs N at γ={gamma}\n(threshold={default_thr}, window={default_win})",
                    fontsize=12)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(config.system_sizes)
        ax.set_ylim(1e-5, 2)
        
        fig.tight_layout()
        fig.savefig(figures_dir / f"acceptance_vs_N_gamma_{gamma}.png", dpi=300)
        fig.savefig(figures_dir / f"acceptance_vs_N_gamma_{gamma}.svg")
        plt.close(fig)


def plot_tts_vs_N_per_gamma(df: pd.DataFrame, figures_dir: Path, config: HighNoiseConfig) -> None:
    """
    Create TTS vs N plots for each gamma value with log y-axis.
    Handles infinity by clipping and annotating.
    """
    default_thr = config.thresholds[1] if len(config.thresholds) > 1 else config.thresholds[0]
    default_win = config.windows[-1] if len(config.windows) > 0 else 1.0
    
    MAX_TTS_DISPLAY = 1e6  # Clip infinite TTS for display
    
    for gamma in config.gamma_values:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        subset = df[
            (df["gamma"] == gamma) & 
            (df["threshold"] == default_thr) & 
            (df["window"] == default_win)
        ]
        
        for mode, color, marker, label in [
            ("global", "#d62728", "o", "Global"),
            ("hierarchical", "#2ca02c", "s", "Hierarchical")
        ]:
            m_df = subset[subset["mode"] == mode].sort_values("N")
            tts = m_df["time_to_solution"].copy()
            
            # Identify infinite values
            inf_mask = np.isinf(tts)
            tts_clipped = tts.clip(upper=MAX_TTS_DISPLAY)
            
            ax.semilogy(m_df["N"], tts_clipped, f"{marker}-", color=color,
                       linewidth=2, markersize=10, label=label)
            
            # Mark infinite points
            if inf_mask.any():
                inf_N = m_df["N"].values[inf_mask]
                ax.scatter(inf_N, [MAX_TTS_DISPLAY] * len(inf_N), 
                          marker='x', s=150, color=color, zorder=5)
        
        # Add annotation for infinity
        ax.axhline(y=MAX_TTS_DISPLAY, color='gray', linestyle='--', alpha=0.3)
        ax.text(config.system_sizes[-1], MAX_TTS_DISPLAY * 1.2, "TTS → ∞",
               fontsize=10, color='gray', ha='right')
        
        ax.set_xlabel("System Size N", fontsize=12)
        ax.set_ylabel("Time-to-Solution (log)", fontsize=12)
        ax.set_title(f"TTS vs N at γ={gamma}\n(threshold={default_thr}, window={default_win})",
                    fontsize=12)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(config.system_sizes)
        
        fig.tight_layout()
        fig.savefig(figures_dir / f"tts_vs_N_gamma_{gamma}.png", dpi=300)
        fig.savefig(figures_dir / f"tts_vs_N_gamma_{gamma}.svg")
        plt.close(fig)


def plot_fidelity_vs_N_per_gamma(df: pd.DataFrame, figures_dir: Path, config: HighNoiseConfig) -> None:
    """
    Plot accepted fidelity (mean/median) vs N for each gamma.
    """
    default_thr = config.thresholds[1] if len(config.thresholds) > 1 else config.thresholds[0]
    default_win = config.windows[-1] if len(config.windows) > 0 else 1.0
    
    for gamma in config.gamma_values:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        subset = df[
            (df["gamma"] == gamma) & 
            (df["threshold"] == default_thr) & 
            (df["window"] == default_win)
        ]
        
        for ax, metric, title in [
            (axes[0], "mean_accepted_final_fidelity", "Mean"),
            (axes[1], "median_accepted_final_fidelity", "Median")
        ]:
            for mode, color, marker, label in [
                ("global", "#d62728", "o", "Global"),
                ("hierarchical", "#2ca02c", "s", "Hierarchical")
            ]:
                m_df = subset[subset["mode"] == mode].sort_values("N")
                valid = m_df[~m_df[metric].isna()]
                
                if len(valid) > 0:
                    ax.plot(valid["N"], valid[metric], f"{marker}-", color=color,
                           linewidth=2, markersize=10, label=label)
            
            ax.set_xlabel("System Size N", fontsize=12)
            ax.set_ylabel(f"{title} Accepted Fidelity", fontsize=12)
            ax.set_title(f"{title} Fidelity at γ={gamma}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(config.system_sizes)
            ax.set_ylim(0, 1)
        
        fig.suptitle(f"Accepted Fidelity vs N (γ={gamma}, threshold={default_thr})", fontsize=12)
        fig.tight_layout()
        fig.savefig(figures_dir / f"fidelity_vs_N_gamma_{gamma}.png", dpi=300)
        fig.savefig(figures_dir / f"fidelity_vs_N_gamma_{gamma}.svg")
        plt.close(fig)


def plot_heatmaps(df: pd.DataFrame, figures_dir: Path, config: HighNoiseConfig) -> None:
    """
    Create heatmaps of acceptance over (gamma, N) for each mode.
    Fixed at default threshold/window.
    """
    default_thr = config.thresholds[1] if len(config.thresholds) > 1 else config.thresholds[0]
    default_win = config.windows[-1] if len(config.windows) > 0 else 1.0
    
    subset = df[(df["threshold"] == default_thr) & (df["window"] == default_win)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, mode in zip(axes, ["global", "hierarchical"]):
        m_df = subset[subset["mode"] == mode]
        
        # Pivot to create heatmap matrix
        pivot = m_df.pivot(index="gamma", columns="N", values="acceptance_probability")
        pivot = pivot.sort_index(ascending=False)  # High gamma at top
        
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn',
                       vmin=0, vmax=1)
        
        # Labels
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{g:.1f}" for g in pivot.index])
        
        ax.set_xlabel("System Size N", fontsize=12)
        ax.set_ylabel("Noise Rate (γ)", fontsize=12)
        ax.set_title(f"{mode.capitalize()} Conditioning", fontsize=12)
        
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                text_color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       color=text_color, fontsize=9)
        
        plt.colorbar(im, ax=ax, label="Acceptance Prob.")
    
    fig.suptitle(f"Acceptance Heatmap (threshold={default_thr}, window={default_win})",
                fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(figures_dir / "heatmap_acceptance.png", dpi=300)
    fig.savefig(figures_dir / "heatmap_acceptance.svg")
    plt.close(fig)


def plot_combined_summary(df: pd.DataFrame, figures_dir: Path, config: HighNoiseConfig) -> None:
    """
    Create a comprehensive 2x2 summary figure.
    """
    default_thr = config.thresholds[1] if len(config.thresholds) > 1 else config.thresholds[0]
    default_win = config.windows[-1] if len(config.windows) > 0 else 1.0
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    subset = df[(df["threshold"] == default_thr) & (df["window"] == default_win)]
    
    # (0,0) Acceptance vs gamma for different N - Global
    ax = axes[0, 0]
    global_df = subset[subset["mode"] == "global"]
    for N in config.system_sizes:
        n_df = global_df[global_df["N"] == N].sort_values("gamma")
        accept = n_df["acceptance_probability"].replace(0, 1e-5)
        ax.semilogy(n_df["gamma"], accept, 'o-', color=N_COLORS.get(N, 'gray'),
                   linewidth=2, markersize=8, label=f"N={N}")
    ax.set_xlabel("Noise Rate (γ)")
    ax.set_ylabel("Acceptance Probability (log)")
    ax.set_title("(a) Global: Acceptance vs Noise")
    ax.legend(title="N", loc="lower left")
    ax.grid(True, alpha=0.3)
    
    # (0,1) Acceptance vs gamma for different N - Hierarchical
    ax = axes[0, 1]
    hier_df = subset[subset["mode"] == "hierarchical"]
    for N in config.system_sizes:
        n_df = hier_df[hier_df["N"] == N].sort_values("gamma")
        accept = n_df["acceptance_probability"].replace(0, 1e-5)
        ax.semilogy(n_df["gamma"], accept, 'o-', color=N_COLORS.get(N, 'gray'),
                   linewidth=2, markersize=8, label=f"N={N}")
    ax.set_xlabel("Noise Rate (γ)")
    ax.set_ylabel("Acceptance Probability (log)")
    ax.set_title("(b) Hierarchical: Acceptance vs Noise")
    ax.legend(title="N", loc="lower left")
    ax.grid(True, alpha=0.3)
    
    # (1,0) TTS comparison at highest gamma
    ax = axes[1, 0]
    high_gamma = max(config.gamma_values)
    high_df = subset[subset["gamma"] == high_gamma]
    
    MAX_TTS = 1e6
    for mode, color, marker in [("global", "#d62728", "o"), ("hierarchical", "#2ca02c", "s")]:
        m_df = high_df[high_df["mode"] == mode].sort_values("N")
        tts = m_df["time_to_solution"].clip(upper=MAX_TTS)
        ax.semilogy(m_df["N"], tts, f"{marker}-", color=color, linewidth=2, 
                   markersize=10, label=mode.capitalize())
    
    ax.axhline(y=MAX_TTS, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel("System Size N")
    ax.set_ylabel("Time-to-Solution (log)")
    ax.set_title(f"(c) TTS at γ={high_gamma}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(config.system_sizes)
    
    # (1,1) Advantage ratio vs gamma
    ax = axes[1, 1]
    
    for N in config.system_sizes:
        ratios = []
        valid_gammas = []
        for gamma in sorted(subset["gamma"].unique()):
            g_row = subset[(subset["N"] == N) & (subset["gamma"] == gamma) & (subset["mode"] == "global")]
            h_row = subset[(subset["N"] == N) & (subset["gamma"] == gamma) & (subset["mode"] == "hierarchical")]
            
            if len(g_row) > 0 and len(h_row) > 0:
                g_acc = g_row["acceptance_probability"].iloc[0]
                h_acc = h_row["acceptance_probability"].iloc[0]
                
                if g_acc > 0 and h_acc > 0:
                    ratio = h_acc / g_acc
                    ratios.append(ratio)
                    valid_gammas.append(gamma)
                elif g_acc == 0 and h_acc > 0:
                    ratios.append(1e5)  # Effectively infinite advantage
                    valid_gammas.append(gamma)
        
        if ratios:
            ax.semilogy(valid_gammas, ratios, 'o-', color=N_COLORS.get(N, 'gray'),
                       linewidth=2, markersize=8, label=f"N={N}")
    
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Noise Rate (γ)")
    ax.set_ylabel("Advantage (Hier/Global)")
    ax.set_title("(d) Hierarchical Advantage Ratio")
    ax.legend(title="N")
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"High-Noise Analysis Summary (threshold={default_thr}, window={default_win})",
                fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(figures_dir / "combined_summary.png", dpi=300)
    fig.savefig(figures_dir / "combined_summary.svg")
    plt.close(fig)


# =============================================================================
# Output Functions
# =============================================================================

def safe_json_serialize(obj: Any) -> Any:
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
    """Save dictionary to JSON with proper type handling."""
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return safe_json_serialize(obj)
    
    with open(path, "w") as f:
        json.dump(convert(data), f, indent=2)


def save_csv(path: Path, df: pd.DataFrame) -> None:
    """Save DataFrame to CSV with proper infinity handling."""
    # Replace inf with "Infinity" for CSV
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=[np.floating]).columns:
        df_copy[col] = df_copy[col].apply(
            lambda x: "Infinity" if (isinstance(x, float) and math.isinf(x) and x > 0)
            else "-Infinity" if (isinstance(x, float) and math.isinf(x) and x < 0)
            else x
        )
    df_copy.to_csv(path, index=False)


def print_key_findings(df: pd.DataFrame, config: HighNoiseConfig) -> None:
    """Print the Key Findings table to console."""
    
    # Use threshold=0.7, window=1.0
    target_thr = 0.7
    target_win = 1.0
    
    # Find closest available
    available_thrs = df["threshold"].unique()
    available_wins = df["window"].unique()
    
    if target_thr not in available_thrs:
        target_thr = min(available_thrs, key=lambda x: abs(x - target_thr))
    if target_win not in available_wins:
        target_win = min(available_wins, key=lambda x: abs(x - target_win))
    
    subset = df[(df["threshold"] == target_thr) & (df["window"] == target_win)]
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS TABLE")
    print(f"(threshold={target_thr}, window={target_win})")
    print("=" * 80)
    print()
    
    # Header
    print(f"{'N':>4} | {'Gamma':>6} | {'Global Accept':>13} | {'Global TTS':>12} | {'Hier Accept':>12} | {'Hier TTS':>12} | {'Advantage':>10}")
    print("-" * 80)
    
    for gamma in sorted(subset["gamma"].unique()):
        for N in sorted(subset["N"].unique()):
            g_row = subset[(subset["N"] == N) & (subset["gamma"] == gamma) & (subset["mode"] == "global")]
            h_row = subset[(subset["N"] == N) & (subset["gamma"] == gamma) & (subset["mode"] == "hierarchical")]
            
            if len(g_row) > 0 and len(h_row) > 0:
                g_acc = g_row["acceptance_probability"].iloc[0]
                g_tts = g_row["time_to_solution"].iloc[0]
                h_acc = h_row["acceptance_probability"].iloc[0]
                h_tts = h_row["time_to_solution"].iloc[0]
                
                # Format TTS
                g_tts_str = f"{g_tts:.1f}" if not math.isinf(g_tts) else "∞"
                h_tts_str = f"{h_tts:.1f}" if not math.isinf(h_tts) else "∞"
                
                # Compute advantage ratio
                if g_acc > 0 and h_acc > 0:
                    ratio = h_acc / g_acc
                    ratio_str = f"{ratio:.1f}×"
                elif g_acc == 0 and h_acc > 0:
                    ratio_str = "∞"
                elif g_acc == 0 and h_acc == 0:
                    ratio_str = "N/A"
                else:
                    ratio_str = f"{h_acc/g_acc:.1f}×"
                
                print(f"{N:>4} | {gamma:>6.1f} | {g_acc:>13.4f} | {g_tts_str:>12} | {h_acc:>12.4f} | {h_tts_str:>12} | {ratio_str:>10}")
        
        print("-" * 80)
    
    # Summary
    global_fail = subset[(subset["mode"] == "global") & (subset["acceptance_probability"] == 0)]
    hier_fail = subset[(subset["mode"] == "hierarchical") & (subset["acceptance_probability"] == 0)]
    
    print()
    print("SUMMARY:")
    print(f"  - Global conditioning: {len(global_fail)} configurations with 0 acceptance")
    print(f"  - Hierarchical conditioning: {len(hier_fail)} configurations with 0 acceptance")
    
    # Find where hierarchical maintains finite acceptance but global fails
    advantage_configs = 0
    for _, g_row in subset[subset["mode"] == "global"].iterrows():
        h_row = subset[
            (subset["N"] == g_row["N"]) & 
            (subset["gamma"] == g_row["gamma"]) & 
            (subset["mode"] == "hierarchical")
        ]
        if len(h_row) > 0:
            if g_row["acceptance_probability"] == 0 and h_row["acceptance_probability"].iloc[0] > 0:
                advantage_configs += 1
    
    print(f"  - Hierarchical advantage (finite vs zero): {advantage_configs} configurations")
    print()


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run high-noise scaling sweep with adaptive trials.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON configuration file")
    parser.add_argument("--system-sizes", type=str, default=None,
                        help="Override: comma-separated system sizes")
    parser.add_argument("--gamma-values", type=str, default=None,
                        help="Override: comma-separated gamma values")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Override: comma-separated thresholds")
    parser.add_argument("--windows", type=str, default=None,
                        help="Override: comma-separated windows")
    parser.add_argument("--initial-trials", type=int, default=None,
                        help="Override: initial trial count")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Override: maximum trial count")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override: random seed")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override: output directory")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = HighNoiseConfig.from_json(Path(args.config))
    else:
        # Try default config location
        default_config = Path("configs/high_noise_config.json")
        if default_config.exists():
            config = HighNoiseConfig.from_json(default_config)
        else:
            config = HighNoiseConfig()
    
    # Apply overrides
    if args.system_sizes:
        config.system_sizes = [int(x.strip()) for x in args.system_sizes.split(",")]
    if args.gamma_values:
        config.gamma_values = [float(x.strip()) for x in args.gamma_values.split(",")]
    if args.thresholds:
        config.thresholds = [float(x.strip()) for x in args.thresholds.split(",")]
    if args.windows:
        config.windows = [float(x.strip()) for x in args.windows.split(",")]
    if args.initial_trials:
        config.adaptive_trials.initial_trials = args.initial_trials
    if args.max_trials:
        config.adaptive_trials.max_trials = args.max_trials
    if args.seed:
        config.seed = args.seed
    if args.out_dir:
        config.output_base_dir = args.out_dir
    
    # Create output directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_base_dir) / f"high_noise_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("HIGH-NOISE REGIME SIMULATION")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print()
    
    # Save config
    save_json(output_dir / "sweep_config.json", config.to_dict())
    
    # Run sweep
    start_time = time.time()
    df = run_high_noise_sweep(config)
    elapsed = time.time() - start_time
    
    print(f"\nSweep completed in {elapsed:.1f} seconds")
    print(f"Total trials run: {df['n_trials'].sum()}")
    
    # Save results
    save_csv(output_dir / "results.csv", df)
    print(f"Saved results.csv ({len(df)} configurations)")
    
    # Compute and save summary stats
    stats = compute_summary_stats(df, config)
    save_json(output_dir / "summary_stats.json", stats)
    print("Saved summary_stats.json")
    
    # Generate plots
    print("\nGenerating plots...")
    
    plot_acceptance_vs_N_per_gamma(df, figures_dir, config)
    print("  - acceptance_vs_N_gamma_*.png/.svg")
    
    plot_tts_vs_N_per_gamma(df, figures_dir, config)
    print("  - tts_vs_N_gamma_*.png/.svg")
    
    plot_fidelity_vs_N_per_gamma(df, figures_dir, config)
    print("  - fidelity_vs_N_gamma_*.png/.svg")
    
    plot_heatmaps(df, figures_dir, config)
    print("  - heatmap_acceptance.png/.svg")
    
    plot_combined_summary(df, figures_dir, config)
    print("  - combined_summary.png/.svg")
    
    # Print key findings
    print_key_findings(df, config)
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
