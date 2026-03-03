#!/usr/bin/env python3
"""
Follow-up High-Noise Sweep with Extended Hierarchical Rules

This script extends the high-noise simulations with:
1. New hierarchical aggregation rules (k-of-n voting)
2. Extended system sizes (up to 32 or 64)
3. Tighter thresholds to make hierarchical non-trivial
4. Comparison with baseline overnight sweep results

Key Innovation (for second provisional):
The "k_of_n" hierarchical rule accepts if at least ceil(k_fraction * N) 
subsystems pass their local tests. This provides a tunable trade-off 
between acceptance probability and quality guarantee.

Usage:
    uv run python scripts/run_followup.py --config configs/followup_config.json
    uv run python scripts/run_followup.py --hier-rule k_of_n --k-fraction 0.9
    uv run python scripts/run_followup.py --hier-rule all  # Original behavior
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
from typing import Any, Dict, List, Optional, Tuple, Literal

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
    max_trials: int = 3000
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
class HierarchicalRuleConfig:
    """Configuration for hierarchical conditioning rules."""
    rule: str = "all"  # "all" or "k_of_n"
    k_fraction: float = 1.0  # For k_of_n: accept if ceil(k_fraction * N) pass
    
    def required_passing(self, N: int) -> int:
        """Compute minimum number of subsystems that must pass."""
        if self.rule == "all":
            return N
        else:  # k_of_n
            return int(math.ceil(self.k_fraction * N))


@dataclass
class FollowupConfig:
    """Complete configuration for follow-up sweep."""
    
    system_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    gamma_values: List[float] = field(default_factory=lambda: [1.0, 2.0, 5.0, 10.0])
    thresholds: List[float] = field(default_factory=lambda: [0.8, 0.85, 0.9, 0.95])
    windows: List[float] = field(default_factory=lambda: [0.2, 0.5, 1.0])
    acceptance_mode: str = "window_max"
    
    adaptive_trials: AdaptiveTrialsConfig = field(default_factory=AdaptiveTrialsConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    hierarchical_rule: HierarchicalRuleConfig = field(default_factory=HierarchicalRuleConfig)
    
    seed: int = 42
    output_base_dir: str = "runs"
    
    @classmethod
    def from_json(cls, path: Path) -> "FollowupConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        adaptive = AdaptiveTrialsConfig(**data.get("adaptive_trials", {}))
        sim = SimulationConfig(**data.get("simulation", {}))
        hier_rule = HierarchicalRuleConfig()  # Will be overridden by CLI
        
        return cls(
            system_sizes=data.get("system_sizes", [1, 2, 4, 8, 16, 32]),
            gamma_values=data.get("gamma_values", [1.0, 2.0, 5.0, 10.0]),
            thresholds=data.get("thresholds", [0.8, 0.85, 0.9, 0.95]),
            windows=data.get("windows", [0.2, 0.5, 1.0]),
            acceptance_mode=data.get("acceptance_mode", "window_max"),
            adaptive_trials=adaptive,
            simulation=sim,
            hierarchical_rule=hier_rule,
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
            "hierarchical_rule": asdict(self.hierarchical_rule),
            "seed": self.seed,
            "output_base_dir": self.output_base_dir,
        }


# =============================================================================
# Deterministic Seeding
# =============================================================================

def compute_config_seed(base_seed: int, N: int, gamma: float, threshold: float, 
                        window: float, mode: str, hier_rule: str = "all") -> int:
    """
    Compute a deterministic seed from configuration parameters.
    Include hier_rule in hash for reproducibility.
    """
    config_str = f"{base_seed}:{N}:{gamma:.6f}:{threshold:.6f}:{window:.6f}:{mode}:{hier_rule}"
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
    """
    Simulate N independent qubit trajectories under dephasing noise.
    """
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
    hier_rule_config: HierarchicalRuleConfig,
) -> Tuple[bool, float, float, int]:
    """
    Apply conditioning strategy and return acceptance + fidelity metrics.
    
    New: Returns number of passing subsystems for diagnostics.
    
    Returns:
        (accepted, final_fidelity, peak_fidelity, n_passing_subsystems)
    """
    N = len(subsystem_fidelities)
    
    if mode == "global":
        # GLOBAL CONDITIONING: Test product fidelity
        accepted, _, _, _ = compute_acceptance(
            tlist, global_fidelity, threshold, accept_mode, accept_window
        )
        n_passing = N if accepted else 0
    else:
        # HIERARCHICAL CONDITIONING with configurable rule
        # 
        # RULE OPTIONS:
        # 1. "all": Original behavior - accept if ALL subsystems pass
        #    - Strictest guarantee: every subsystem meets threshold
        #    - Acceptance probability: p^N where p is single-subsystem prob
        #
        # 2. "k_of_n": Accept if at least ceil(k_fraction * N) pass
        #    - Relaxed guarantee: allows some subsystems to fail
        #    - Higher acceptance probability at cost of weaker guarantee
        #    - PATENT RELEVANCE: Demonstrates tunable trade-off
        #    - At k_fraction=1.0, equivalent to "all"
        #    - At k_fraction=0.5, allows up to half to fail
        
        n_passing = 0
        for fid in subsystem_fidelities:
            sub_accepted, _, _, _ = compute_acceptance(
                tlist, fid, threshold, accept_mode, accept_window
            )
            if sub_accepted:
                n_passing += 1
        
        required = hier_rule_config.required_passing(N)
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
    config: FollowupConfig,
) -> Dict[str, Any]:
    """
    Run trials for a single configuration with adaptive trial count.
    """
    hier_rule = config.hierarchical_rule.rule if mode == "hierarchical" else "N/A"
    k_fraction = config.hierarchical_rule.k_fraction if mode == "hierarchical" else float('nan')
    
    base_seed = compute_config_seed(
        config.seed, N, gamma, threshold, window, mode, 
        config.hierarchical_rule.rule
    )
    
    trials_run = 0
    accepted_count = 0
    final_fidelities = []
    peak_fidelities = []
    passing_counts = []
    
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
        
        accepted, final_fid, peak_fid, n_passing = apply_conditioning(
            tlist, global_fidelity, subsystem_fidelities,
            mode, threshold, config.acceptance_mode, window,
            config.hierarchical_rule,
        )
        
        trials_run += 1
        passing_counts.append(n_passing)
        
        if accepted:
            accepted_count += 1
            final_fidelities.append(final_fid)
            peak_fidelities.append(peak_fid)
        
        # Adaptive trial increase
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
    tts = 1.0 / acceptance_prob if acceptance_prob > 0 else float('inf')
    
    # Undersampled flag
    if n_accepted == 0 and trials_run < adaptive.max_trials:
        undersampled = True
    
    # Average passing subsystems (diagnostic)
    mean_passing = float(np.mean(passing_counts))
    
    return {
        "N": N,
        "gamma": gamma,
        "threshold": threshold,
        "window": window,
        "mode": mode,
        "hier_rule": hier_rule,
        "k_fraction": k_fraction,
        "n_trials": trials_run,
        "n_accepted": n_accepted,
        "acceptance_probability": acceptance_prob,
        "mean_accepted_final_fidelity": mean_fid,
        "median_accepted_final_fidelity": median_fid,
        "p10_accepted_final_fidelity": p10_fid,
        "time_to_solution": tts,
        "mean_passing_subsystems": mean_passing,
        "undersampled": undersampled,
    }


# =============================================================================
# Main Sweep
# =============================================================================

def run_followup_sweep(config: FollowupConfig) -> pd.DataFrame:
    """Execute the follow-up sweep with extended hierarchical rules."""
    
    # Build parameter grid
    grid = list(product(
        config.system_sizes,
        config.gamma_values,
        config.thresholds,
        config.windows,
        ["global", "hierarchical"],
    ))
    
    print(f"Follow-up Sweep Configuration")
    print("=" * 60)
    print(f"System sizes: {config.system_sizes}")
    print(f"Gamma values: {config.gamma_values}")
    print(f"Thresholds: {config.thresholds}")
    print(f"Windows: {config.windows}")
    print(f"Hierarchical rule: {config.hierarchical_rule.rule}")
    if config.hierarchical_rule.rule == "k_of_n":
        print(f"  k_fraction: {config.hierarchical_rule.k_fraction}")
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

def compute_summary_stats(df: pd.DataFrame, config: FollowupConfig) -> Dict[str, Any]:
    """Compute comprehensive summary statistics."""
    
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
    
    # Undersampled configurations
    undersampled = df[df["undersampled"] == True][
        ["N", "gamma", "threshold", "window", "mode", "n_trials", "n_accepted"]
    ].to_dict(orient="records")
    
    summary["undersampled_configurations"] = undersampled
    summary["n_undersampled"] = len(undersampled)
    
    # Mode comparison
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
            "rule": config.hierarchical_rule.rule,
            "k_fraction": config.hierarchical_rule.k_fraction,
        }
    }
    
    return summary


# =============================================================================
# Plotting Functions
# =============================================================================

N_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 4: '#2ca02c', 8: '#d62728', 
            16: '#9467bd', 32: '#8c564b', 64: '#e377c2'}


def plot_acceptance_vs_N_per_gamma(df: pd.DataFrame, figures_dir: Path, config: FollowupConfig) -> None:
    """Create acceptance vs N plots for each gamma."""
    default_thr = 0.9 if 0.9 in config.thresholds else config.thresholds[len(config.thresholds)//2]
    default_win = 0.5 if 0.5 in config.windows else config.windows[0]
    
    for gamma in config.gamma_values:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        subset = df[
            (df["gamma"] == gamma) & 
            (df["threshold"] == default_thr) & 
            (df["window"] == default_win)
        ]
        
        for mode, color, marker, label in [
            ("global", "#d62728", "o", "Global"),
            ("hierarchical", "#2ca02c", "s", f"Hierarchical ({config.hierarchical_rule.rule})")
        ]:
            m_df = subset[subset["mode"] == mode].sort_values("N")
            accept = m_df["acceptance_probability"].replace(0, 1e-5)
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
        ax.set_ylim(1e-6, 2)
        
        fig.tight_layout()
        fig.savefig(figures_dir / f"acceptance_vs_N_gamma_{gamma}.png", dpi=300)
        fig.savefig(figures_dir / f"acceptance_vs_N_gamma_{gamma}.svg")
        plt.close(fig)


def plot_tts_vs_N_per_gamma(df: pd.DataFrame, figures_dir: Path, config: FollowupConfig) -> None:
    """Create TTS vs N plots with log y-axis."""
    default_thr = 0.9 if 0.9 in config.thresholds else config.thresholds[len(config.thresholds)//2]
    default_win = 0.5 if 0.5 in config.windows else config.windows[0]
    
    MAX_TTS_DISPLAY = 1e6
    
    for gamma in config.gamma_values:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        subset = df[
            (df["gamma"] == gamma) & 
            (df["threshold"] == default_thr) & 
            (df["window"] == default_win)
        ]
        
        for mode, color, marker, label in [
            ("global", "#d62728", "o", "Global"),
            ("hierarchical", "#2ca02c", "s", f"Hierarchical ({config.hierarchical_rule.rule})")
        ]:
            m_df = subset[subset["mode"] == mode].sort_values("N")
            tts = m_df["time_to_solution"].copy()
            
            inf_mask = np.isinf(tts)
            tts_clipped = tts.clip(upper=MAX_TTS_DISPLAY)
            
            ax.semilogy(m_df["N"], tts_clipped, f"{marker}-", color=color,
                       linewidth=2, markersize=10, label=label)
            
            if inf_mask.any():
                inf_N = m_df["N"].values[inf_mask]
                ax.scatter(inf_N, [MAX_TTS_DISPLAY] * len(inf_N), 
                          marker='x', s=150, color=color, zorder=5)
        
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


def plot_heatmaps(df: pd.DataFrame, figures_dir: Path, config: FollowupConfig) -> None:
    """Create heatmaps of acceptance over (gamma, N)."""
    default_thr = 0.9 if 0.9 in config.thresholds else config.thresholds[len(config.thresholds)//2]
    default_win = 0.5 if 0.5 in config.windows else config.windows[0]
    
    subset = df[(df["threshold"] == default_thr) & (df["window"] == default_win)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, mode in zip(axes, ["global", "hierarchical"]):
        m_df = subset[subset["mode"] == mode]
        
        pivot = m_df.pivot(index="gamma", columns="N", values="acceptance_probability")
        pivot = pivot.sort_index(ascending=False)
        
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{g:.1f}" for g in pivot.index])
        
        ax.set_xlabel("System Size N", fontsize=12)
        ax.set_ylabel("Noise Rate (γ)", fontsize=12)
        
        title = mode.capitalize()
        if mode == "hierarchical":
            title += f" ({config.hierarchical_rule.rule})"
        ax.set_title(title, fontsize=12)
        
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


def plot_combined_summary(df: pd.DataFrame, figures_dir: Path, config: FollowupConfig) -> None:
    """Create comprehensive 2x2 summary figure."""
    default_thr = 0.9 if 0.9 in config.thresholds else config.thresholds[len(config.thresholds)//2]
    default_win = 0.5 if 0.5 in config.windows else config.windows[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    subset = df[(df["threshold"] == default_thr) & (df["window"] == default_win)]
    
    # (0,0) Acceptance vs gamma - Global
    ax = axes[0, 0]
    global_df = subset[subset["mode"] == "global"]
    for N in config.system_sizes:
        n_df = global_df[global_df["N"] == N].sort_values("gamma")
        accept = n_df["acceptance_probability"].replace(0, 1e-6)
        ax.semilogy(n_df["gamma"], accept, 'o-', color=N_COLORS.get(N, 'gray'),
                   linewidth=2, markersize=8, label=f"N={N}")
    ax.set_xlabel("Noise Rate (γ)")
    ax.set_ylabel("Acceptance Probability (log)")
    ax.set_title("(a) Global: Acceptance vs Noise")
    ax.legend(title="N", loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # (0,1) Acceptance vs gamma - Hierarchical
    ax = axes[0, 1]
    hier_df = subset[subset["mode"] == "hierarchical"]
    for N in config.system_sizes:
        n_df = hier_df[hier_df["N"] == N].sort_values("gamma")
        accept = n_df["acceptance_probability"].replace(0, 1e-6)
        ax.semilogy(n_df["gamma"], accept, 'o-', color=N_COLORS.get(N, 'gray'),
                   linewidth=2, markersize=8, label=f"N={N}")
    ax.set_xlabel("Noise Rate (γ)")
    ax.set_ylabel("Acceptance Probability (log)")
    ax.set_title(f"(b) Hierarchical ({config.hierarchical_rule.rule}): Acceptance vs Noise")
    ax.legend(title="N", loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # (1,0) TTS at highest gamma
    ax = axes[1, 0]
    high_gamma = max(config.gamma_values)
    high_df = subset[subset["gamma"] == high_gamma]
    
    MAX_TTS = 1e6
    for mode, color, marker in [("global", "#d62728", "o"), ("hierarchical", "#2ca02c", "s")]:
        m_df = high_df[high_df["mode"] == mode].sort_values("N")
        tts = m_df["time_to_solution"].clip(upper=MAX_TTS)
        label = mode.capitalize()
        if mode == "hierarchical":
            label += f" ({config.hierarchical_rule.rule})"
        ax.semilogy(m_df["N"], tts, f"{marker}-", color=color, linewidth=2, 
                   markersize=10, label=label)
    
    ax.axhline(y=MAX_TTS, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel("System Size N")
    ax.set_ylabel("Time-to-Solution (log)")
    ax.set_title(f"(c) TTS at γ={high_gamma}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(config.system_sizes)
    
    # (1,1) Advantage ratio
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
                    ratios.append(1e5)
                    valid_gammas.append(gamma)
        
        if ratios:
            ax.semilogy(valid_gammas, ratios, 'o-', color=N_COLORS.get(N, 'gray'),
                       linewidth=2, markersize=8, label=f"N={N}")
    
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Noise Rate (γ)")
    ax.set_ylabel("Advantage (Hier/Global)")
    ax.set_title("(d) Hierarchical Advantage Ratio")
    ax.legend(title="N", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Follow-up Analysis (threshold={default_thr}, window={default_win})",
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
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=[np.floating]).columns:
        df_copy[col] = df_copy[col].apply(
            lambda x: "Infinity" if (isinstance(x, float) and math.isinf(x) and x > 0)
            else "-Infinity" if (isinstance(x, float) and math.isinf(x) and x < 0)
            else x
        )
    df_copy.to_csv(path, index=False)


def print_key_findings(df: pd.DataFrame, config: FollowupConfig) -> None:
    """Print key findings table to console."""
    
    target_thr = 0.9 if 0.9 in config.thresholds else config.thresholds[0]
    target_win = 0.5 if 0.5 in config.windows else config.windows[0]
    
    subset = df[(df["threshold"] == target_thr) & (df["window"] == target_win)]
    
    print("\n" + "=" * 90)
    print("KEY FINDINGS TABLE")
    print(f"(threshold={target_thr}, window={target_win}, hier_rule={config.hierarchical_rule.rule})")
    print("=" * 90)
    print()
    
    print(f"{'N':>4} | {'Gamma':>6} | {'Global Accept':>13} | {'Global TTS':>12} | {'Hier Accept':>12} | {'Hier TTS':>12} | {'Advantage':>10}")
    print("-" * 90)
    
    for gamma in sorted(subset["gamma"].unique()):
        for N in sorted(subset["N"].unique()):
            g_row = subset[(subset["N"] == N) & (subset["gamma"] == gamma) & (subset["mode"] == "global")]
            h_row = subset[(subset["N"] == N) & (subset["gamma"] == gamma) & (subset["mode"] == "hierarchical")]
            
            if len(g_row) > 0 and len(h_row) > 0:
                g_acc = g_row["acceptance_probability"].iloc[0]
                g_tts = g_row["time_to_solution"].iloc[0]
                h_acc = h_row["acceptance_probability"].iloc[0]
                h_tts = h_row["time_to_solution"].iloc[0]
                
                g_tts_str = f"{g_tts:.1f}" if not math.isinf(g_tts) else "∞"
                h_tts_str = f"{h_tts:.1f}" if not math.isinf(h_tts) else "∞"
                
                if g_acc > 0 and h_acc > 0:
                    ratio_str = f"{h_acc / g_acc:.1f}×"
                elif g_acc == 0 and h_acc > 0:
                    ratio_str = "∞"
                elif g_acc == 0 and h_acc == 0:
                    ratio_str = "N/A"
                else:
                    ratio_str = f"{h_acc/g_acc:.1f}×" if g_acc > 0 else "N/A"
                
                print(f"{N:>4} | {gamma:>6.1f} | {g_acc:>13.4f} | {g_tts_str:>12} | {h_acc:>12.4f} | {h_tts_str:>12} | {ratio_str:>10}")
        
        print("-" * 90)
    
    # Summary
    global_fail = subset[(subset["mode"] == "global") & (subset["acceptance_probability"] == 0)]
    hier_fail = subset[(subset["mode"] == "hierarchical") & (subset["acceptance_probability"] == 0)]
    
    print()
    print("SUMMARY:")
    print(f"  - Global conditioning: {len(global_fail)} configs with 0 acceptance")
    print(f"  - Hierarchical ({config.hierarchical_rule.rule}): {len(hier_fail)} configs with 0 acceptance")
    print()


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run follow-up sweep with extended hierarchical rules.",
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
    
    # NEW: Hierarchical rule arguments
    parser.add_argument("--hier-rule", type=str, choices=["all", "k_of_n"], default="all",
                        help="Hierarchical aggregation rule: 'all' (original) or 'k_of_n' (relaxed)")
    parser.add_argument("--k-fraction", type=float, default=0.9,
                        help="For k_of_n rule: fraction of subsystems that must pass (0.0-1.0)")
    
    # Extended system sizes flag
    parser.add_argument("--include-64", action="store_true",
                        help="Include N=64 in system sizes (longer runtime)")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = FollowupConfig.from_json(Path(args.config))
    else:
        default_config = Path("configs/followup_config.json")
        if default_config.exists():
            config = FollowupConfig.from_json(default_config)
        else:
            config = FollowupConfig()
    
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
    
    # Apply hierarchical rule settings
    config.hierarchical_rule = HierarchicalRuleConfig(
        rule=args.hier_rule,
        k_fraction=args.k_fraction if args.hier_rule == "k_of_n" else 1.0
    )
    
    # Include N=64 if requested
    if args.include_64 and 64 not in config.system_sizes:
        config.system_sizes = config.system_sizes + [64]
    
    # Create output directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_base_dir) / f"followup_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("FOLLOW-UP SWEEP SIMULATION")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print()
    
    # Save config
    save_json(output_dir / "followup_config.json", config.to_dict())
    
    # Run sweep
    start_time = time.time()
    df = run_followup_sweep(config)
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
    
    plot_heatmaps(df, figures_dir, config)
    print("  - heatmap_acceptance.png/.svg")
    
    plot_combined_summary(df, figures_dir, config)
    print("  - combined_summary.png/.svg")
    
    # Print key findings
    print_key_findings(df, config)
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
