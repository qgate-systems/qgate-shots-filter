#!/usr/bin/env python3
"""
Multi-Frequency Monitoring Sweep

Evaluates TWO fusion strategies for combining low-frequency (LF) and high-frequency (HF) monitors:

1. Baseline (LF only):
   - Accept iff peak_lf >= threshold_lf
   - peak_lf = max_{t in last window_lf} F(t)

2. Variant A (Logical Fusion):
   - Early reject if min_hf < threshold_hf at any step
   - Final accept if peak_lf >= threshold_lf AND HF never triggered
   - Saves compute by rejecting early

3. Variant B (Score Fusion):
   - combined_score = alpha * peak_lf + (1-alpha) * peak_hf
   - Accept iff combined_score >= threshold_combined
   - Optional hard_floor guard for early rejection

Outputs:
- results_multifreq.csv
- summary_multifreq.json
- figures/*.png and *.svg

Usage:
    uv run python scripts/run_multifreq.py
    uv run python scripts/run_multifreq.py --trials 100  # Quick test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sim import run_simulation


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MultiFreqConfig:
    """Configuration for multi-frequency monitoring sweep."""
    # System parameters
    system_sizes: List[int] = None
    gamma_values: List[float] = None
    
    # LF monitor (low-frequency, coarse)
    window_lf: float = 1.0
    threshold_lf: float = 0.7
    
    # HF monitor (high-frequency, fine)
    window_hf: float = 0.2
    threshold_hf: float = 0.55  # Early reject threshold
    
    # Score fusion parameters
    alpha_values: List[float] = None
    threshold_combined_values: List[float] = None
    hard_floor: Optional[float] = None  # Optional early reject guard for score fusion
    
    # Hierarchical k-of-n
    k_fraction: float = 0.9
    
    # Trial settings
    initial_trials: int = 300
    max_trials: int = 2000
    low_acceptance_threshold: float = 0.02
    
    # Simulation physics
    drive_amp: float = 1.0
    drive_freq: float = 1.0
    t_max: float = 12.0
    n_steps: int = 500
    
    seed: int = 42
    
    def __post_init__(self):
        if self.system_sizes is None:
            self.system_sizes = [16, 32, 64]
        if self.gamma_values is None:
            self.gamma_values = [2.0, 5.0, 10.0]
        if self.alpha_values is None:
            self.alpha_values = [0.3, 0.7]
        if self.threshold_combined_values is None:
            self.threshold_combined_values = [0.65, 0.7]


# =============================================================================
# Deterministic Seeding
# =============================================================================

def compute_config_seed(base_seed: int, N: int, gamma: float, variant: str,
                        alpha: float = 0.0, threshold_combined: float = 0.0) -> int:
    """Compute deterministic seed from configuration."""
    config_str = f"{base_seed}:{N}:{gamma:.6f}:{variant}:{alpha:.4f}:{threshold_combined:.4f}"
    hash_bytes = hashlib.sha256(config_str.encode()).digest()
    return int.from_bytes(hash_bytes[:4], byteorder='big')


# =============================================================================
# Configuration Key for Incremental Execution
# =============================================================================

@dataclass(frozen=True)
class ConfigKey:
    """Unique key for a configuration."""
    N: int
    gamma: float
    variant: str
    alpha: float
    threshold_combined: float
    
    def to_tuple(self) -> tuple:
        return (self.N, round(self.gamma, 4), self.variant, 
                round(self.alpha, 4), round(self.threshold_combined, 4))


def extract_existing_keys(df: pd.DataFrame) -> Set[ConfigKey]:
    """Extract configuration keys from existing results."""
    keys = set()
    for _, row in df.iterrows():
        alpha = row.get('alpha', 0.0)
        if pd.isna(alpha):
            alpha = 0.0
        tc = row.get('threshold_combined', 0.0)
        if pd.isna(tc):
            tc = 0.0
        keys.add(ConfigKey(
            N=int(row['N']),
            gamma=float(row['gamma']),
            variant=str(row['variant']),
            alpha=float(alpha),
            threshold_combined=float(tc),
        ))
    return keys


# =============================================================================
# Multi-Frequency Monitoring Logic
# =============================================================================

def compute_lf_metrics(tlist: np.ndarray, fidelity: np.ndarray, 
                       window_lf: float) -> Tuple[float, int]:
    """
    Compute LF monitor metrics.
    Returns (peak_lf, peak_step_index)
    """
    t_end = tlist[-1]
    t_start = max(0, t_end - window_lf)
    
    mask = tlist >= t_start
    if not mask.any():
        return fidelity[-1], len(fidelity) - 1
    
    window_fidelity = fidelity[mask]
    peak_idx_in_window = np.argmax(window_fidelity)
    peak_lf = float(window_fidelity[peak_idx_in_window])
    
    # Get absolute step index
    window_start_idx = np.argmax(mask)
    peak_step = window_start_idx + peak_idx_in_window
    
    return peak_lf, peak_step


def compute_hf_metrics(tlist: np.ndarray, fidelity: np.ndarray,
                       window_hf: float) -> Tuple[float, float, int]:
    """
    Compute HF monitor metrics for the final window.
    Returns (min_hf, peak_hf, min_step_index)
    """
    t_end = tlist[-1]
    t_start = max(0, t_end - window_hf)
    
    mask = tlist >= t_start
    if not mask.any():
        return fidelity[-1], fidelity[-1], len(fidelity) - 1
    
    window_fidelity = fidelity[mask]
    min_hf = float(np.min(window_fidelity))
    peak_hf = float(np.max(window_fidelity))
    
    min_idx_in_window = np.argmin(window_fidelity)
    window_start_idx = np.argmax(mask)
    min_step = window_start_idx + min_idx_in_window
    
    return min_hf, peak_hf, min_step


def check_hf_early_reject(tlist: np.ndarray, fidelity: np.ndarray,
                          window_hf: float, threshold_hf: float) -> Tuple[bool, int]:
    """
    Check if HF monitor would trigger early rejection at any point.
    Scans through trajectory checking rolling minimum in HF window.
    
    Returns (triggered, step_of_first_trigger)
    """
    n_steps = len(tlist)
    dt = tlist[1] - tlist[0] if n_steps > 1 else 0.1
    window_steps = max(1, int(window_hf / dt))
    
    for i in range(window_steps, n_steps):
        window_min = np.min(fidelity[max(0, i - window_steps):i + 1])
        if window_min < threshold_hf:
            return True, i
    
    return False, n_steps


def apply_baseline_lf(tlist: np.ndarray, fidelity: np.ndarray,
                      window_lf: float, threshold_lf: float) -> Dict[str, Any]:
    """
    Baseline LF-only acceptance.
    """
    peak_lf, peak_step = compute_lf_metrics(tlist, fidelity, window_lf)
    accepted = peak_lf >= threshold_lf
    
    return {
        'accepted': accepted,
        'peak_lf': peak_lf,
        'steps_until_decision': len(tlist),  # Always uses full trajectory
        'hf_triggered': False,
    }


def apply_logical_fusion(tlist: np.ndarray, fidelity: np.ndarray,
                         window_lf: float, threshold_lf: float,
                         window_hf: float, threshold_hf: float) -> Dict[str, Any]:
    """
    Variant A: Logical Fusion.
    - Early reject if HF monitor triggers (min < threshold_hf)
    - Final accept if peak_lf >= threshold_lf AND HF never triggered
    """
    # Check for HF early rejection
    hf_triggered, trigger_step = check_hf_early_reject(
        tlist, fidelity, window_hf, threshold_hf
    )
    
    if hf_triggered:
        return {
            'accepted': False,
            'peak_lf': float('nan'),  # Didn't complete LF check
            'steps_until_decision': trigger_step,
            'hf_triggered': True,
        }
    
    # HF passed, now check LF
    peak_lf, peak_step = compute_lf_metrics(tlist, fidelity, window_lf)
    accepted = peak_lf >= threshold_lf
    
    return {
        'accepted': accepted,
        'peak_lf': peak_lf,
        'steps_until_decision': len(tlist),
        'hf_triggered': False,
    }


def apply_score_fusion(tlist: np.ndarray, fidelity: np.ndarray,
                       window_lf: float, threshold_lf: float,
                       window_hf: float, alpha: float,
                       threshold_combined: float,
                       hard_floor: Optional[float] = None) -> Dict[str, Any]:
    """
    Variant B: Score Fusion.
    - combined_score = alpha * peak_lf + (1-alpha) * peak_hf
    - Accept if combined_score >= threshold_combined
    - Optional hard_floor early rejection
    """
    # Optional hard floor check
    if hard_floor is not None:
        min_hf, _, min_step = compute_hf_metrics(tlist, fidelity, window_hf)
        if min_hf < hard_floor:
            return {
                'accepted': False,
                'peak_lf': float('nan'),
                'peak_hf': float('nan'),
                'combined_score': float('nan'),
                'steps_until_decision': min_step,
                'hf_triggered': True,
            }
    
    # Compute both monitors
    peak_lf, _ = compute_lf_metrics(tlist, fidelity, window_lf)
    _, peak_hf, _ = compute_hf_metrics(tlist, fidelity, window_hf)
    
    combined_score = alpha * peak_lf + (1 - alpha) * peak_hf
    accepted = combined_score >= threshold_combined
    
    return {
        'accepted': accepted,
        'peak_lf': peak_lf,
        'peak_hf': peak_hf,
        'combined_score': combined_score,
        'steps_until_decision': len(tlist),
        'hf_triggered': False,
    }


# =============================================================================
# Simulation and Trial Execution
# =============================================================================

def simulate_subsystem(gamma: float, config: MultiFreqConfig, 
                       seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a single subsystem and return (tlist, fidelity)."""
    df, _ = run_simulation(
        drive_amp=config.drive_amp,
        drive_freq=config.drive_freq,
        gamma_phi=gamma,
        threshold=0.0,
        t_max=config.t_max,
        n_steps=config.n_steps,
        accept_mode="final",
        accept_window=0.0,
        seed=seed
    )
    return df["t"].values, df["fidelity"].values


def run_trial_baseline(N: int, gamma: float, config: MultiFreqConfig,
                       trial_seed: int) -> Dict[str, Any]:
    """Run one trial with baseline LF-only for hierarchical k-of-n."""
    n_passing = 0
    total_steps = 0
    
    for i in range(N):
        subsystem_seed = trial_seed + i * 10000
        tlist, fidelity = simulate_subsystem(gamma, config, subsystem_seed)
        
        result = apply_baseline_lf(tlist, fidelity, config.window_lf, config.threshold_lf)
        if result['accepted']:
            n_passing += 1
        total_steps += result['steps_until_decision']
    
    required = int(math.ceil(config.k_fraction * N))
    accepted = n_passing >= required
    
    return {
        'accepted': accepted,
        'n_passing': n_passing,
        'total_steps': total_steps,
        'hf_triggers': 0,
    }


def run_trial_logical_fusion(N: int, gamma: float, config: MultiFreqConfig,
                             trial_seed: int) -> Dict[str, Any]:
    """Run one trial with logical fusion for hierarchical k-of-n."""
    n_passing = 0
    total_steps = 0
    hf_triggers = 0
    
    for i in range(N):
        subsystem_seed = trial_seed + i * 10000
        tlist, fidelity = simulate_subsystem(gamma, config, subsystem_seed)
        
        result = apply_logical_fusion(
            tlist, fidelity,
            config.window_lf, config.threshold_lf,
            config.window_hf, config.threshold_hf
        )
        
        if result['accepted']:
            n_passing += 1
        if result['hf_triggered']:
            hf_triggers += 1
        total_steps += result['steps_until_decision']
    
    required = int(math.ceil(config.k_fraction * N))
    accepted = n_passing >= required
    
    return {
        'accepted': accepted,
        'n_passing': n_passing,
        'total_steps': total_steps,
        'hf_triggers': hf_triggers,
    }


def run_trial_score_fusion(N: int, gamma: float, config: MultiFreqConfig,
                           alpha: float, threshold_combined: float,
                           trial_seed: int) -> Dict[str, Any]:
    """Run one trial with score fusion for hierarchical k-of-n."""
    n_passing = 0
    total_steps = 0
    hf_triggers = 0
    
    for i in range(N):
        subsystem_seed = trial_seed + i * 10000
        tlist, fidelity = simulate_subsystem(gamma, config, subsystem_seed)
        
        result = apply_score_fusion(
            tlist, fidelity,
            config.window_lf, config.threshold_lf,
            config.window_hf, alpha, threshold_combined,
            config.hard_floor
        )
        
        if result['accepted']:
            n_passing += 1
        if result.get('hf_triggered', False):
            hf_triggers += 1
        total_steps += result['steps_until_decision']
    
    required = int(math.ceil(config.k_fraction * N))
    accepted = n_passing >= required
    
    return {
        'accepted': accepted,
        'n_passing': n_passing,
        'total_steps': total_steps,
        'hf_triggers': hf_triggers,
    }


def compute_false_reject_rate(N: int, gamma: float, config: MultiFreqConfig,
                              variant: str, alpha: float, threshold_combined: float,
                              n_samples: int = 100) -> float:
    """
    Estimate false reject rate: fraction of HF-triggered rejects that baseline would accept.
    """
    base_seed = compute_config_seed(config.seed, N, gamma, "false_reject_check", alpha, threshold_combined)
    
    false_rejects = 0
    hf_rejects = 0
    
    for trial in range(n_samples):
        trial_seed = base_seed + trial
        
        # Simulate subsystems
        subsystem_results_baseline = []
        subsystem_results_variant = []
        
        for i in range(N):
            subsystem_seed = trial_seed + i * 10000
            tlist, fidelity = simulate_subsystem(gamma, config, subsystem_seed)
            
            # Baseline result
            baseline_result = apply_baseline_lf(tlist, fidelity, config.window_lf, config.threshold_lf)
            subsystem_results_baseline.append(baseline_result['accepted'])
            
            # Variant result
            if variant == "logical_fusion":
                var_result = apply_logical_fusion(
                    tlist, fidelity, config.window_lf, config.threshold_lf,
                    config.window_hf, config.threshold_hf
                )
            else:  # score_fusion
                var_result = apply_score_fusion(
                    tlist, fidelity, config.window_lf, config.threshold_lf,
                    config.window_hf, alpha, threshold_combined, config.hard_floor
                )
            subsystem_results_variant.append((var_result['accepted'], var_result.get('hf_triggered', False)))
        
        # Check hierarchical acceptance
        required = int(math.ceil(config.k_fraction * N))
        baseline_passing = sum(subsystem_results_baseline)
        variant_passing = sum(1 for acc, _ in subsystem_results_variant if acc)
        any_hf_triggered = any(hf for _, hf in subsystem_results_variant)
        
        baseline_accepted = baseline_passing >= required
        variant_accepted = variant_passing >= required
        
        if any_hf_triggered and not variant_accepted and baseline_accepted:
            false_rejects += 1
            hf_rejects += 1
        elif any_hf_triggered and not variant_accepted:
            hf_rejects += 1
    
    return false_rejects / max(1, hf_rejects) if hf_rejects > 0 else 0.0


# =============================================================================
# Main Sweep Function
# =============================================================================

def run_config_trials(N: int, gamma: float, variant: str,
                      alpha: float, threshold_combined: float,
                      config: MultiFreqConfig) -> Dict[str, Any]:
    """Run adaptive trials for a single configuration."""
    
    base_seed = compute_config_seed(config.seed, N, gamma, variant, alpha, threshold_combined)
    
    trials_run = 0
    accepted_count = 0
    total_steps_all = 0
    total_hf_triggers = 0
    
    target_trials = config.initial_trials
    
    while trials_run < target_trials:
        trial_seed = base_seed + trials_run
        
        if variant == "baseline":
            result = run_trial_baseline(N, gamma, config, trial_seed)
        elif variant == "logical_fusion":
            result = run_trial_logical_fusion(N, gamma, config, trial_seed)
        else:  # score_fusion
            result = run_trial_score_fusion(N, gamma, config, alpha, threshold_combined, trial_seed)
        
        trials_run += 1
        if result['accepted']:
            accepted_count += 1
        total_steps_all += result['total_steps']
        total_hf_triggers += result['hf_triggers']
        
        # Adaptive trial increase
        if trials_run == target_trials:
            current_rate = accepted_count / trials_run
            if current_rate < config.low_acceptance_threshold:
                if target_trials < config.max_trials:
                    target_trials = min(target_trials * 5, config.max_trials)
    
    # Compute metrics
    acceptance_prob = accepted_count / trials_run if trials_run > 0 else 0.0
    tts = 1.0 / acceptance_prob if acceptance_prob > 0 else float('inf')
    mean_steps = total_steps_all / trials_run if trials_run > 0 else 0
    hf_trigger_rate = total_hf_triggers / (trials_run * N) if trials_run > 0 else 0.0
    
    # Cost per accepted = total steps / accepted count
    cost_per_accepted = total_steps_all / accepted_count if accepted_count > 0 else float('inf')
    
    # Estimate false reject rate (sample-based)
    if variant in ["logical_fusion", "score_fusion"] and hf_trigger_rate > 0:
        false_reject_rate = compute_false_reject_rate(
            N, gamma, config, variant, alpha, threshold_combined, n_samples=50
        )
    else:
        false_reject_rate = 0.0
    
    return {
        'N': N,
        'gamma': gamma,
        'variant': variant,
        'alpha': alpha if variant == "score_fusion" else float('nan'),
        'threshold_combined': threshold_combined if variant == "score_fusion" else float('nan'),
        'window_hf': config.window_hf,
        'threshold_hf': config.threshold_hf,
        'window_lf': config.window_lf,
        'threshold_lf': config.threshold_lf,
        'n_trials': trials_run,
        'n_accepted': accepted_count,
        'acceptance_probability': acceptance_prob,
        'TTS': tts,
        'mean_steps_until_decision': mean_steps,
        'cost_per_accepted': cost_per_accepted,
        'hf_trigger_rate': hf_trigger_rate,
        'false_reject_rate': false_reject_rate,
    }


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_cost_per_accepted(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot cost_per_accepted vs N by gamma for all variants."""
    
    gammas = sorted(df['gamma'].unique())
    fig, axes = plt.subplots(1, len(gammas), figsize=(5 * len(gammas), 5), sharey=True)
    if len(gammas) == 1:
        axes = [axes]
    
    colors = {'baseline': '#1f77b4', 'logical_fusion': '#2ca02c', 
              'score_fusion_0.3': '#ff7f0e', 'score_fusion_0.7': '#d62728'}
    markers = {'baseline': 'o', 'logical_fusion': 's', 
               'score_fusion_0.3': '^', 'score_fusion_0.7': 'v'}
    
    for ax, gamma in zip(axes, gammas):
        for variant in df['variant'].unique():
            v_df = df[(df['gamma'] == gamma) & (df['variant'] == variant)]
            
            if variant == 'score_fusion':
                for alpha in v_df['alpha'].dropna().unique():
                    a_df = v_df[v_df['alpha'] == alpha].sort_values('N')
                    label = f"score_fusion_{alpha}"
                    color = colors.get(label, '#999999')
                    marker = markers.get(label, 'x')
                    
                    cost = a_df['cost_per_accepted'].replace([float('inf')], float('nan'))
                    ax.semilogy(a_df['N'], cost, f'{marker}-', color=color,
                               linewidth=2, markersize=8, label=f"Score (α={alpha})")
            else:
                v_df = v_df.sort_values('N')
                label = variant
                color = colors.get(label, '#999999')
                marker = markers.get(label, 'x')
                
                cost = v_df['cost_per_accepted'].replace([float('inf')], float('nan'))
                ax.semilogy(v_df['N'], cost, f'{marker}-', color=color,
                           linewidth=2, markersize=8, 
                           label=variant.replace('_', ' ').title())
        
        ax.set_xlabel('System Size N', fontsize=11)
        ax.set_title(f'γ = {gamma}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(df['N'].unique()))
        if ax == axes[0]:
            ax.set_ylabel('Cost per Accepted (steps)', fontsize=11)
            ax.legend(fontsize=9, loc='upper left')
    
    fig.suptitle('Multi-Frequency Monitoring: Cost Efficiency\n'
                 '"Early rejection reduces wasted compute"', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(figures_dir / 'cost_per_accepted_vs_N_by_gamma.png', dpi=300)
    fig.savefig(figures_dir / 'cost_per_accepted_vs_N_by_gamma.svg')
    plt.close(fig)


def plot_acceptance_vs_N(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot acceptance_probability vs N by gamma for all variants."""
    
    gammas = sorted(df['gamma'].unique())
    fig, axes = plt.subplots(1, len(gammas), figsize=(5 * len(gammas), 5), sharey=True)
    if len(gammas) == 1:
        axes = [axes]
    
    colors = {'baseline': '#1f77b4', 'logical_fusion': '#2ca02c', 
              'score_fusion_0.3': '#ff7f0e', 'score_fusion_0.7': '#d62728'}
    markers = {'baseline': 'o', 'logical_fusion': 's', 
               'score_fusion_0.3': '^', 'score_fusion_0.7': 'v'}
    
    for ax, gamma in zip(axes, gammas):
        for variant in df['variant'].unique():
            v_df = df[(df['gamma'] == gamma) & (df['variant'] == variant)]
            
            if variant == 'score_fusion':
                for alpha in v_df['alpha'].dropna().unique():
                    a_df = v_df[v_df['alpha'] == alpha].sort_values('N')
                    label = f"score_fusion_{alpha}"
                    color = colors.get(label, '#999999')
                    marker = markers.get(label, 'x')
                    
                    acc = a_df['acceptance_probability'].replace(0, 1e-6)
                    ax.semilogy(a_df['N'], acc, f'{marker}-', color=color,
                               linewidth=2, markersize=8, label=f"Score (α={alpha})")
            else:
                v_df = v_df.sort_values('N')
                label = variant
                color = colors.get(label, '#999999')
                marker = markers.get(label, 'x')
                
                acc = v_df['acceptance_probability'].replace(0, 1e-6)
                ax.semilogy(v_df['N'], acc, f'{marker}-', color=color,
                           linewidth=2, markersize=8, 
                           label=variant.replace('_', ' ').title())
        
        ax.set_xlabel('System Size N', fontsize=11)
        ax.set_title(f'γ = {gamma}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(df['N'].unique()))
        ax.set_ylim(1e-4, 2)
        if ax == axes[0]:
            ax.set_ylabel('Acceptance Probability', fontsize=11)
            ax.legend(fontsize=9, loc='lower left')
    
    fig.suptitle('Multi-Frequency Monitoring: Acceptance Rates\n'
                 '"Score fusion tunes sensitivity"', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(figures_dir / 'acceptance_vs_N_by_gamma.png', dpi=300)
    fig.savefig(figures_dir / 'acceptance_vs_N_by_gamma.svg')
    plt.close(fig)


def plot_mean_steps_vs_gamma(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot mean_steps_until_decision vs gamma for each variant."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'baseline': '#1f77b4', 'logical_fusion': '#2ca02c', 
              'score_fusion_0.3': '#ff7f0e', 'score_fusion_0.7': '#d62728'}
    markers = {'baseline': 'o', 'logical_fusion': 's', 
               'score_fusion_0.3': '^', 'score_fusion_0.7': 'v'}
    
    # Average across N for each gamma
    for variant in df['variant'].unique():
        v_df = df[df['variant'] == variant]
        
        if variant == 'score_fusion':
            for alpha in v_df['alpha'].dropna().unique():
                a_df = v_df[v_df['alpha'] == alpha]
                grouped = a_df.groupby('gamma')['mean_steps_until_decision'].mean()
                label = f"score_fusion_{alpha}"
                color = colors.get(label, '#999999')
                marker = markers.get(label, 'x')
                
                ax.plot(grouped.index, grouped.values, f'{marker}-', color=color,
                       linewidth=2, markersize=10, label=f"Score (α={alpha})")
        else:
            grouped = v_df.groupby('gamma')['mean_steps_until_decision'].mean()
            label = variant
            color = colors.get(label, '#999999')
            marker = markers.get(label, 'x')
            
            ax.plot(grouped.index, grouped.values, f'{marker}-', color=color,
                   linewidth=2, markersize=10, 
                   label=variant.replace('_', ' ').title())
    
    ax.set_xlabel('Noise Rate (γ)', fontsize=12)
    ax.set_ylabel('Mean Steps Until Decision', fontsize=12)
    ax.set_title('Multi-Frequency Monitoring: Early Rejection Efficiency\n'
                 '"Logical fusion reduces steps via early HF rejection"', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(figures_dir / 'mean_steps_until_decision_vs_gamma.png', dpi=300)
    fig.savefig(figures_dir / 'mean_steps_until_decision_vs_gamma.svg')
    plt.close(fig)


# =============================================================================
# JSON/CSV Helpers
# =============================================================================

def safe_serialize(obj):
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
        return safe_serialize(obj)
    
    with open(path, 'w') as f:
        json.dump(convert(data), f, indent=2)


def save_csv(path: Path, df: pd.DataFrame) -> None:
    """Save DataFrame to CSV with infinity handling."""
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

def main():
    parser = argparse.ArgumentParser(
        description="Run multi-frequency monitoring sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--trials", type=int, default=300,
                        help="Initial trials per configuration")
    parser.add_argument("--max-trials", type=int, default=2000,
                        help="Maximum trials for low-acceptance configs")
    parser.add_argument("--out-dir", type=str, default="runs/multifreq",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Setup configuration
    config = MultiFreqConfig(
        initial_trials=args.trials,
        max_trials=args.max_trials,
    )
    
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    results_path = output_dir / "results_multifreq.csv"
    
    print("=" * 80)
    print("MULTI-FREQUENCY MONITORING SWEEP")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"System sizes: {config.system_sizes}")
    print(f"Gamma values: {config.gamma_values}")
    print(f"LF: window={config.window_lf}, threshold={config.threshold_lf}")
    print(f"HF: window={config.window_hf}, threshold={config.threshold_hf}")
    print(f"Score fusion: alpha={config.alpha_values}, threshold_combined={config.threshold_combined_values}")
    print(f"Trials: {config.initial_trials} (adaptive to {config.max_trials})")
    print()
    
    # Load existing results for incremental execution
    existing_keys: Set[ConfigKey] = set()
    existing_df = None
    if results_path.exists():
        existing_df = pd.read_csv(results_path)
        existing_keys = extract_existing_keys(existing_df)
        print(f"Loaded {len(existing_keys)} existing configurations")
    
    # Build configuration grid
    configs_to_run = []
    configs_skipped = []
    
    for N, gamma in product(config.system_sizes, config.gamma_values):
        # Baseline
        key = ConfigKey(N, gamma, "baseline", 0.0, 0.0)
        if key not in existing_keys:
            configs_to_run.append(('baseline', N, gamma, 0.0, 0.0))
        else:
            configs_skipped.append(key)
        
        # Logical fusion
        key = ConfigKey(N, gamma, "logical_fusion", 0.0, 0.0)
        if key not in existing_keys:
            configs_to_run.append(('logical_fusion', N, gamma, 0.0, 0.0))
        else:
            configs_skipped.append(key)
        
        # Score fusion variants
        for alpha, tc in product(config.alpha_values, config.threshold_combined_values):
            key = ConfigKey(N, gamma, "score_fusion", alpha, tc)
            if key not in existing_keys:
                configs_to_run.append(('score_fusion', N, gamma, alpha, tc))
            else:
                configs_skipped.append(key)
    
    print(f"Configurations to run: {len(configs_to_run)}")
    print(f"Configurations skipped: {len(configs_skipped)}")
    print()
    
    # Run new configurations
    new_results = []
    if configs_to_run:
        print("=" * 60)
        print("RUNNING CONFIGURATIONS")
        print("=" * 60)
        
        for variant, N, gamma, alpha, tc in tqdm(configs_to_run, desc="Sweeping", unit="config"):
            result = run_config_trials(N, gamma, variant, alpha, tc, config)
            new_results.append(result)
        
        new_df = pd.DataFrame(new_results)
        
        # Merge with existing
        if existing_df is not None:
            merged_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            merged_df = new_df
    else:
        merged_df = existing_df if existing_df is not None else pd.DataFrame()
        print("No new configurations to run.")
    
    # Save results
    if len(merged_df) > 0:
        save_csv(results_path, merged_df)
        print(f"\nSaved results_multifreq.csv ({len(merged_df)} configurations)")
    
    # Save run log
    run_log = {
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "executed_configs": len(configs_to_run),
        "skipped_configs": len(configs_skipped),
        "total_configs": len(merged_df),
        "skipped_keys": [k.to_tuple() for k in configs_skipped[:20]],
    }
    save_json(output_dir / "run_log_multifreq.json", run_log)
    print("Saved run_log_multifreq.json")
    
    # Generate summary
    if len(merged_df) > 0:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_configurations": len(merged_df),
            "variants": list(merged_df['variant'].unique()),
            "key_findings": {},
        }
        
        # Compare variants
        for gamma in config.gamma_values:
            gamma_summary = {}
            for variant in merged_df['variant'].unique():
                v_df = merged_df[(merged_df['gamma'] == gamma) & (merged_df['variant'] == variant)]
                if len(v_df) > 0:
                    gamma_summary[variant] = {
                        "mean_acceptance": float(v_df['acceptance_probability'].mean()),
                        "mean_cost_per_accepted": float(v_df['cost_per_accepted'].replace([float('inf')], float('nan')).mean()),
                        "mean_steps": float(v_df['mean_steps_until_decision'].mean()),
                        "mean_hf_trigger_rate": float(v_df['hf_trigger_rate'].mean()),
                    }
            summary["key_findings"][f"gamma_{gamma}"] = gamma_summary
        
        save_json(output_dir / "summary_multifreq.json", summary)
        print("Saved summary_multifreq.json")
        
        # Generate plots
        print("\nGenerating plots...")
        plot_cost_per_accepted(merged_df, figures_dir)
        print("  - cost_per_accepted_vs_N_by_gamma.png/svg")
        
        plot_acceptance_vs_N(merged_df, figures_dir)
        print("  - acceptance_vs_N_by_gamma.png/svg")
        
        plot_mean_steps_vs_gamma(merged_df, figures_dir)
        print("  - mean_steps_until_decision_vs_gamma.png/svg")
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
