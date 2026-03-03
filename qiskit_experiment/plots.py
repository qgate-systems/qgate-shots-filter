"""
plots.py – Visualisation for patent appendix figures.

Generates:
  1. Acceptance probability vs N  (one panel per D, grouped by variant).
  2. TTS vs N                     (same layout).
  3. Score-fusion acceptance vs α  at high D (D=8), per N.
  4. Probe pass-rate heatmap      (N × D).

All figures saved as PNGs (300 dpi) into <output_dir>/patent_appendix/.

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")            # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "global":         "#2196F3",   # blue
    "hierarchical":   "#4CAF50",   # green  (averaged across k)
    "score_fusion":   "#FF9800",   # orange (averaged across α)
}

VARIANT_LABELS = {
    "global":       "Global (all pass)",
    "hierarchical": "Hierarchical k-of-N",
    "score_fusion": "Score fusion (α-blend)",
}


def _empty_figure(msg: str, path: Path) -> None:
    """Save a placeholder figure when there is no data to plot."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color="gray")
    ax.set_axis_off()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_all_plots(
    results: list[dict[str, Any]],
    output_dir: str = "patent_appendix",
) -> list[str]:
    """Generate all patent-appendix figures. Returns list of file paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    # Filter out batch-aborted rows for main plots (show them separately)
    df_active = df[df["batch_aborted"] == False].copy()  # noqa: E712

    saved: list[str] = []
    saved.append(_plot_acceptance_vs_N(df_active, out))
    saved.append(_plot_tts_vs_N(df_active, out))
    saved.append(_plot_fusion_vs_alpha(df_active, out))
    saved.append(_plot_probe_heatmap(df, out))
    saved.append(_plot_acceptance_by_depth(df_active, out))

    print(f"Saved {len(saved)} figures to {out.resolve()}")
    return saved


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def _plot_acceptance_vs_N(df: pd.DataFrame, out: Path) -> str:
    """Acceptance probability vs N, faceted by D, one line per variant."""
    path = out / "acceptance_vs_N.png"
    if df.empty:
        _empty_figure("No active (non-aborted) data for acceptance vs N", path)
        return str(path)

    D_values = sorted(df["D"].unique())
    W_values = sorted(df["W"].unique())

    fig, axes = plt.subplots(1, max(1, len(D_values)),
                             figsize=(5 * max(1, len(D_values)), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, D in zip(axes, D_values):
        for variant, color in COLORS.items():
            sub = df[(df["D"] == D) & (df["variant"] == variant)]
            if sub.empty:
                continue
            # Average across W, k, α for this variant
            grouped = sub.groupby("N")["acceptance_probability"].mean()
            ax.plot(grouped.index, grouped.values, "o-", color=color,
                    label=VARIANT_LABELS.get(variant, variant), linewidth=2,
                    markersize=6)

        ax.set_title(f"D = {D}", fontsize=13, fontweight="bold")
        ax.set_xlabel("N (subsystems)", fontsize=11)
        ax.set_xticks(sorted(df["N"].unique()))
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Acceptance probability", fontsize=11)

    axes[-1].legend(fontsize=9, loc="best")
    fig.suptitle("Acceptance Probability vs System Size\n"
                 "US 63/983,831 & 63/989,632 | IL 326915", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_tts_vs_N(df: pd.DataFrame, out: Path) -> str:
    """Time-to-solution (1/P_accept) vs N, faceted by D."""
    path = out / "tts_vs_N.png"
    if df.empty:
        _empty_figure("No active data for TTS vs N", path)
        return str(path)

    D_values = sorted(df["D"].unique())

    fig, axes = plt.subplots(1, max(1, len(D_values)),
                             figsize=(5 * max(1, len(D_values)), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, D in zip(axes, D_values):
        for variant, color in COLORS.items():
            sub = df[(df["D"] == D) & (df["variant"] == variant)]
            if sub.empty:
                continue
            grouped = sub.groupby("N")["TTS"].mean()
            # Cap TTS for plotting (inf → large value)
            vals = grouped.values.copy()
            cap = 1000.0
            vals = np.where(np.isinf(vals), cap, vals)
            ax.semilogy(grouped.index, vals, "s--", color=color,
                        label=VARIANT_LABELS.get(variant, variant),
                        linewidth=2, markersize=6)

        ax.set_title(f"D = {D}", fontsize=13, fontweight="bold")
        ax.set_xlabel("N (subsystems)", fontsize=11)
        ax.set_xticks(sorted(df["N"].unique()))
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("TTS (1 / acceptance prob.)", fontsize=11)

    axes[-1].legend(fontsize=9, loc="best")
    fig.suptitle("Time-to-Solution vs System Size\n"
                 "US 63/983,831 & 63/989,632 | IL 326915", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_fusion_vs_alpha(df: pd.DataFrame, out: Path) -> str:
    """Score-fusion acceptance vs α at the highest D, grouped by N."""
    D_max = df["D"].max()
    sub = df[(df["variant"] == "score_fusion") & (df["D"] == D_max)]
    if sub.empty:
        # Nothing to plot
        path = out / "fusion_vs_alpha.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No score-fusion data at high D",
                transform=ax.transAxes, ha="center")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return str(path)

    N_values = sorted(sub["N"].unique())
    W_values = sorted(sub["W"].unique())

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(N_values)))

    for i, N in enumerate(N_values):
        s = sub[sub["N"] == N]
        grouped = s.groupby("alpha")["acceptance_probability"].mean()
        ax.plot(grouped.index, grouped.values, "o-", color=cmap[i],
                label=f"N={N}", linewidth=2, markersize=7)

    ax.set_xlabel("α (LF weight in fusion)", fontsize=12)
    ax.set_ylabel("Acceptance probability", fontsize=12)
    ax.set_title(f"Score Fusion: Acceptance vs α  (D={D_max})\n"
                 f"US 63/983,831 & 63/989,632 | IL 326915", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()

    path = out / "fusion_vs_alpha.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_probe_heatmap(df: pd.DataFrame, out: Path) -> str:
    """Probe pass-rate heatmap (N × D)."""
    # Use only one row per (N, D) – probe rate is the same for all variants
    sub = df.drop_duplicates(subset=["N", "D"])[["N", "D", "probe_pass_rate"]]
    sub = sub.dropna(subset=["probe_pass_rate"])

    if sub.empty:
        path = out / "probe_heatmap.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No probe data", transform=ax.transAxes, ha="center")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return str(path)

    # Average probe rate across W
    pivot = sub.groupby(["N", "D"])["probe_pass_rate"].mean().reset_index()
    pivot_table = pivot.pivot(index="N", columns="D", values="probe_pass_rate")

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot_table.values, cmap="RdYlGn", vmin=0, vmax=1,
                   aspect="auto")

    ax.set_xticks(range(len(pivot_table.columns)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticks(range(len(pivot_table.index)))
    ax.set_yticklabels(pivot_table.index)
    ax.set_xlabel("Scramble depth D", fontsize=11)
    ax.set_ylabel("N (subsystems)", fontsize=11)

    # Annotate cells
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            val = pivot_table.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if val < 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Probe pass-rate")
    ax.set_title("Batch-Level Abort: Probe Pass-Rate\n"
                 "US 63/983,831 & 63/989,632 | IL 326915", fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = out / "probe_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_acceptance_by_depth(df: pd.DataFrame, out: Path) -> str:
    """Acceptance vs D for each variant, faceted by N."""
    path = out / "acceptance_vs_depth.png"
    if df.empty:
        _empty_figure("No active data for acceptance vs depth", path)
        return str(path)

    N_values = sorted(df["N"].unique())

    fig, axes = plt.subplots(1, max(1, len(N_values)),
                             figsize=(4.5 * max(1, len(N_values)), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, N in zip(axes, N_values):
        for variant, color in COLORS.items():
            sub = df[(df["N"] == N) & (df["variant"] == variant)]
            if sub.empty:
                continue
            grouped = sub.groupby("D")["acceptance_probability"].mean()
            ax.plot(grouped.index, grouped.values, "o-", color=color,
                    label=VARIANT_LABELS.get(variant, variant),
                    linewidth=2, markersize=6)

        ax.set_title(f"N = {N}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Scramble depth D", fontsize=11)
        ax.set_xticks(sorted(df["D"].unique()))
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Acceptance probability", fontsize=11)

    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Acceptance vs Scramble Depth\n"
                 "US 63/983,831 & 63/989,632 | IL 326915", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)
