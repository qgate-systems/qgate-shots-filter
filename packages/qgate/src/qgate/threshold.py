"""
threshold.py — Dynamic threshold gating strategies.

Provides two adaptive threshold classes:

:class:`DynamicThreshold`
    Rolling z-score gating (legacy ``rolling_z`` mode).  Operates on
    **batch-level** mean scores.

:class:`GaltonAdaptiveThreshold`
    Distribution-aware gating (``galton`` mode) inspired by diffusion /
    central-limit principles.  Operates on **per-shot** combined scores
    and supports empirical-quantile and robust z-score sub-modes.

Both classes share the same :class:`~qgate.config.DynamicThresholdConfig`
and are wired into :class:`~qgate.filter.TrajectoryFilter` transparently.

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from qgate.config import DynamicThresholdConfig

logger = logging.getLogger("qgate.threshold")

# ═══════════════════════════════════════════════════════════════════════════
# MAD constant — converts MAD to a consistent sigma estimator for normal data.
# ═══════════════════════════════════════════════════════════════════════════
_MAD_TO_SIGMA: float = 1.4826


# ═══════════════════════════════════════════════════════════════════════════
# GaltonSnapshot — lightweight telemetry snapshot
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GaltonSnapshot:
    """Telemetry snapshot emitted after every :meth:`GaltonAdaptiveThreshold.update`.

    All fields are populated regardless of the active sub-mode; fields
    that do not apply in the current mode are set to ``None``.

    Attributes:
        rolling_mean:           Mean of the rolling window.
        rolling_sigma:          Std-dev (or MAD-based σ) of the window.
        rolling_quantile:       Empirical quantile at 1 − target_acceptance.
        effective_threshold:    Threshold actually used for gating.
        window_size_current:    Number of scores in the window right now.
        acceptance_rate_rolling: Fraction of window scores ≥ threshold.
        in_warmup:              True if window < min_window_size.
    """

    rolling_mean: float | None = None
    rolling_sigma: float | None = None
    rolling_quantile: float | None = None
    effective_threshold: float = 0.65
    window_size_current: int = 0
    acceptance_rate_rolling: float | None = None
    in_warmup: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# DynamicThreshold — legacy rolling-z (unchanged interface)
# ═══════════════════════════════════════════════════════════════════════════


class DynamicThreshold:
    """Rolling z-score threshold adjuster.

    Maintains a sliding window of recent batch scores and computes an
    adaptive threshold each time :meth:`update` is called.

    Args:
        config: Threshold configuration parameters.

    Example::

        from qgate.config import DynamicThresholdConfig
        cfg = DynamicThresholdConfig(enabled=True, baseline=0.65,
                                      z_factor=1.5, window_size=10)
        dt = DynamicThreshold(cfg)
        dt.update(0.70)
        dt.update(0.68)
        print(dt.current_threshold)
    """

    def __init__(self, config: DynamicThresholdConfig) -> None:
        self._config = config
        self._history: deque[float] = deque(maxlen=config.window_size)
        self._current: float = config.baseline

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> DynamicThresholdConfig:
        return self._config

    @property
    def current_threshold(self) -> float:
        """The most recent threshold value."""
        return self._current

    @property
    def history(self) -> list[float]:
        """Copy of the rolling score history."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def update(self, batch_score: float) -> float:
        """Record a new batch score and recompute the threshold.

        Args:
            batch_score: The mean combined score of the latest batch.

        Returns:
            The updated threshold value.
        """
        self._history.append(batch_score)

        if not self._config.enabled or len(self._history) < 2:
            self._current = self._config.baseline
            return self._current

        arr = np.array(self._history)
        rolling_mean = float(np.mean(arr))
        rolling_std = float(np.std(arr, ddof=1))

        raw = rolling_mean + self._config.z_factor * rolling_std
        clamped = max(self._config.min_threshold, min(self._config.max_threshold, raw))

        self._current = float(clamped)
        return self._current

    def reset(self) -> None:
        """Clear history and reset to baseline."""
        self._history.clear()
        self._current = self._config.baseline


# ═══════════════════════════════════════════════════════════════════════════
# GaltonAdaptiveThreshold — distribution-aware gating
# ═══════════════════════════════════════════════════════════════════════════

class GaltonAdaptiveThreshold:
    """Distribution-aware adaptive threshold (Galton / diffusion mode).

    Maintains a **per-shot** rolling window of combined scores and
    computes a threshold that targets a stable acceptance fraction.

    Two sub-modes are available (selected via ``config.use_quantile``):

    **Quantile mode** (default, recommended)
        Uses the empirical quantile of the window:

        .. math:: \\theta = Q_{1 - \\text{target\\_acceptance}}(\\text{window})

        This is the most robust option — it makes no distributional
        assumptions and naturally tracks hardware drift.

    **Z-score mode** (``use_quantile=False``)
        Estimates μ and σ from the window and sets:

        .. math:: \\theta = \\mu + z_{\\sigma} \\cdot \\sigma

        When ``robust_stats=True`` (default), the median and
        MAD-derived σ are used, making the estimate resilient to
        outliers.  When ``robust_stats=False``, ordinary mean and
        sample std are used.

    **Warmup:** While ``len(window) < min_window_size`` the threshold
    falls back to ``config.baseline``.  This avoids noisy estimates
    from too few observations.

    All operations are O(1) amortised — the window is backed by a
    :class:`collections.deque` with bounded capacity.

    Args:
        config: :class:`~qgate.config.DynamicThresholdConfig` with
                ``mode="galton"`` (or ``"diffusion"``).

    Example::

        from qgate.config import DynamicThresholdConfig
        cfg = DynamicThresholdConfig(
            mode="galton",
            window_size=500,
            target_acceptance=0.05,
            robust_stats=True,
            use_quantile=True,
        )
        gat = GaltonAdaptiveThreshold(cfg)
        for score in batch_scores:
            gat.observe(score)
        print(gat.current_threshold)
    """

    def __init__(self, config: DynamicThresholdConfig) -> None:
        self._config = config
        self._window: deque[float] = deque(maxlen=config.window_size)
        self._current: float = config.baseline
        self._last_snapshot: GaltonSnapshot = GaltonSnapshot(
            effective_threshold=config.baseline,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> DynamicThresholdConfig:
        return self._config

    @property
    def current_threshold(self) -> float:
        """The most recent effective threshold value."""
        return self._current

    @property
    def window(self) -> list[float]:
        """Copy of the rolling score window."""
        return list(self._window)

    @property
    def window_size_current(self) -> int:
        """Number of scores currently in the window."""
        return len(self._window)

    @property
    def in_warmup(self) -> bool:
        """True while the window is smaller than ``min_window_size``."""
        return len(self._window) < self._config.min_window_size

    @property
    def last_snapshot(self) -> GaltonSnapshot:
        """The most recent telemetry snapshot."""
        return self._last_snapshot

    # ------------------------------------------------------------------
    # Core — observe individual scores
    # ------------------------------------------------------------------

    def observe(self, score: float) -> float:
        """Add a single score to the window and recompute the threshold.

        Call this once per shot (or per combined score).  The threshold
        is evaluated *before* the new score is appended, so the
        returned threshold was computed without the new score.

        Args:
            score: A per-shot combined score.

        Returns:
            The effective threshold after update.
        """
        # Append first, then recompute
        self._window.append(score)
        self._recompute()
        return self._current

    def observe_batch(self, scores: list[float] | np.ndarray) -> float:
        """Convenience: observe a whole batch of scores at once.

        Args:
            scores: Iterable of per-shot combined scores.

        Returns:
            The effective threshold after the last observation.
        """
        for s in scores:
            self._window.append(float(s))
        self._recompute()
        return self._current

    def reset(self) -> None:
        """Clear the window and reset to baseline."""
        self._window.clear()
        self._current = self._config.baseline
        self._last_snapshot = GaltonSnapshot(
            effective_threshold=self._config.baseline,
        )

    # ------------------------------------------------------------------
    # Private: threshold computation
    # ------------------------------------------------------------------

    def _recompute(self) -> None:
        """Recompute the threshold from the current window."""
        n = len(self._window)

        # Warmup: not enough data yet — fall back to baseline
        if n < self._config.min_window_size:
            self._current = self._config.baseline
            self._last_snapshot = GaltonSnapshot(
                effective_threshold=self._current,
                window_size_current=n,
                in_warmup=True,
            )
            logger.debug(
                "Galton warmup: %d / %d samples — using baseline %.4f",
                n, self._config.min_window_size, self._current,
            )
            return

        arr = np.asarray(self._window, dtype=np.float64)

        # --- Statistics (always computed for telemetry) ----------------
        if self._config.robust_stats:
            mu = float(np.median(arr))
            mad = float(np.median(np.abs(arr - mu)))
            sigma = mad * _MAD_TO_SIGMA
        else:
            mu = float(np.mean(arr))
            sigma = float(np.std(arr, ddof=1)) if n > 1 else 0.0

        quantile_val = float(
            np.quantile(arr, 1.0 - self._config.target_acceptance)
        )

        # --- Threshold selection --------------------------------------
        raw = (
            quantile_val
            if self._config.use_quantile
            else mu + self._config.z_sigma * sigma
        )

        clamped = float(
            max(self._config.min_threshold, min(self._config.max_threshold, raw))
        )
        self._current = clamped

        # --- Acceptance rate ------------------------------------------
        accept_rate = float(np.mean(arr >= self._current))

        # --- Snapshot -------------------------------------------------
        self._last_snapshot = GaltonSnapshot(
            rolling_mean=mu,
            rolling_sigma=sigma,
            rolling_quantile=quantile_val,
            effective_threshold=self._current,
            window_size_current=n,
            acceptance_rate_rolling=accept_rate,
            in_warmup=False,
        )

        logger.debug(
            "Galton threshold: %.4f  (μ=%.4f, σ=%.4f, Q=%.4f, "
            "accept=%.3f, window=%d)",
            self._current, mu, sigma, quantile_val, accept_rate, n,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Utility — diffusion width estimation
# ═══════════════════════════════════════════════════════════════════════════

def estimate_diffusion_width(
    window: list[float] | np.ndarray,
    robust: bool = True,
) -> float:
    """Estimate the variance (diffusion width) of a score window.

    This is a simple dispersion estimator that can serve as a diagnostic
    for diffusion-scaling validation in future work.

    When ``robust=True`` (default) the MAD-based σ² is returned; otherwise
    the ordinary sample variance is used.

    Args:
        window: 1-D array-like of scores.
        robust: Use MAD-derived variance estimate.

    Returns:
        Estimated variance (σ²).

    Raises:
        ValueError: If *window* has fewer than 2 elements.
    """
    arr = np.asarray(window, dtype=np.float64)
    if arr.size < 2:
        raise ValueError(
            f"estimate_diffusion_width requires ≥ 2 observations, got {arr.size}"
        )
    if robust:
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        sigma = mad * _MAD_TO_SIGMA
        return sigma * sigma
    return float(np.var(arr, ddof=1))
