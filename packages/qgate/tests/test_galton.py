"""Tests for qgate.threshold — GaltonAdaptiveThreshold & estimate_diffusion_width."""

from __future__ import annotations

import numpy as np
import pytest

from qgate.config import DynamicThresholdConfig, GateConfig
from qgate.threshold import (
    GaltonAdaptiveThreshold,
    GaltonSnapshot,
    estimate_diffusion_width,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _galton_cfg(**overrides) -> DynamicThresholdConfig:
    """Build a Galton-mode config with sensible test defaults."""
    defaults = dict(
        mode="galton",
        window_size=500,
        min_window_size=100,
        target_acceptance=0.05,
        robust_stats=True,
        use_quantile=True,
        min_threshold=0.0,
        max_threshold=1.0,
    )
    defaults.update(overrides)
    return DynamicThresholdConfig(**defaults)


# ---------------------------------------------------------------------------
# GaltonAdaptiveThreshold — quantile mode
# ---------------------------------------------------------------------------


class TestGaltonQuantileMode:
    """Verify quantile-based gating targets the expected acceptance."""

    def test_acceptance_approximately_target(self):
        """With N(0.7, 0.05) scores, ~5% should exceed the threshold."""
        rng = np.random.default_rng(42)
        cfg = _galton_cfg(
            window_size=2000,
            min_window_size=200,
            target_acceptance=0.05,
            use_quantile=True,
        )
        gat = GaltonAdaptiveThreshold(cfg)

        # Feed 2000 scores from a normal distribution
        scores = rng.normal(0.7, 0.05, size=2000).clip(0, 1).tolist()
        gat.observe_batch(scores)

        theta = gat.current_threshold
        # The threshold should be near the 95th percentile of N(0.7, 0.05)
        # ≈ 0.7 + 1.645 * 0.05 ≈ 0.782
        assert 0.72 < theta < 0.85, f"theta={theta:.4f}"
        assert not gat.in_warmup

        # Actual acceptance rate of the window should be near 5%
        snap = gat.last_snapshot
        assert snap.acceptance_rate_rolling is not None
        assert 0.01 < snap.acceptance_rate_rolling < 0.15

    def test_different_target_acceptance(self):
        """Target 20% acceptance → lower threshold."""
        rng = np.random.default_rng(99)
        cfg_5 = _galton_cfg(target_acceptance=0.05, window_size=1000, min_window_size=100)
        cfg_20 = _galton_cfg(target_acceptance=0.20, window_size=1000, min_window_size=100)

        scores = rng.normal(0.7, 0.05, size=1000).clip(0, 1).tolist()

        g5 = GaltonAdaptiveThreshold(cfg_5)
        g5.observe_batch(scores)
        g20 = GaltonAdaptiveThreshold(cfg_20)
        g20.observe_batch(scores)

        # More permissive target → lower threshold
        assert g20.current_threshold < g5.current_threshold


# ---------------------------------------------------------------------------
# GaltonAdaptiveThreshold — robust z-score mode
# ---------------------------------------------------------------------------


class TestGaltonRobustMode:
    """Verify robust stats (MAD-based) are resilient to outliers."""

    def test_robust_threshold_stable_with_outliers(self):
        """Inject extreme outliers; robust threshold stays reasonable."""
        rng = np.random.default_rng(123)
        cfg = _galton_cfg(
            use_quantile=False,
            robust_stats=True,
            z_sigma=1.645,
            window_size=500,
            min_window_size=50,
        )
        gat = GaltonAdaptiveThreshold(cfg)

        # 480 normal scores + 20 extreme outliers
        clean = rng.normal(0.6, 0.04, size=480).clip(0, 1)
        outliers = np.ones(20) * 0.99
        scores = np.concatenate([clean, outliers]).tolist()
        rng.shuffle(scores)

        gat.observe_batch(scores)
        snap = gat.last_snapshot

        # Robust μ (median) should be near 0.6, not pulled to 0.99
        assert snap.rolling_mean is not None
        assert 0.55 < snap.rolling_mean < 0.65
        # Robust sigma should be modest
        assert snap.rolling_sigma is not None
        assert snap.rolling_sigma < 0.10

    def test_non_robust_affected_by_outliers(self):
        """Without robust stats, mean is pulled by outliers."""
        rng = np.random.default_rng(123)
        cfg_robust = _galton_cfg(
            use_quantile=False,
            robust_stats=True,
            window_size=500,
            min_window_size=50,
        )
        cfg_non_robust = _galton_cfg(
            use_quantile=False,
            robust_stats=False,
            window_size=500,
            min_window_size=50,
        )

        clean = rng.normal(0.5, 0.03, size=450).clip(0, 1)
        outliers = np.ones(50) * 0.99
        scores = np.concatenate([clean, outliers]).tolist()

        g_r = GaltonAdaptiveThreshold(cfg_robust)
        g_r.observe_batch(scores)
        g_nr = GaltonAdaptiveThreshold(cfg_non_robust)
        g_nr.observe_batch(scores)

        # Non-robust mean is pulled higher
        assert g_nr.last_snapshot.rolling_mean > g_r.last_snapshot.rolling_mean  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Warmup behaviour
# ---------------------------------------------------------------------------


class TestGaltonWarmup:
    """Verify threshold falls back to baseline during warmup."""

    def test_no_gating_before_min_window(self):
        cfg = _galton_cfg(min_window_size=100, baseline=0.65)
        gat = GaltonAdaptiveThreshold(cfg)

        # Feed 99 scores (one short of warmup)
        for _ in range(99):
            gat.observe(0.8)
        assert gat.in_warmup
        assert gat.current_threshold == 0.65
        snap = gat.last_snapshot
        assert snap.in_warmup is True

        # The 100th score exits warmup
        gat.observe(0.8)
        assert not gat.in_warmup
        assert gat.current_threshold != 0.65  # now adapting

    def test_warmup_snapshot_fields(self):
        cfg = _galton_cfg(min_window_size=50)
        gat = GaltonAdaptiveThreshold(cfg)
        gat.observe(0.7)
        snap = gat.last_snapshot
        assert snap.in_warmup is True
        assert snap.rolling_mean is None
        assert snap.rolling_sigma is None
        assert snap.window_size_current == 1


# ---------------------------------------------------------------------------
# Window management
# ---------------------------------------------------------------------------


class TestGaltonWindow:
    def test_window_bounded_by_maxlen(self):
        cfg = _galton_cfg(window_size=100, min_window_size=10)
        gat = GaltonAdaptiveThreshold(cfg)
        for i in range(500):
            gat.observe(float(i) / 500)
        assert gat.window_size_current == 100
        assert len(gat.window) == 100

    def test_window_returns_copy(self):
        cfg = _galton_cfg(min_window_size=1)
        gat = GaltonAdaptiveThreshold(cfg)
        gat.observe(0.5)
        w = gat.window
        w.append(999.0)
        assert 999.0 not in gat.window


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestGaltonReset:
    def test_reset_restores_baseline(self):
        cfg = _galton_cfg(baseline=0.70, min_window_size=10)
        gat = GaltonAdaptiveThreshold(cfg)
        for _ in range(200):
            gat.observe(0.9)
        assert gat.current_threshold != 0.70

        gat.reset()
        assert gat.current_threshold == 0.70
        assert gat.window_size_current == 0
        assert gat.in_warmup


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------


class TestGaltonClamping:
    def test_threshold_clamped_to_min(self):
        cfg = _galton_cfg(
            min_threshold=0.5,
            max_threshold=0.9,
            min_window_size=10,
            window_size=200,
            use_quantile=True,
            target_acceptance=0.95,  # very permissive → low threshold
        )
        gat = GaltonAdaptiveThreshold(cfg)
        rng = np.random.default_rng(7)
        scores = rng.uniform(0.1, 0.3, size=200).tolist()
        gat.observe_batch(scores)
        assert gat.current_threshold >= 0.5

    def test_threshold_clamped_to_max(self):
        cfg = _galton_cfg(
            min_threshold=0.3,
            max_threshold=0.8,
            min_window_size=10,
            window_size=200,
            use_quantile=True,
            target_acceptance=0.01,  # very strict → high threshold
        )
        gat = GaltonAdaptiveThreshold(cfg)
        rng = np.random.default_rng(8)
        scores = rng.uniform(0.85, 0.99, size=200).tolist()
        gat.observe_batch(scores)
        assert gat.current_threshold <= 0.8


# ---------------------------------------------------------------------------
# Config auto-enable
# ---------------------------------------------------------------------------


class TestGaltonConfig:
    def test_mode_galton_auto_enables(self):
        cfg = DynamicThresholdConfig(mode="galton")
        assert cfg.enabled is True

    def test_mode_rolling_z_auto_enables(self):
        cfg = DynamicThresholdConfig(mode="rolling_z")
        assert cfg.enabled is True

    def test_mode_fixed_stays_disabled(self):
        cfg = DynamicThresholdConfig(mode="fixed")
        assert cfg.enabled is False

    def test_mode_fixed_explicit_enabled(self):
        cfg = DynamicThresholdConfig(mode="fixed", enabled=True)
        assert cfg.enabled is True

    def test_galton_config_fields(self):
        cfg = DynamicThresholdConfig(
            mode="galton",
            window_size=1000,
            min_window_size=200,
            target_acceptance=0.10,
            robust_stats=False,
            use_quantile=False,
            z_sigma=2.0,
        )
        assert cfg.window_size == 1000
        assert cfg.min_window_size == 200
        assert cfg.target_acceptance == 0.10
        assert cfg.robust_stats is False
        assert cfg.use_quantile is False
        assert cfg.z_sigma == 2.0

    def test_galton_in_gate_config(self):
        gc = GateConfig(
            dynamic_threshold=DynamicThresholdConfig(
                mode="galton",
                window_size=500,
                target_acceptance=0.05,
            ),
        )
        assert gc.dynamic_threshold.mode == "galton"
        assert gc.dynamic_threshold.enabled is True


# ---------------------------------------------------------------------------
# Integration with TrajectoryFilter
# ---------------------------------------------------------------------------


class TestGaltonFilterIntegration:
    def test_galton_run(self):
        from qgate.adapters.base import MockAdapter
        from qgate.filter import TrajectoryFilter

        config = GateConfig(
            n_subsystems=4,
            n_cycles=2,
            shots=500,
            variant="score_fusion",
            dynamic_threshold=DynamicThresholdConfig(
                mode="galton",
                window_size=500,
                min_window_size=50,
                target_acceptance=0.10,
            ),
        )
        adapter = MockAdapter(error_rate=0.1, seed=42)
        tf = TrajectoryFilter(config, adapter)
        result = tf.run()

        assert result.total_shots == 500
        assert result.dynamic_threshold_final is not None
        assert isinstance(result.dynamic_threshold_final, float)
        # Galton metadata should be present
        assert "galton" in result.metadata
        galton = result.metadata["galton"]
        assert "galton_effective_threshold" in galton
        assert "galton_window_size_current" in galton

    def test_galton_threshold_property(self):
        from qgate.adapters.base import MockAdapter
        from qgate.filter import TrajectoryFilter

        config = GateConfig(
            shots=200,
            dynamic_threshold=DynamicThresholdConfig(
                mode="galton",
                baseline=0.60,
                min_window_size=10,
            ),
        )
        tf = TrajectoryFilter(config, MockAdapter(seed=42))
        # Before run: baseline
        assert tf.current_threshold == 0.60

        tf.run()
        # After run: adapted (likely different from baseline)
        theta = tf.current_threshold
        assert isinstance(theta, float)
        assert 0.0 <= theta <= 1.0

    def test_galton_reset(self):
        from qgate.adapters.base import MockAdapter
        from qgate.filter import TrajectoryFilter

        config = GateConfig(
            shots=200,
            dynamic_threshold=DynamicThresholdConfig(
                mode="galton",
                baseline=0.55,
                min_window_size=10,
            ),
        )
        tf = TrajectoryFilter(config, MockAdapter(seed=42))
        tf.run()
        tf.reset_threshold()
        assert tf.current_threshold == 0.55

    def test_galton_snapshot_property(self):
        from qgate.adapters.base import MockAdapter
        from qgate.filter import TrajectoryFilter

        config = GateConfig(
            shots=200,
            dynamic_threshold=DynamicThresholdConfig(
                mode="galton",
                min_window_size=10,
            ),
        )
        tf = TrajectoryFilter(config, MockAdapter(seed=42))
        tf.run()
        snap = tf.galton_snapshot
        assert snap is not None
        assert isinstance(snap, GaltonSnapshot)

    def test_non_galton_has_no_snapshot(self):
        from qgate.adapters.base import MockAdapter
        from qgate.filter import TrajectoryFilter

        config = GateConfig(shots=100)
        tf = TrajectoryFilter(config, MockAdapter(seed=42))
        assert tf.galton_snapshot is None


# ---------------------------------------------------------------------------
# estimate_diffusion_width
# ---------------------------------------------------------------------------


class TestEstimateDiffusionWidth:
    def test_basic_variance(self):
        rng = np.random.default_rng(55)
        data = rng.normal(0.5, 0.1, size=1000)
        var_est = estimate_diffusion_width(data, robust=False)
        assert 0.008 < var_est < 0.012  # ~ 0.01

    def test_robust_variance(self):
        rng = np.random.default_rng(55)
        data = rng.normal(0.5, 0.1, size=1000)
        var_est = estimate_diffusion_width(data, robust=True)
        assert 0.007 < var_est < 0.013

    def test_too_few_raises(self):
        with pytest.raises(ValueError, match="≥ 2"):
            estimate_diffusion_width([0.5])

    def test_list_input(self):
        var_est = estimate_diffusion_width([0.1, 0.2, 0.3, 0.4, 0.5])
        assert var_est > 0

    def test_robust_resists_outliers(self):
        clean = [0.5] * 98
        outlier = [0.5, 10.0]
        # Non-robust variance is huge
        var_nr = estimate_diffusion_width(clean + outlier, robust=False)
        var_r = estimate_diffusion_width(clean + outlier, robust=True)
        assert var_r < var_nr
