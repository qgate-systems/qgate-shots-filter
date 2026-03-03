"""Tests for qgate.threshold — DynamicThreshold."""

from __future__ import annotations

import pytest

from qgate.config import DynamicThresholdConfig
from qgate.threshold import DynamicThreshold


class TestDynamicThreshold:
    def test_initial_state(self):
        cfg = DynamicThresholdConfig(enabled=True, baseline=0.65)
        dt = DynamicThreshold(cfg)
        assert dt.current_threshold == 0.65
        assert dt.history == []

    def test_disabled_stays_at_baseline(self):
        cfg = DynamicThresholdConfig(enabled=False, baseline=0.70)
        dt = DynamicThreshold(cfg)
        dt.update(0.80)
        dt.update(0.90)
        assert dt.current_threshold == 0.70

    def test_enabled_adapts(self):
        cfg = DynamicThresholdConfig(
            enabled=True,
            baseline=0.65,
            z_factor=1.0,
            window_size=5,
            min_threshold=0.3,
            max_threshold=0.95,
        )
        dt = DynamicThreshold(cfg)
        # First update: only 1 sample → stays at baseline
        dt.update(0.70)
        assert dt.current_threshold == 0.65

        # Second update: now we have 2 samples
        dt.update(0.80)
        assert dt.current_threshold > 0.65  # should adapt upward

    def test_threshold_clamp_min(self):
        cfg = DynamicThresholdConfig(
            enabled=True,
            baseline=0.10,
            z_factor=0.0,
            min_threshold=0.5,
            max_threshold=0.9,
        )
        dt = DynamicThreshold(cfg)
        dt.update(0.20)
        dt.update(0.20)
        # mean=0.2, std*0=0 → raw=0.2 → clamped to min=0.5
        assert dt.current_threshold == pytest.approx(0.5)

    def test_threshold_clamp_max(self):
        cfg = DynamicThresholdConfig(
            enabled=True,
            baseline=0.65,
            z_factor=10.0,
            min_threshold=0.3,
            max_threshold=0.9,
        )
        dt = DynamicThreshold(cfg)
        dt.update(0.95)
        dt.update(0.95)
        # high z_factor → raw > max → clamped
        assert dt.current_threshold == pytest.approx(0.9)

    def test_reset(self):
        cfg = DynamicThresholdConfig(enabled=True, baseline=0.65)
        dt = DynamicThreshold(cfg)
        dt.update(0.80)
        dt.update(0.90)
        dt.reset()
        assert dt.current_threshold == 0.65
        assert dt.history == []

    def test_window_size_limit(self):
        cfg = DynamicThresholdConfig(enabled=True, window_size=3)
        dt = DynamicThreshold(cfg)
        for v in [0.5, 0.6, 0.7, 0.8, 0.9]:
            dt.update(v)
        assert len(dt.history) == 3  # only last 3 kept

    def test_history_returns_copy(self):
        cfg = DynamicThresholdConfig(enabled=True)
        dt = DynamicThreshold(cfg)
        dt.update(0.5)
        h = dt.history
        h.append(999)
        assert 999 not in dt.history
