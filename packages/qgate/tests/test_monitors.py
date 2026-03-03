"""Tests for qgate.monitors module."""

from __future__ import annotations

import numpy as np
import pytest

from qgate.monitors import (
    MultiRateMonitor,
    compute_window_metric,
    score_fusion,
    should_abort_batch,
)

# ---------------------------------------------------------------------------
# compute_window_metric
# ---------------------------------------------------------------------------


class TestComputeWindowMetric:
    def test_max_mode(self):
        t = np.linspace(0, 10, 200)
        v = np.sin(t)
        metric, ws, we = compute_window_metric(t, v, window=2.0, mode="max")
        assert we == pytest.approx(10.0)
        assert ws == pytest.approx(8.0)
        # max of sin in [8, 10] ≈ sin(8.0..10.0)
        expected_max = float(np.max(v[(t >= 8.0) & (t <= 10.0)]))
        assert metric == pytest.approx(expected_max)

    def test_mean_mode(self):
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        v = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        metric, _ws, _we = compute_window_metric(t, v, window=2.0, mode="mean")
        # window [2, 4] → values 0.7, 0.8, 0.9 → mean = 0.8
        assert metric == pytest.approx(0.8)

    def test_zero_window(self):
        t = np.array([0.0, 1.0, 2.0])
        v = np.array([0.1, 0.2, 0.3])
        metric, _, _ = compute_window_metric(t, v, window=0.0, mode="max")
        assert metric == pytest.approx(0.3)  # just the last point

    def test_large_window_covers_all(self):
        t = np.array([0.0, 1.0, 2.0])
        v = np.array([0.5, 0.2, 0.8])
        metric, ws, _ = compute_window_metric(t, v, window=100.0, mode="max")
        assert ws == pytest.approx(0.0)
        assert metric == pytest.approx(0.8)

    def test_unknown_mode(self):
        t = np.array([0.0, 1.0])
        v = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="Unknown mode"):
            compute_window_metric(t, v, mode="median")


# ---------------------------------------------------------------------------
# score_fusion
# ---------------------------------------------------------------------------


class TestScoreFusion:
    def test_basic_fusion(self):
        accepted, score = score_fusion(0.8, 0.6, alpha=0.5, threshold=0.65)
        assert score == pytest.approx(0.7)
        assert accepted is True

    def test_below_threshold(self):
        accepted, score = score_fusion(0.2, 0.3, alpha=0.5, threshold=0.65)
        assert score == pytest.approx(0.25)
        assert accepted is False

    def test_alpha_zero(self):
        # all weight on HF
        accepted, score = score_fusion(0.0, 1.0, alpha=0.0, threshold=0.5)
        assert score == pytest.approx(1.0)
        assert accepted is True

    def test_alpha_one(self):
        # all weight on LF
        accepted, score = score_fusion(1.0, 0.0, alpha=1.0, threshold=0.5)
        assert score == pytest.approx(1.0)
        assert accepted is True

    def test_exact_threshold(self):
        accepted, score = score_fusion(0.65, 0.65, alpha=0.5, threshold=0.65)
        assert score == pytest.approx(0.65)
        assert accepted is True  # ≥ threshold


# ---------------------------------------------------------------------------
# MultiRateMonitor
# ---------------------------------------------------------------------------


class TestMultiRateMonitor:
    def test_record_and_fuse(self):
        mon = MultiRateMonitor(n_subsystems=4, alpha=0.5, threshold_combined=0.65)
        mon.record_cycle(0, 0.75)  # HF + LF
        mon.record_cycle(1, 0.50)  # HF only
        mon.record_cycle(2, 0.80)  # HF + LF
        accepted, score = mon.fused_decision()
        # LF: mean(0.75, 0.80) = 0.775
        # HF: mean(0.75, 0.50, 0.80) = 0.6833..
        # combined = 0.5*0.775 + 0.5*0.6833 = 0.7292
        assert score == pytest.approx(0.5 * 0.775 + 0.5 * (0.75 + 0.50 + 0.80) / 3)
        assert accepted is True

    def test_reset(self):
        mon = MultiRateMonitor(n_subsystems=2)
        mon.record_cycle(0, 0.9)
        mon.reset()
        assert len(mon.hf_scores) == 0
        assert len(mon.lf_scores) == 0

    def test_fused_decision_empty(self):
        mon = MultiRateMonitor()
        accepted, score = mon.fused_decision()
        assert score == pytest.approx(0.0)
        assert accepted is False

    def test_only_hf_cycles(self):
        mon = MultiRateMonitor(alpha=0.5, threshold_combined=0.0)
        mon.record_cycle(1, 0.6)  # odd → HF only
        mon.record_cycle(3, 0.8)  # odd → HF only
        _accepted, score = mon.fused_decision()
        # LF empty → 0.0;  HF mean = 0.7
        # combined = 0.5*0 + 0.5*0.7 = 0.35
        assert score == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# should_abort_batch
# ---------------------------------------------------------------------------


class TestShouldAbortBatch:
    def test_above_theta(self):
        assert should_abort_batch(0.80, theta=0.65) is False

    def test_below_theta(self):
        assert should_abort_batch(0.30, theta=0.65) is True

    def test_at_theta(self):
        assert should_abort_batch(0.65, theta=0.65) is False

    def test_zero_probe(self):
        assert should_abort_batch(0.0, theta=0.65) is True
