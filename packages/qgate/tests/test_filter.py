"""Tests for qgate.filter — TrajectoryFilter."""

from __future__ import annotations

import pytest

from qgate.adapters.base import MockAdapter
from qgate.config import (
    DynamicThresholdConfig,
    GateConfig,
)
from qgate.filter import TrajectoryFilter
from qgate.run_logging import FilterResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_filter(
    variant: str = "score_fusion",
    error_rate: float = 0.05,
    shots: int = 200,
    seed: int = 42,
    **kwargs,
) -> TrajectoryFilter:
    config = GateConfig(
        n_subsystems=4,
        n_cycles=2,
        shots=shots,
        variant=variant,
        **kwargs,
    )
    adapter = MockAdapter(error_rate=error_rate, seed=seed)
    return TrajectoryFilter(config, adapter)


# ---------------------------------------------------------------------------
# TrajectoryFilter.run
# ---------------------------------------------------------------------------


class TestTrajectoryFilterRun:
    def test_basic_run(self):
        tf = _make_filter()
        result = tf.run()
        assert isinstance(result, FilterResult)
        assert result.total_shots == 200
        assert result.accepted_shots >= 0
        assert 0.0 <= result.acceptance_probability <= 1.0

    def test_run_global(self):
        tf = _make_filter(variant="global", error_rate=0.0)
        result = tf.run()
        # zero error → all should be accepted
        assert result.accepted_shots == 200

    def test_run_hierarchical(self):
        tf = _make_filter(variant="hierarchical", k_fraction=0.5, error_rate=0.0)
        result = tf.run()
        assert result.accepted_shots == 200

    def test_run_score_fusion(self):
        tf = _make_filter(variant="score_fusion")
        result = tf.run()
        assert result.variant == "score_fusion"
        assert result.mean_combined_score is not None

    def test_zero_error_all_accepted(self):
        tf = _make_filter(error_rate=0.0)
        result = tf.run()
        assert result.acceptance_probability == pytest.approx(1.0)
        assert result.tts == pytest.approx(1.0)

    def test_full_error_global(self):
        tf = _make_filter(variant="global", error_rate=1.0)
        result = tf.run()
        assert result.accepted_shots == 0
        assert result.tts == float("inf")


# ---------------------------------------------------------------------------
# TrajectoryFilter.filter
# ---------------------------------------------------------------------------


class TestTrajectoryFilterFilter:
    def test_filter_prebuilt_outcomes(self):
        adapter = MockAdapter(error_rate=0.05, seed=42)
        outcomes = adapter.build_and_run(4, 2, 100)
        config = GateConfig(n_subsystems=4, n_cycles=2, shots=100)
        tf = TrajectoryFilter(config, adapter)
        result = tf.filter(outcomes)
        assert result.total_shots == 100


# ---------------------------------------------------------------------------
# Dynamic threshold integration
# ---------------------------------------------------------------------------


class TestDynamicThreshold:
    def test_dynamic_threshold_run(self):
        config = GateConfig(
            n_subsystems=4,
            n_cycles=2,
            shots=200,
            variant="score_fusion",
            dynamic_threshold=DynamicThresholdConfig(
                enabled=True,
                baseline=0.65,
                z_factor=1.0,
                window_size=5,
            ),
        )
        adapter = MockAdapter(error_rate=0.1, seed=42)
        tf = TrajectoryFilter(config, adapter)
        result = tf.run()
        assert result.dynamic_threshold_final is not None
        assert isinstance(result.dynamic_threshold_final, float)

    def test_threshold_property(self):
        config = GateConfig(
            dynamic_threshold=DynamicThresholdConfig(enabled=True, baseline=0.5),
        )
        adapter = MockAdapter(seed=42)
        tf = TrajectoryFilter(config, adapter)
        assert tf.current_threshold == 0.5

    def test_reset_threshold(self):
        config = GateConfig(
            dynamic_threshold=DynamicThresholdConfig(enabled=True, baseline=0.5),
        )
        adapter = MockAdapter(seed=42)
        tf = TrajectoryFilter(config, adapter)
        tf.run()
        tf.reset_threshold()
        assert tf.current_threshold == 0.5


# ---------------------------------------------------------------------------
# FilterResult
# ---------------------------------------------------------------------------


class TestFilterResult:
    def test_result_has_all_fields(self):
        tf = _make_filter()
        result = tf.run()
        assert hasattr(result, "run_id")
        assert hasattr(result, "variant")
        assert hasattr(result, "total_shots")
        assert hasattr(result, "accepted_shots")
        assert hasattr(result, "acceptance_probability")
        assert hasattr(result, "tts")
        assert hasattr(result, "mean_combined_score")
        assert hasattr(result, "threshold_used")
        assert hasattr(result, "scores")
        assert hasattr(result, "config_json")
        assert hasattr(result, "timestamp")

    def test_as_dict(self):
        tf = _make_filter()
        result = tf.run()
        d = result.as_dict()
        assert "variant" in d
        assert "scores" not in d  # scores excluded from summary dict

    def test_config_json_is_valid(self):
        import json

        tf = _make_filter()
        result = tf.run()
        parsed = json.loads(result.config_json)
        assert parsed["n_subsystems"] == 4

    def test_reproducible(self):
        r1 = _make_filter(seed=77).run()
        r2 = _make_filter(seed=77).run()
        assert r1.accepted_shots == r2.accepted_shots
        assert r1.scores == r2.scores
