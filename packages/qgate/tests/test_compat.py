"""Tests for qgate.compat — backward-compatible re-exports."""

from __future__ import annotations


class TestConditioningCompat:
    """Verify that qgate.conditioning still works as a public import."""

    def test_import_from_top_level(self):
        from qgate.conditioning import (
            ConditioningStats,
            ParityOutcome,
            apply_rule_to_batch,
            decide_global,
            decide_hierarchical,
            decide_score_fusion,
        )

        # Smoke-check: all symbols are callable / classes
        assert callable(decide_global)
        assert callable(decide_hierarchical)
        assert callable(decide_score_fusion)
        assert callable(apply_rule_to_batch)
        assert ParityOutcome is not None
        assert ConditioningStats is not None

    def test_import_from_compat(self):
        from qgate.compat.conditioning import (
            ConditioningStats,
            ParityOutcome,
            apply_rule_to_batch,
            decide_global,
            decide_hierarchical,
            decide_score_fusion,
        )

        assert callable(decide_global)
        assert callable(decide_hierarchical)
        assert callable(decide_score_fusion)
        assert callable(apply_rule_to_batch)
        assert ParityOutcome is not None
        assert ConditioningStats is not None

    def test_same_objects(self):
        import qgate.compat.conditioning as compat
        import qgate.conditioning as legacy

        assert legacy.ParityOutcome is compat.ParityOutcome
        assert legacy.decide_global is compat.decide_global
        assert legacy.ConditioningStats is compat.ConditioningStats


class TestMonitorsCompat:
    """Verify that qgate.monitors still works as a public import."""

    def test_import_from_top_level(self):
        from qgate.monitors import (
            MultiRateMonitor,
            compute_window_metric,
            score_fusion,
            should_abort_batch,
        )

        assert callable(compute_window_metric)
        assert callable(score_fusion)
        assert callable(should_abort_batch)
        assert MultiRateMonitor is not None

    def test_import_from_compat(self):
        from qgate.compat.monitors import (
            MultiRateMonitor,
            compute_window_metric,
            score_fusion,
            should_abort_batch,
        )

        assert callable(compute_window_metric)
        assert callable(score_fusion)
        assert callable(should_abort_batch)
        assert MultiRateMonitor is not None

    def test_same_objects(self):
        import qgate.compat.monitors as compat
        import qgate.monitors as legacy

        assert legacy.MultiRateMonitor is compat.MultiRateMonitor
        assert legacy.score_fusion is compat.score_fusion
