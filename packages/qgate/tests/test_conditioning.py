"""Tests for qgate.conditioning module."""

from __future__ import annotations

import pytest

from qgate.conditioning import (
    ConditioningStats,
    ParityOutcome,
    apply_rule_to_batch,
    decide_global,
    decide_hierarchical,
    decide_score_fusion,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _outcome(matrix: list[list[int]]) -> ParityOutcome:
    """Shortcut to build a ParityOutcome from a parity matrix."""
    n_cycles = len(matrix)
    n_sub = len(matrix[0]) if n_cycles else 0
    return ParityOutcome(n_subsystems=n_sub, n_cycles=n_cycles, parity_matrix=matrix)


# ---------------------------------------------------------------------------
# ParityOutcome
# ---------------------------------------------------------------------------


class TestParityOutcome:
    def test_subsystem_pass_count_all_pass(self):
        o = _outcome([[0, 0, 0, 0]])
        assert o.subsystem_pass_count(0) == 4

    def test_subsystem_pass_count_some_fail(self):
        o = _outcome([[0, 1, 0, 1]])
        assert o.subsystem_pass_count(0) == 2

    def test_subsystem_pass_rate(self):
        o = _outcome([[0, 1, 0, 1]])
        assert o.subsystem_pass_rate(0) == pytest.approx(0.5)

    def test_cycle_all_pass_true(self):
        o = _outcome([[0, 0, 0]])
        assert o.cycle_all_pass(0) is True

    def test_cycle_all_pass_false(self):
        o = _outcome([[0, 0, 1]])
        assert o.cycle_all_pass(0) is False


# ---------------------------------------------------------------------------
# decide_global
# ---------------------------------------------------------------------------


class TestDecideGlobal:
    def test_all_pass(self):
        o = _outcome([[0, 0, 0], [0, 0, 0]])
        assert decide_global(o) is True

    def test_one_failure(self):
        o = _outcome([[0, 0, 0], [0, 1, 0]])
        assert decide_global(o) is False

    def test_single_subsystem_single_cycle(self):
        assert decide_global(_outcome([[0]])) is True
        assert decide_global(_outcome([[1]])) is False

    def test_all_fail(self):
        o = _outcome([[1, 1], [1, 1]])
        assert decide_global(o) is False


# ---------------------------------------------------------------------------
# decide_hierarchical
# ---------------------------------------------------------------------------


class TestDecideHierarchical:
    def test_exact_threshold(self):
        # k_fraction=0.75, N=4 → ceil(3.0) = 3 needed
        o = _outcome([[0, 0, 0, 1]])  # 3 pass
        assert decide_hierarchical(o, k_fraction=0.75) is True

    def test_below_threshold(self):
        o = _outcome([[0, 0, 1, 1]])  # 2 pass, need 3
        assert decide_hierarchical(o, k_fraction=0.75) is False

    def test_all_pass(self):
        o = _outcome([[0, 0, 0, 0]])
        assert decide_hierarchical(o, k_fraction=0.9) is True

    def test_multiple_cycles(self):
        # cycle 0: 4 pass  → ok
        # cycle 1: 2 pass  → need ceil(0.5*4) = 2  → ok
        o = _outcome([[0, 0, 0, 0], [0, 0, 1, 1]])
        assert decide_hierarchical(o, k_fraction=0.5) is True

    def test_multiple_cycles_fails_one(self):
        # cycle 0: 4 pass  → ok
        # cycle 1: 1 pass  → need ceil(0.9*4) = 4  → FAIL
        o = _outcome([[0, 0, 0, 0], [0, 1, 1, 1]])
        assert decide_hierarchical(o, k_fraction=0.9) is False

    def test_k_fraction_out_of_range(self):
        o = _outcome([[0, 0]])
        with pytest.raises(ValueError, match="k_fraction"):
            decide_hierarchical(o, k_fraction=0.0)
        with pytest.raises(ValueError, match="k_fraction"):
            decide_hierarchical(o, k_fraction=1.5)

    def test_k_fraction_equals_one(self):
        # k_fraction=1.0 should behave like global (all must pass)
        o = _outcome([[0, 0, 1, 0]])
        assert decide_hierarchical(o, k_fraction=1.0) is False


# ---------------------------------------------------------------------------
# decide_score_fusion
# ---------------------------------------------------------------------------


class TestDecideScoreFusion:
    def test_all_pass_accepted(self):
        # All pass → rates = 1.0 → combined = 1.0
        o = _outcome([[0, 0], [0, 0], [0, 0], [0, 0]])
        accepted, score = decide_score_fusion(o, alpha=0.5, threshold_combined=0.5)
        assert accepted is True
        assert score == pytest.approx(1.0)

    def test_all_fail_rejected(self):
        o = _outcome([[1, 1], [1, 1]])
        accepted, score = decide_score_fusion(o, alpha=0.5, threshold_combined=0.5)
        assert accepted is False
        assert score == pytest.approx(0.0)

    def test_mixed_outcome(self):
        # 4 cycles, 2 subsystems: half pass
        o = _outcome([[0, 1], [0, 1], [0, 1], [0, 1]])
        accepted, score = decide_score_fusion(o, alpha=0.5, threshold_combined=0.5)
        assert accepted is True
        assert score == pytest.approx(0.5)

    def test_threshold_boundary(self):
        o = _outcome([[0, 1], [0, 1]])  # pass rate = 0.5 everywhere
        accepted, _score = decide_score_fusion(o, alpha=0.5, threshold_combined=0.501)
        assert accepted is False

    def test_custom_cycle_partitions(self):
        # 3 cycles; LF = [0], HF = [1, 2]
        o = _outcome([[0, 0], [1, 1], [0, 0]])
        accepted, score = decide_score_fusion(
            o,
            alpha=0.5,
            threshold_combined=0.5,
            lf_cycles=[0],
            hf_cycles=[1, 2],
        )
        # LF: cycle0 rate = 1.0 → mean = 1.0
        # HF: cycle1 rate = 0.0, cycle2 rate = 1.0 → mean = 0.5
        # combined = 0.5*1.0 + 0.5*0.5 = 0.75
        assert accepted is True
        assert score == pytest.approx(0.75)

    def test_returns_score_tuple(self):
        o = _outcome([[0, 0]])
        result = decide_score_fusion(o)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# ConditioningStats
# ---------------------------------------------------------------------------


class TestConditioningStats:
    def test_acceptance_probability(self):
        stats = ConditioningStats(variant="global", total_shots=100, accepted_shots=25)
        assert stats.acceptance_probability == pytest.approx(0.25)

    def test_tts(self):
        stats = ConditioningStats(variant="global", total_shots=100, accepted_shots=25)
        assert stats.tts == pytest.approx(4.0)

    def test_tts_zero_accepted(self):
        stats = ConditioningStats(variant="global", total_shots=100, accepted_shots=0)
        assert stats.tts == float("inf")

    def test_as_dict(self):
        stats = ConditioningStats(variant="hierarchical", total_shots=10, accepted_shots=5)
        d = stats.as_dict()
        assert d["variant"] == "hierarchical"
        assert d["acceptance_probability"] == pytest.approx(0.5)
        assert d["TTS"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# apply_rule_to_batch
# ---------------------------------------------------------------------------


class TestApplyRuleToBatch:
    def _make_batch(self):
        return [
            _outcome([[0, 0, 0, 0]]),  # all pass
            _outcome([[0, 0, 1, 0]]),  # 3/4 pass
            _outcome([[1, 1, 1, 1]]),  # 0 pass
        ]

    def test_global(self):
        batch = self._make_batch()
        stats = apply_rule_to_batch(batch, variant="global")
        assert stats.total_shots == 3
        assert stats.accepted_shots == 1  # only first all-pass

    def test_hierarchical(self):
        batch = self._make_batch()
        stats = apply_rule_to_batch(batch, variant="hierarchical", k_fraction=0.75)
        # ceil(0.75*4)=3; shot0=4≥3✓, shot1=3≥3✓, shot2=0<3✗
        assert stats.accepted_shots == 2

    def test_score_fusion(self):
        batch = self._make_batch()
        stats = apply_rule_to_batch(
            batch,
            variant="score_fusion",
            alpha=0.5,
            threshold_combined=0.5,
        )
        # shot0: rate=1.0 → combined=1.0 ✓
        # shot1: rate=0.75 → combined=0.75 ✓
        # shot2: rate=0.0 → combined=0.0 ✗
        assert stats.accepted_shots == 2
        assert len(stats.scores) == 3

    def test_unknown_variant(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            apply_rule_to_batch([_outcome([[0]])], variant="xyz")
