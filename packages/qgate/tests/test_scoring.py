"""Tests for qgate.scoring — score computation and fusion."""

from __future__ import annotations

import numpy as np
import pytest

from qgate.conditioning import ParityOutcome
from qgate.scoring import compute_window_metric, fuse_scores, score_batch, score_outcome


def _outcome(matrix: list[list[int]]) -> ParityOutcome:
    n_cycles = len(matrix)
    n_sub = len(matrix[0]) if n_cycles else 0
    return ParityOutcome(n_subsystems=n_sub, n_cycles=n_cycles, parity_matrix=matrix)


# ---------------------------------------------------------------------------
# fuse_scores
# ---------------------------------------------------------------------------


class TestFuseScores:
    def test_basic(self):
        accepted, score = fuse_scores(0.8, 0.6, alpha=0.5, threshold=0.65)
        assert score == pytest.approx(0.7)
        assert accepted is True

    def test_below_threshold(self):
        accepted, score = fuse_scores(0.2, 0.3, alpha=0.5, threshold=0.65)
        assert score == pytest.approx(0.25)
        assert accepted is False

    def test_alpha_zero(self):
        _, score = fuse_scores(0.0, 1.0, alpha=0.0)
        assert score == pytest.approx(1.0)

    def test_alpha_one(self):
        _, score = fuse_scores(1.0, 0.0, alpha=1.0)
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# score_outcome
# ---------------------------------------------------------------------------


class TestScoreOutcome:
    def test_all_pass(self):
        o = _outcome([[0, 0], [0, 0], [0, 0], [0, 0]])
        lf, hf, combined = score_outcome(o, alpha=0.5)
        assert lf == pytest.approx(1.0)
        assert hf == pytest.approx(1.0)
        assert combined == pytest.approx(1.0)

    def test_all_fail(self):
        o = _outcome([[1, 1], [1, 1]])
        _lf, _hf, combined = score_outcome(o, alpha=0.5)
        assert combined == pytest.approx(0.0)

    def test_half_pass(self):
        o = _outcome([[0, 1], [0, 1]])
        _lf, _hf, combined = score_outcome(o, alpha=0.5)
        assert combined == pytest.approx(0.5)

    def test_custom_cycles(self):
        o = _outcome([[0, 0], [1, 1], [0, 0]])
        lf, hf, combined = score_outcome(o, alpha=0.5, lf_cycles=[0], hf_cycles=[1, 2])
        # LF: cycle0 rate=1.0; HF: cycle1 rate=0.0, cycle2 rate=1.0 → mean=0.5
        assert lf == pytest.approx(1.0)
        assert hf == pytest.approx(0.5)
        assert combined == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# score_batch
# ---------------------------------------------------------------------------


class TestScoreBatch:
    def test_batch_length(self):
        batch = [_outcome([[0, 0]]), _outcome([[1, 1]]), _outcome([[0, 1]])]
        results = score_batch(batch, alpha=0.5)
        assert len(results) == 3
        for r in results:
            assert len(r) == 3  # (lf, hf, combined)


# ---------------------------------------------------------------------------
# compute_window_metric
# ---------------------------------------------------------------------------


class TestComputeWindowMetric:
    def test_max_mode(self):
        t = np.linspace(0, 10, 200)
        v = np.sin(t)
        metric, _ws, we = compute_window_metric(t, v, window=2.0, mode="max")
        assert we == pytest.approx(10.0)
        expected = float(np.max(v[(t >= 8.0) & (t <= 10.0)]))
        assert metric == pytest.approx(expected)

    def test_mean_mode(self):
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        v = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        metric, _, _ = compute_window_metric(t, v, window=2.0, mode="mean")
        assert metric == pytest.approx(0.8)

    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            compute_window_metric(np.array([0.0, 1.0]), np.array([0.5, 0.5]), mode="median")
