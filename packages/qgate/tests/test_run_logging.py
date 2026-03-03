"""Tests for qgate.run_logging — RunLogger and FilterResult."""

from __future__ import annotations

import json

import pytest

from qgate.run_logging import FilterResult, RunLogger


@pytest.fixture
def sample_result() -> FilterResult:
    return FilterResult(
        variant="score_fusion",
        total_shots=100,
        accepted_shots=75,
        acceptance_probability=0.75,
        tts=1.333,
        mean_combined_score=0.82,
        threshold_used=0.65,
        scores=[0.8, 0.9, 0.7],
        config_json='{"n_subsystems": 4}',
        metadata={"experiment": "test"},
    )


class TestFilterResult:
    def test_as_dict_excludes_scores(self, sample_result):
        d = sample_result.as_dict()
        assert "scores" not in d
        assert d["variant"] == "score_fusion"
        assert d["total_shots"] == 100

    def test_timestamp_present(self, sample_result):
        assert sample_result.timestamp is not None
        assert len(sample_result.timestamp) > 0


class TestRunLoggerJsonl:
    def test_log_jsonl(self, tmp_path, sample_result):
        log_path = tmp_path / "test.jsonl"
        logger = RunLogger(log_path)
        assert logger.format == "jsonl"
        logger.log(sample_result)

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["variant"] == "score_fusion"

    def test_multiple_logs(self, tmp_path, sample_result):
        log_path = tmp_path / "test.jsonl"
        logger = RunLogger(log_path)
        logger.log(sample_result)
        logger.log(sample_result)

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2


class TestRunLoggerCsv:
    def test_log_csv(self, tmp_path, sample_result):
        log_path = tmp_path / "test.csv"
        logger = RunLogger(log_path)
        assert logger.format == "csv"
        logger.log(sample_result)

        import pandas as pd

        df = pd.read_csv(log_path)
        assert len(df) == 1
        assert df["variant"].iloc[0] == "score_fusion"


class TestRunLoggerExplicitFormat:
    def test_explicit_format(self, tmp_path, sample_result):
        log_path = tmp_path / "output.dat"
        logger = RunLogger(log_path, fmt="jsonl")
        assert logger.format == "jsonl"
        logger.log(sample_result)
        assert log_path.exists()


class TestRunLoggerFlushAll:
    def test_flush_all_jsonl(self, tmp_path, sample_result):
        log_path = tmp_path / "test.jsonl"
        logger = RunLogger(log_path)
        logger.log(sample_result)
        logger.log(sample_result)
        logger.flush_all()

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
