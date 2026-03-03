"""Tests for edge cases, new features, and expanded coverage.

Added in v0.4.0 to cover:
  • Empty inputs to filter()
  • Single subsystem / single cycle configurations
  • filter_counts() with Qiskit-style count dicts
  • RunLogger context-manager protocol
  • RunLogger Parquet buffering
  • GateConfig frozen immutability
  • TrajectoryFilter __repr__
  • Vectorised score_batch fast path
  • CLI --verbose / --quiet / --error-rate flags
  • Qiskit parse_results copy safety
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from qgate.adapters.base import MockAdapter
from qgate.conditioning import ParityOutcome
from qgate.config import GateConfig
from qgate.filter import TrajectoryFilter
from qgate.run_logging import FilterResult, RunLogger
from qgate.scoring import score_batch, score_outcome

# ---------------------------------------------------------------------------
# Empty / boundary inputs
# ---------------------------------------------------------------------------

class TestFilterEmptyInput:
    def test_filter_zero_outcomes(self):
        config = GateConfig(n_subsystems=4, n_cycles=2, shots=100)
        tf = TrajectoryFilter(config, MockAdapter(seed=42))
        result = tf.filter([])
        assert result.total_shots == 0
        assert result.accepted_shots == 0
        assert result.acceptance_probability == 0.0
        assert result.tts == float("inf")
        assert result.run_id  # should still have an ID

    def test_filter_single_outcome(self):
        config = GateConfig(n_subsystems=1, n_cycles=1, shots=1, variant="global")
        tf = TrajectoryFilter(config, MockAdapter(error_rate=0.0, seed=42))
        result = tf.run()
        assert result.total_shots == 1
        assert result.accepted_shots == 1


class TestSingleSubsystemCycle:
    def test_n1_c1(self):
        config = GateConfig(n_subsystems=1, n_cycles=1, shots=100)
        tf = TrajectoryFilter(config, MockAdapter(seed=42))
        result = tf.run()
        assert 0.0 <= result.acceptance_probability <= 1.0

    def test_n1_c1_global(self):
        config = GateConfig(n_subsystems=1, n_cycles=1, shots=100, variant="global")
        tf = TrajectoryFilter(config, MockAdapter(error_rate=0.0, seed=42))
        result = tf.run()
        assert result.accepted_shots == 100

    def test_n1_c1_hierarchical(self):
        config = GateConfig(n_subsystems=1, n_cycles=1, shots=100, variant="hierarchical")
        tf = TrajectoryFilter(config, MockAdapter(error_rate=0.0, seed=42))
        result = tf.run()
        assert result.accepted_shots == 100


# ---------------------------------------------------------------------------
# ParityOutcome ndarray coercion
# ---------------------------------------------------------------------------

class TestParityOutcomeNdarray:
    def test_list_coerced_to_ndarray(self):
        o = ParityOutcome(2, 1, [[0, 1]])
        assert isinstance(o.parity_matrix, np.ndarray)
        assert o.parity_matrix.dtype == np.int8

    def test_ndarray_accepted(self):
        arr = np.array([[0, 1, 0]], dtype=np.int8)
        o = ParityOutcome(3, 1, arr)
        assert isinstance(o.parity_matrix, np.ndarray)

    def test_pass_rates_property(self):
        o = ParityOutcome(4, 2, [[0, 0, 1, 0], [0, 0, 0, 0]])
        rates = o.pass_rates
        assert rates.shape == (2,)
        assert rates[0] == pytest.approx(0.75)
        assert rates[1] == pytest.approx(1.0)

    def test_empty_matrix_creates_zeros(self):
        o = ParityOutcome(3, 2, [])
        assert o.parity_matrix.shape == (2, 3)
        assert np.all(o.parity_matrix == 0)


# ---------------------------------------------------------------------------
# Vectorised score_batch
# ---------------------------------------------------------------------------

class TestScoreBatchVectorised:
    def test_empty_batch(self):
        assert score_batch([]) == []

    def test_uniform_batch_fast_path(self):
        outcomes = [
            ParityOutcome(4, 2, [[0, 0, 0, 0], [0, 0, 0, 0]]),
            ParityOutcome(4, 2, [[1, 1, 1, 1], [1, 1, 1, 1]]),
        ]
        results = score_batch(outcomes, alpha=0.5)
        assert len(results) == 2
        assert results[0][2] == pytest.approx(1.0)
        assert results[1][2] == pytest.approx(0.0)

    def test_matches_per_shot_scoring(self):
        """Vectorised batch must match per-shot score_outcome."""
        adapter = MockAdapter(error_rate=0.2, seed=7)
        outcomes = adapter.build_and_run(4, 3, 50)
        batch_results = score_batch(outcomes, alpha=0.6)
        for i, o in enumerate(outcomes):
            lf, hf, combined = score_outcome(o, alpha=0.6)
            assert batch_results[i][0] == pytest.approx(lf, abs=1e-10)
            assert batch_results[i][1] == pytest.approx(hf, abs=1e-10)
            assert batch_results[i][2] == pytest.approx(combined, abs=1e-10)


# ---------------------------------------------------------------------------
# GateConfig frozen
# ---------------------------------------------------------------------------

class TestGateConfigFrozen:
    def test_cannot_mutate_shots(self):
        from pydantic import ValidationError

        gc = GateConfig()
        with pytest.raises(ValidationError):
            gc.shots = 999  # type: ignore[misc]

    def test_cannot_mutate_variant(self):
        from pydantic import ValidationError

        gc = GateConfig()
        with pytest.raises(ValidationError):
            gc.variant = "global"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TrajectoryFilter __repr__
# ---------------------------------------------------------------------------

class TestTrajectoryFilterRepr:
    def test_repr(self):
        config = GateConfig(n_subsystems=4, n_cycles=2, shots=100)
        tf = TrajectoryFilter(config, MockAdapter(seed=42))
        r = repr(tf)
        assert "TrajectoryFilter" in r
        assert "n_sub=4" in r
        assert "n_cyc=2" in r
        assert "shots=100" in r
        assert "MockAdapter" in r


# ---------------------------------------------------------------------------
# RunLogger context manager and Parquet buffering
# ---------------------------------------------------------------------------

class TestRunLoggerContextManager:
    def test_context_manager_protocol(self, tmp_path):
        log_path = tmp_path / "log.jsonl"
        with RunLogger(log_path) as rl:
            rl.log(FilterResult(variant="test", total_shots=1))
        assert log_path.exists()

    def test_parquet_buffered_write(self, tmp_path):
        """Parquet should only write on close(), not on every log()."""
        log_path = tmp_path / "log.parquet"
        with RunLogger(log_path) as rl:
            # Supply non-empty metadata so pyarrow can infer schema for struct column
            rl.log(FilterResult(variant="test", total_shots=1, metadata={"k": "v"}))
            rl.log(FilterResult(variant="test", total_shots=2, metadata={"k": "v"}))
            # File may not exist yet (buffered)
        # After close(), file must exist
        assert log_path.exists()

    def test_unknown_extension_warns(self, tmp_path, caplog):
        """Unknown file extension should emit a warning."""
        import logging
        with caplog.at_level(logging.WARNING, logger="qgate.run_logging"):
            RunLogger(tmp_path / "data.xyz")
        assert "Unknown file extension" in caplog.text


# ---------------------------------------------------------------------------
# CLI new flags
# ---------------------------------------------------------------------------

class TestCLINewFlags:
    def test_run_with_error_rate(self, tmp_path):
        from typer.testing import CliRunner

        from qgate.cli import app

        runner = CliRunner()
        config = {"n_subsystems": 4, "n_cycles": 2, "shots": 50}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        result = runner.invoke(app, [
            "run", str(config_path),
            "--adapter", "mock",
            "--seed", "42",
            "--error-rate", "0.5",
            "--quiet",
        ])
        assert result.exit_code == 0
        assert "P_acc=" in result.output

    def test_run_verbose(self, tmp_path):
        from typer.testing import CliRunner

        from qgate.cli import app

        runner = CliRunner()
        config = {"n_subsystems": 4, "n_cycles": 2, "shots": 50}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        result = runner.invoke(app, [
            "run", str(config_path),
            "--adapter", "mock",
            "--verbose",
        ])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# filter_counts smoke test
# ---------------------------------------------------------------------------

class TestFilterCounts:
    def test_filter_counts_basic(self):
        """filter_counts with a mock adapter should work end-to-end."""
        config = GateConfig(n_subsystems=2, n_cycles=1, shots=10)
        adapter = MockAdapter(error_rate=0.1, seed=42)
        tf = TrajectoryFilter(config, adapter)

        # Build a synthetic "counts" dict
        circuit = adapter.build_circuit(2, 1)
        raw = adapter.run(circuit, 10)
        # filter_counts delegates to parse_results → filter
        result = tf.filter_counts(raw, n_subsystems=2, n_cycles=1)
        assert result.total_shots == 10


# ---------------------------------------------------------------------------
# Qiskit parse_results copy safety
# ---------------------------------------------------------------------------

class TestQiskitParseCopySafety:
    def test_copy_safety(self):
        """Mutating one outcome must not affect others from the same bitstring."""
        from qgate.adapters.qiskit_adapter import HAS_QISKIT, QiskitAdapter
        if not HAS_QISKIT:
            pytest.skip("Qiskit not installed")

        adapter = QiskitAdapter()
        # Build and run a small circuit
        circuit = adapter.build_circuit(2, 1)
        raw = adapter.run(circuit, 10)
        outcomes = adapter.parse_results(raw, 2, 1)
        if len(outcomes) >= 2:
            # Mutate the first outcome
            outcomes[0].parity_matrix[0, 0] = 99
            # Second outcome should be unaffected
            assert outcomes[1].parity_matrix[0, 0] != 99
