"""Tests for qgate.run_logging.compute_run_id — deterministic run IDs."""

from __future__ import annotations

import json

from qgate.run_logging import compute_run_id


class TestComputeRunId:
    def test_returns_12_char_hex(self):
        rid = compute_run_id('{"shots": 100}')
        assert len(rid) == 12
        assert all(c in "0123456789abcdef" for c in rid)

    def test_deterministic(self):
        cfg = '{"n_subsystems": 4, "shots": 1024}'
        r1 = compute_run_id(cfg, adapter_name="MockAdapter")
        r2 = compute_run_id(cfg, adapter_name="MockAdapter")
        assert r1 == r2

    def test_different_config_different_id(self):
        r1 = compute_run_id('{"shots": 100}', adapter_name="MockAdapter")
        r2 = compute_run_id('{"shots": 200}', adapter_name="MockAdapter")
        assert r1 != r2

    def test_different_adapter_different_id(self):
        cfg = '{"shots": 100}'
        r1 = compute_run_id(cfg, adapter_name="MockAdapter")
        r2 = compute_run_id(cfg, adapter_name="QiskitAdapter")
        assert r1 != r2

    def test_circuit_hash_changes_id(self):
        cfg = '{"shots": 100}'
        r1 = compute_run_id(cfg, adapter_name="Mock")
        r2 = compute_run_id(cfg, adapter_name="Mock", circuit_hash="abc123")
        assert r1 != r2

    def test_json_key_order_irrelevant(self):
        """Config key order should not affect run_id (canonical sort)."""
        cfg_a = '{"shots": 100, "n_subsystems": 4}'
        cfg_b = '{"n_subsystems": 4, "shots": 100}'
        assert compute_run_id(cfg_a) == compute_run_id(cfg_b)


class TestFilterResultRunId:
    """FilterResult carries a run_id populated by TrajectoryFilter."""

    def test_run_id_populated(self):
        from qgate.adapters.base import MockAdapter
        from qgate.config import GateConfig
        from qgate.filter import TrajectoryFilter

        config = GateConfig(n_subsystems=4, n_cycles=2, shots=50)
        tf = TrajectoryFilter(config, MockAdapter(seed=42))
        result = tf.run()
        assert result.run_id
        assert len(result.run_id) == 12

    def test_run_id_reproducible(self):
        from qgate.adapters.base import MockAdapter
        from qgate.config import GateConfig
        from qgate.filter import TrajectoryFilter

        config = GateConfig(n_subsystems=4, n_cycles=2, shots=50)
        r1 = TrajectoryFilter(config, MockAdapter(seed=42)).run()
        r2 = TrajectoryFilter(config, MockAdapter(seed=42)).run()
        assert r1.run_id == r2.run_id

    def test_run_id_in_as_dict(self):
        from qgate.adapters.base import MockAdapter
        from qgate.config import GateConfig
        from qgate.filter import TrajectoryFilter

        config = GateConfig(n_subsystems=4, n_cycles=2, shots=50)
        result = TrajectoryFilter(config, MockAdapter(seed=42)).run()
        d = result.as_dict()
        assert "run_id" in d
        assert d["run_id"] == result.run_id

    def test_run_id_in_logged_output(self, tmp_path):
        from qgate.adapters.base import MockAdapter
        from qgate.config import GateConfig
        from qgate.filter import TrajectoryFilter
        from qgate.run_logging import RunLogger

        log_path = tmp_path / "log.jsonl"
        config = GateConfig(n_subsystems=4, n_cycles=2, shots=50)
        logger = RunLogger(log_path)
        tf = TrajectoryFilter(config, MockAdapter(seed=42), logger=logger)
        result = tf.run()

        lines = log_path.read_text().strip().split("\n")
        rec = json.loads(lines[0])
        assert rec["run_id"] == result.run_id
