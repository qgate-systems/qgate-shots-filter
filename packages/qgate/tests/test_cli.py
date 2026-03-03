"""Tests for qgate.cli — typer CLI commands."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from qgate.cli import app

runner = CliRunner()


class TestVersionCommand:
    def test_version(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "qgate" in result.output
        assert "0.5.0" in result.output


class TestValidateCommand:
    def test_validate_valid_config(self, tmp_path):
        config = {"n_subsystems": 4, "n_cycles": 2, "shots": 100}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        result = runner.invoke(app, ["validate", str(config_path)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower() or "✅" in result.output

    def test_validate_invalid_config(self, tmp_path):
        config = {"n_subsystems": -1}
        config_path = tmp_path / "bad.json"
        config_path.write_text(json.dumps(config))

        result = runner.invoke(app, ["validate", str(config_path)])
        assert result.exit_code == 1


class TestRunCommand:
    def test_run_mock(self, tmp_path):
        config = {"n_subsystems": 4, "n_cycles": 2, "shots": 50}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        result = runner.invoke(
            app,
            [
                "run",
                str(config_path),
                "--adapter",
                "mock",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code == 0
        assert "shots=50" in result.output
        assert "P_acc=" in result.output

    def test_run_with_output(self, tmp_path):
        config = {"n_subsystems": 4, "n_cycles": 2, "shots": 50}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))
        out_path = tmp_path / "log.jsonl"

        result = runner.invoke(
            app,
            [
                "run",
                str(config_path),
                "--adapter",
                "mock",
                "--seed",
                "42",
                "--output",
                str(out_path),
            ],
        )
        assert result.exit_code == 0
        assert out_path.exists()
        lines = out_path.read_text().strip().split("\n")
        assert len(lines) >= 1

    def test_run_unknown_adapter(self, tmp_path):
        config = {"n_subsystems": 4, "n_cycles": 2, "shots": 50}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        result = runner.invoke(
            app,
            [
                "run",
                str(config_path),
                "--adapter",
                "nonexistent",
            ],
        )
        assert result.exit_code == 1

    def test_run_shows_run_id(self, tmp_path):
        config = {"n_subsystems": 4, "n_cycles": 2, "shots": 50}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        result = runner.invoke(
            app,
            [
                "run",
                str(config_path),
                "--adapter",
                "mock",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code == 0
        assert "run_id=" in result.output


class TestAdaptersCommand:
    def test_adapters_lists_mock(self):
        result = runner.invoke(app, ["adapters"])
        assert result.exit_code == 0
        assert "mock" in result.output

    def test_adapters_lists_builtins(self):
        result = runner.invoke(app, ["adapters"])
        assert result.exit_code == 0
        assert "qiskit" in result.output
        assert "cirq" in result.output
        assert "pennylane" in result.output


class TestSchemaCommand:
    def test_schema_outputs_json(self):
        result = runner.invoke(app, ["schema"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "properties" in parsed
        assert "n_subsystems" in parsed["properties"]

    def test_schema_includes_schema_version(self):
        result = runner.invoke(app, ["schema"])
        parsed = json.loads(result.output)
        assert "schema_version" in parsed["properties"]
