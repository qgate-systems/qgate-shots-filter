"""Tests for qgate.adapters.registry — entry-point adapter discovery."""
from __future__ import annotations

import pytest

from qgate.adapters.base import BaseAdapter, MockAdapter
from qgate.adapters.registry import list_adapters, load_adapter


class TestListAdapters:
    def test_returns_dict(self):
        result = list_adapters()
        assert isinstance(result, dict)

    def test_mock_always_present(self):
        result = list_adapters()
        assert "mock" in result
        assert "MockAdapter" in result["mock"]

    def test_all_builtins_registered(self):
        result = list_adapters()
        assert "qiskit" in result
        assert "cirq" in result
        assert "pennylane" in result


class TestLoadAdapter:
    def test_load_mock(self):
        cls = load_adapter("mock")
        assert issubclass(cls, BaseAdapter)
        assert cls is MockAdapter

    def test_load_unknown_raises_keyerror(self):
        with pytest.raises(KeyError, match="nonexistent"):
            load_adapter("nonexistent")

    def test_load_unknown_lists_available(self):
        with pytest.raises(KeyError) as exc_info:
            load_adapter("nonexistent")
        msg = str(exc_info.value)
        assert "mock" in msg


class TestTrajectoryFilterStringAdapter:
    """TrajectoryFilter accepts adapter name strings via the registry."""

    def test_filter_with_string_adapter(self):
        from qgate.config import GateConfig
        from qgate.filter import TrajectoryFilter

        config = GateConfig(n_subsystems=4, n_cycles=2, shots=50)
        tf = TrajectoryFilter(config, "mock")
        assert isinstance(tf.adapter, MockAdapter)

    def test_filter_with_class_adapter(self):
        from qgate.config import GateConfig
        from qgate.filter import TrajectoryFilter

        config = GateConfig(n_subsystems=4, n_cycles=2, shots=50)
        tf = TrajectoryFilter(config, MockAdapter)
        assert isinstance(tf.adapter, MockAdapter)

    def test_filter_with_invalid_type_raises(self):
        from qgate.config import GateConfig
        from qgate.filter import TrajectoryFilter

        config = GateConfig(n_subsystems=4, n_cycles=2, shots=50)
        with pytest.raises(TypeError, match="adapter must be"):
            TrajectoryFilter(config, 42)  # type: ignore[arg-type]
