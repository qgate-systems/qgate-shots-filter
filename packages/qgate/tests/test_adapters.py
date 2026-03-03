"""Tests for qgate.adapters — BaseAdapter, MockAdapter."""

from __future__ import annotations

import pytest

from qgate.adapters.base import BaseAdapter, MockAdapter
from qgate.conditioning import ParityOutcome


class TestMockAdapter:
    def test_build_circuit(self):
        adapter = MockAdapter(error_rate=0.05, seed=42)
        circuit = adapter.build_circuit(n_subsystems=4, n_cycles=2)
        assert circuit == {"n_subsystems": 4, "n_cycles": 2}

    def test_run_returns_correct_shape(self):
        adapter = MockAdapter(error_rate=0.05, seed=42)
        circuit = adapter.build_circuit(4, 2)
        results = adapter.run(circuit, shots=100)
        assert len(results) == 100
        for matrix in results:
            assert len(matrix) == 2  # n_cycles
            for row in matrix:
                assert len(row) == 4  # n_subsystems
                for val in row:
                    assert val in (0, 1)

    def test_parse_results(self):
        adapter = MockAdapter(error_rate=0.05, seed=42)
        circuit = adapter.build_circuit(3, 2)
        raw = adapter.run(circuit, shots=10)
        outcomes = adapter.parse_results(raw, 3, 2)
        assert len(outcomes) == 10
        for o in outcomes:
            assert isinstance(o, ParityOutcome)
            assert o.n_subsystems == 3
            assert o.n_cycles == 2

    def test_build_and_run(self):
        adapter = MockAdapter(error_rate=0.10, seed=123)
        outcomes = adapter.build_and_run(n_subsystems=4, n_cycles=2, shots=50)
        assert len(outcomes) == 50
        assert all(isinstance(o, ParityOutcome) for o in outcomes)

    def test_reproducibility(self):
        a1 = MockAdapter(error_rate=0.1, seed=99)
        a2 = MockAdapter(error_rate=0.1, seed=99)
        o1 = a1.build_and_run(4, 2, 20)
        o2 = a2.build_and_run(4, 2, 20)
        for p1, p2 in zip(o1, o2):
            assert (p1.parity_matrix == p2.parity_matrix).all()

    def test_zero_error_rate(self):
        adapter = MockAdapter(error_rate=0.0, seed=42)
        outcomes = adapter.build_and_run(4, 2, 100)
        for o in outcomes:
            for cycle in o.parity_matrix:
                assert all(v == 0 for v in cycle)

    def test_full_error_rate(self):
        adapter = MockAdapter(error_rate=1.0, seed=42)
        outcomes = adapter.build_and_run(4, 2, 100)
        for o in outcomes:
            for cycle in o.parity_matrix:
                assert all(v == 1 for v in cycle)


class TestBaseAdapterAbstract:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseAdapter()  # type: ignore[abstract]
