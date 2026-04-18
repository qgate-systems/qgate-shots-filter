"""Tests for qgate.execute — the high-level context manager."""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")

from qiskit.circuit import QuantumCircuit
from qiskit.result import Result as QiskitResult

# Import the *module* (not the class) so we can patch its namespace
import importlib
_execute_mod = importlib.import_module("qgate.execute")

from qgate.execute import ExecutionContext, execute


def _ghz3() -> QuantumCircuit:
    """3-qubit GHZ circuit."""
    qc = QuantumCircuit(3, 3, name="ghz3")
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


class TestExecuteContextManager:
    def test_enter_returns_execution_context(self):
        with execute(api_key="k") as ctx:
            assert isinstance(ctx, ExecutionContext)

    @patch.object(_execute_mod, "post_payload")
    def test_full_pipeline(self, mock_post: MagicMock):
        mock_post.return_value = {
            "counts": {"0x0": 500, "0x7": 524},
            "job_id": "test-job-1",
            "metadata": {},
        }

        ctx = ExecutionContext(api_key="test", backend="ibm_fez")
        result = ctx(_ghz3(), shots=1024)

        assert isinstance(result, QiskitResult)
        assert result.success is True
        counts = result.get_counts()
        assert sum(counts.values()) > 0

        # post_payload was called once with bytes (compressed payload)
        mock_post.assert_called_once()
        payload_arg = mock_post.call_args[0][0]
        assert isinstance(payload_arg, bytes)

    @patch.object(_execute_mod, "post_payload")
    def test_custom_metadata(self, mock_post: MagicMock):
        mock_post.return_value = {"counts": {"0x0": 1}, "metadata": {}}

        ctx = ExecutionContext()
        ctx(_ghz3(), shots=1, metadata={"experiment": "test"})

        # The payload should encode the metadata
        mock_post.assert_called_once()

    @patch.object(_execute_mod, "post_payload")
    def test_backend_passed_to_result(self, mock_post: MagicMock):
        mock_post.return_value = {"counts": {"0x0": 1}, "metadata": {}}

        ctx = ExecutionContext(backend="ibm_kyoto")
        result = ctx(_ghz3(), shots=1)

        assert result.backend_name == "ibm_kyoto"
