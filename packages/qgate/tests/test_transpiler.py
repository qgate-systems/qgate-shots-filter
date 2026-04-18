"""Tests for qgate.transpiler — DAG conversion, telemetry injection, serialisation."""
from __future__ import annotations

import gzip
import json
import pytest


# ---------------------------------------------------------------------------
# Skip the whole module when qiskit is not installed
# ---------------------------------------------------------------------------
qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.dagcircuit import DAGCircuit

from qgate.transpiler import (
    _PROBE_REG_PREFIX,
    _TELEMETRY_REG_PREFIX,
    circuit_to_dag,
    dag_to_circuit,
    deserialise_payload,
    inject_telemetry,
    serialise_payload,
    strip_telemetry_registers,
)


# ── helpers ────────────────────────────────────────────────────────────────

def _bell_circuit() -> QuantumCircuit:
    """Return a simple 2-qubit Bell-state circuit with measurements."""
    qc = QuantumCircuit(2, 2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


# ── circuit_to_dag / dag_to_circuit ───────────────────────────────────────


class TestCircuitToDag:
    def test_basic_roundtrip(self):
        qc = _bell_circuit()
        dag = circuit_to_dag(qc)
        assert isinstance(dag, DAGCircuit)
        assert dag.num_qubits() == 2

        restored = dag_to_circuit(dag)
        assert isinstance(restored, QuantumCircuit)
        assert restored.num_qubits == 2

    def test_rejects_non_circuit(self):
        with pytest.raises(TypeError, match="Expected a qiskit"):
            circuit_to_dag("not a circuit")


# ── inject_telemetry ──────────────────────────────────────────────────────


class TestInjectTelemetry:
    def test_adds_default_registers(self):
        dag = circuit_to_dag(_bell_circuit())
        original_cregs = len(dag.cregs)

        inject_telemetry(dag)

        # Two new classical registers should have been added
        assert len(dag.cregs) == original_cregs + 2
        names = [creg.name for creg in dag.cregs.values()]
        assert any(n.startswith(_TELEMETRY_REG_PREFIX) for n in names)
        assert any(n.startswith(_PROBE_REG_PREFIX) for n in names)

    def test_custom_widths(self):
        dag = circuit_to_dag(_bell_circuit())
        inject_telemetry(dag, n_routing_bits=5, n_probe_bits=3)

        sizes = {creg.name: creg.size for creg in dag.cregs.values()}
        assert sizes[f"{_TELEMETRY_REG_PREFIX}_route"] == 5
        assert sizes[f"{_PROBE_REG_PREFIX}_aux"] == 3

    def test_does_not_alter_quantum_gates(self):
        dag = circuit_to_dag(_bell_circuit())
        ops_before = dag.count_ops()

        inject_telemetry(dag)

        assert dag.count_ops() == ops_before, "Quantum ops must not change"


# ── strip_telemetry_registers ─────────────────────────────────────────────


class TestStripTelemetry:
    def test_roundtrip(self):
        qc = _bell_circuit()
        dag = circuit_to_dag(qc)
        inject_telemetry(dag)

        modified = dag_to_circuit(dag)
        assert modified.num_clbits > qc.num_clbits  # telemetry present

        cleaned = strip_telemetry_registers(modified)
        assert cleaned.num_clbits == qc.num_clbits  # telemetry gone


# ── serialise / deserialise payload ───────────────────────────────────────


class TestPayloadSerde:
    def test_roundtrip(self):
        dag = circuit_to_dag(_bell_circuit())
        inject_telemetry(dag)

        blob = serialise_payload(dag, shots=1024, backend="ibm_fez")
        assert isinstance(blob, bytes)
        assert len(blob) > 0

        parsed = deserialise_payload(blob)
        assert parsed["version"] == "1.0"
        assert parsed["shots"] == 1024
        assert parsed["backend"] == "ibm_fez"
        assert parsed["n_qubits"] == 2
        assert "qasm" in parsed
        assert "checksum" in parsed

    def test_compressed_smaller_than_raw(self):
        dag = circuit_to_dag(_bell_circuit())
        inject_telemetry(dag)

        blob = serialise_payload(dag, shots=4096)
        raw = gzip.decompress(blob)
        assert len(blob) < len(raw)

    def test_metadata_passthrough(self):
        dag = circuit_to_dag(_bell_circuit())
        inject_telemetry(dag)

        blob = serialise_payload(dag, metadata={"user": "test"})
        parsed = deserialise_payload(blob)
        assert parsed["metadata"] == {"user": "test"}
