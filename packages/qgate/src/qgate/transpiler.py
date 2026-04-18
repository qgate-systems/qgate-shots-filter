"""
qgate.transpiler — DAG interception, telemetry injection & payload serialisation.

This module is a **dumb** wrapper: it contains **zero** proprietary algorithms.
It converts a user's :class:`~qiskit.circuit.QuantumCircuit` into a DAG, injects
benign telemetry routing registers, and serialises everything into a compressed
JSON payload ready for the remote Qgate API backend.

Pipeline steps handled here
----------------------------
1. DAG conversion  (``circuit_to_dag``)
2. Telemetry injection  (``inject_telemetry``)
3. Payload serialisation  (``serialise_payload``)
"""

from __future__ import annotations

import gzip
import json
import base64
import hashlib
import logging
import uuid
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Qiskit imports — only required when the qiskit extra is installed
# ---------------------------------------------------------------------------
try:
    from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.converters import circuit_to_dag as _qiskit_circuit_to_dag
    from qiskit.converters import dag_to_circuit as _qiskit_dag_to_circuit
    from qiskit.dagcircuit import DAGCircuit
    from qiskit.qasm2 import dumps as qasm2_dumps

    _QISKIT_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    _QISKIT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TELEMETRY_REG_PREFIX = "_qg_tel"
"""Prefix used for injected classical registers so they can be stripped later."""

_PROBE_REG_PREFIX = "_qg_prb"
"""Prefix used for auxiliary probe measurement registers."""

_PAYLOAD_VERSION = "1.0"
"""Wire-format version embedded in every serialised payload."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_qiskit() -> None:
    """Raise a clear error when Qiskit is not installed."""
    if not _QISKIT_AVAILABLE:
        raise ImportError(
            "Qiskit is required for circuit transpilation.  "
            "Install it with:  pip install qgate[qiskit]"
        )


# ---------------------------------------------------------------------------
# 1.  DAG conversion
# ---------------------------------------------------------------------------
def circuit_to_dag(circuit: "QuantumCircuit") -> "DAGCircuit":
    """Convert a :class:`~qiskit.circuit.QuantumCircuit` to a :class:`DAGCircuit`.

    Parameters
    ----------
    circuit:
        The user's quantum circuit.

    Returns
    -------
    DAGCircuit
        The directed-acyclic-graph representation of *circuit*.

    Raises
    ------
    ImportError
        If ``qiskit`` is not installed.
    TypeError
        If *circuit* is not a :class:`QuantumCircuit`.
    """
    _require_qiskit()
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(
            f"Expected a qiskit.circuit.QuantumCircuit, got {type(circuit).__name__}"
        )
    return _qiskit_circuit_to_dag(circuit)


def dag_to_circuit(dag: "DAGCircuit") -> "QuantumCircuit":
    """Convert a :class:`DAGCircuit` back to a :class:`QuantumCircuit`.

    This is the inverse of :func:`circuit_to_dag`.
    """
    _require_qiskit()
    return _qiskit_dag_to_circuit(dag)


# ---------------------------------------------------------------------------
# 2.  Telemetry injection
# ---------------------------------------------------------------------------
def inject_telemetry(
    dag: "DAGCircuit",
    *,
    n_routing_bits: int = 2,
    n_probe_bits: int = 1,
) -> "DAGCircuit":
    """Inject dummy classical routing registers and probe measurement bits.

    The injected registers do **not** alter the user's algorithmic path.
    They are used by the remote backend for internal bookkeeping (routing
    decisions, ancilla-readout tagging, etc.).

    Parameters
    ----------
    dag:
        A :class:`DAGCircuit` obtained from :func:`circuit_to_dag`.
    n_routing_bits:
        Number of classical routing bits to add (default ``2``).
    n_probe_bits:
        Number of auxiliary measurement probe bits to add (default ``1``).

    Returns
    -------
    DAGCircuit
        The *same* DAG object, mutated in-place with additional classical
        registers.  The original quantum gates are untouched.
    """
    _require_qiskit()

    # --- routing register ---------------------------------------------------
    routing_reg = ClassicalRegister(n_routing_bits, f"{_TELEMETRY_REG_PREFIX}_route")
    dag.add_creg(routing_reg)

    # --- probe register ------------------------------------------------------
    probe_reg = ClassicalRegister(n_probe_bits, f"{_PROBE_REG_PREFIX}_aux")
    dag.add_creg(probe_reg)

    logger.debug(
        "Injected telemetry: %d routing bits + %d probe bits",
        n_routing_bits,
        n_probe_bits,
    )
    return dag


def strip_telemetry_registers(circuit: "QuantumCircuit") -> "QuantumCircuit":
    """Return a copy of *circuit* with all qgate-injected registers removed.

    Useful on the reconstruction side to give the user a clean result that
    matches their original register layout.
    """
    _require_qiskit()
    keep_cregs = [
        creg
        for creg in circuit.cregs
        if not creg.name.startswith((_TELEMETRY_REG_PREFIX, _PROBE_REG_PREFIX))
    ]
    new_qc = QuantumCircuit(*circuit.qregs, *keep_cregs, name=circuit.name)
    # Copy every instruction that targets only kept registers
    kept_bits = set()
    for creg in keep_cregs:
        kept_bits.update(creg)
    kept_bits.update(circuit.qubits)

    for instruction in circuit.data:
        qargs = instruction.qubits
        cargs = instruction.clbits
        if all(b in kept_bits for b in list(qargs) + list(cargs)):
            new_qc.append(instruction.operation, qargs, cargs)
    return new_qc


# ---------------------------------------------------------------------------
# 3.  Payload serialisation
# ---------------------------------------------------------------------------
def serialise_payload(
    dag: "DAGCircuit",
    *,
    shots: int = 4096,
    backend: str = "ibm_fez",
    metadata: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Serialise the DAG into a gzip-compressed JSON payload.

    The resulting ``bytes`` object is ready to POST to the Qgate backend.

    Payload schema (after decompression)::

        {
            "version":     "1.0",
            "payload_id":  "<uuid4>",
            "qasm":        "<OpenQASM 2.0 string>",
            "shots":       4096,
            "backend":     "ibm_fez",
            "n_qubits":    <int>,
            "dag_depth":   <int>,
            "checksum":    "<sha256 of qasm>",
            "metadata":    { ... }
        }

    Parameters
    ----------
    dag:
        The (telemetry-injected) DAG.
    shots:
        Requested shot count.
    backend:
        Target backend identifier.
    metadata:
        Arbitrary extra key-value pairs to include.

    Returns
    -------
    bytes
        A gzip-compressed JSON blob.
    """
    _require_qiskit()

    circuit = dag_to_circuit(dag)
    qasm_str = qasm2_dumps(circuit)

    payload: Dict[str, Any] = {
        "version": _PAYLOAD_VERSION,
        "payload_id": str(uuid.uuid4()),
        "qasm": qasm_str,
        "shots": shots,
        "backend": backend,
        "n_qubits": circuit.num_qubits,
        "dag_depth": dag.depth(),
        "checksum": hashlib.sha256(qasm_str.encode()).hexdigest(),
        "metadata": metadata or {},
    }

    raw = json.dumps(payload, separators=(",", ":")).encode()
    compressed = gzip.compress(raw, compresslevel=9)
    logger.debug(
        "Payload serialised: %d bytes raw → %d bytes compressed (%.1f%%)",
        len(raw),
        len(compressed),
        100 * len(compressed) / len(raw) if raw else 0,
    )
    return compressed


def deserialise_payload(data: bytes) -> Dict[str, Any]:
    """Decompress and parse a payload previously created by :func:`serialise_payload`.

    Useful for testing / debugging.
    """
    raw = gzip.decompress(data)
    return json.loads(raw)
