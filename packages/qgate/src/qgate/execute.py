"""
qgate.execute — High-level context manager that ties the full pipeline together.

Usage::

    from qiskit.circuit import QuantumCircuit
    import qgate

    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()

    with qgate.execute(api_key="…", backend="ibm_fez") as run:
        result = run(qc, shots=4096)

    print(result.get_counts())

The context manager orchestrates the six pipeline steps end-to-end:

1. DAG conversion
2. Telemetry injection
3. Payload serialisation
4. Async API hand-off
5. Blind reconstruction
6. Return a clean Qiskit ``Result``
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from qgate.client import (
    AsyncQgateClient,
    ClientConfig,
    QgateBackendError,
    QgateClientError,
    post_payload,
    reconstruct_result,
)
from qgate.transpiler import (
    circuit_to_dag,
    inject_telemetry,
    serialise_payload,
    strip_telemetry_registers,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Qiskit import
# ---------------------------------------------------------------------------
try:
    from qiskit.result import Result as QiskitResult

    _QISKIT_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    _QISKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# ExecutionContext — the callable returned by the context manager
# ---------------------------------------------------------------------------
class ExecutionContext:
    """Callable wrapper returned by :func:`execute`.

    Instances are used *inside* the ``with`` block to submit circuits::

        with qgate.execute(api_key="…") as run:
            result = run(circuit, shots=4096)
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        endpoint: str = "https://api.qgate-compute.com/v1/execute",
        backend: str = "ibm_fez",
        timeout_s: float = 120.0,
        retries: int = 3,
        n_routing_bits: int = 2,
        n_probe_bits: int = 1,
    ) -> None:
        self._backend = backend
        self._n_routing_bits = n_routing_bits
        self._n_probe_bits = n_probe_bits
        self._client_config = ClientConfig(
            api_key=api_key,
            endpoint=endpoint,
            timeout_s=timeout_s,
            retries=retries,
        )

    # ── public call interface ──────────────────────────────────────────────

    def __call__(
        self,
        circuit: "Any",
        *,
        shots: int = 4096,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "QiskitResult":
        """Execute the full pipeline for *circuit*.

        Parameters
        ----------
        circuit:
            A :class:`~qiskit.circuit.QuantumCircuit`.
        shots:
            Number of shots to request.
        metadata:
            Extra key-value pairs forwarded to the backend.

        Returns
        -------
        qiskit.result.Result
            Standard Qiskit result with mitigated counts.
        """
        logger.info("Pipeline start — %d qubits, %d shots", circuit.num_qubits, shots)

        # Step 1 — DAG conversion
        dag = circuit_to_dag(circuit)
        logger.debug("Step 1 (DAG conversion) complete — depth %d", dag.depth())

        # Step 2 — Telemetry injection
        dag = inject_telemetry(
            dag,
            n_routing_bits=self._n_routing_bits,
            n_probe_bits=self._n_probe_bits,
        )
        logger.debug("Step 2 (telemetry injection) complete")

        # Step 3 — Payload serialisation
        payload = serialise_payload(
            dag,
            shots=shots,
            backend=self._backend,
            metadata=metadata,
        )
        logger.debug("Step 3 (payload serialisation) complete — %d bytes", len(payload))

        # Step 4 + 5 — API hand-off (sync wrapper drives the async client)
        raw_response = post_payload(payload, config=self._client_config)
        logger.debug("Step 4-5 (API hand-off) complete")

        # Step 6 — Blind reconstruction
        result = reconstruct_result(
            raw_response,
            shots=shots,
            backend_name=self._backend,
        )
        logger.info("Pipeline complete — result reconstructed")
        return result


# ---------------------------------------------------------------------------
# Context manager factory
# ---------------------------------------------------------------------------
class execute:
    """Context manager that provides an :class:`ExecutionContext`.

    Example::

        with qgate.execute(api_key="…", backend="ibm_fez") as run:
            result = run(my_circuit, shots=4096)

    All keyword arguments are forwarded to :class:`ExecutionContext`.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        self._ctx: Optional[ExecutionContext] = None

    def __enter__(self) -> ExecutionContext:
        self._ctx = ExecutionContext(**self._kwargs)
        return self._ctx

    def __exit__(self, *exc: Any) -> None:
        self._ctx = None
