"""
qgate.client — Async HTTP client for the Qgate Compute API.

This module handles **step 5** (API hand-off) and **step 6** (blind
reconstruction) of the execution pipeline.  It POSTs a compressed payload
to the remote Qgate backend and reconstructs a standard Qiskit
:class:`~qiskit.result.Result` from the response — without exposing any
backend-side mitigation details to the caller.

The module ships two client flavours:

* :class:`AsyncQgateClient` — ``async / await`` based (``aiohttp``).
* :func:`post_payload`       — Convenience synchronous wrapper.
"""

from __future__ import annotations

import asyncio
import json
import gzip
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------
try:
    import aiohttp

    _AIOHTTP_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    _AIOHTTP_AVAILABLE = False

try:
    from qiskit.result import Result as QiskitResult
    from qiskit.result.models import ExperimentResult, ExperimentResultData

    _QISKIT_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    _QISKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class QgateClientError(RuntimeError):
    """Raised for any transport or HTTP-level error."""

    def __init__(self, message: str, *, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


class QgateBackendError(RuntimeError):
    """Raised when the remote backend returns an application-level error."""

    def __init__(self, message: str, *, error_code: str = "") -> None:
        super().__init__(message)
        self.error_code = error_code


# ---------------------------------------------------------------------------
# Default API endpoint
# ---------------------------------------------------------------------------
_DEFAULT_ENDPOINT = "https://api.qgate-compute.com/v1/execute"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClientConfig:
    """Configuration for :class:`AsyncQgateClient`.

    Parameters
    ----------
    api_key:
        API key sent in the ``Authorization`` header.
    endpoint:
        Full URL of the ``/v1/execute`` endpoint.
    timeout_s:
        Total request timeout in seconds.
    retries:
        Number of retries on transient failures (5xx / timeout).
    retry_backoff_s:
        Base back-off between retries (doubled each attempt).
    """

    api_key: str = ""
    endpoint: str = _DEFAULT_ENDPOINT
    timeout_s: float = 120.0
    retries: int = 3
    retry_backoff_s: float = 1.0


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------
class AsyncQgateClient:
    """Asynchronous HTTP client that POSTs payloads to the Qgate backend.

    Usage::

        async with AsyncQgateClient(config) as client:
            raw_result = await client.submit(compressed_payload)
    """

    def __init__(self, config: Optional[ClientConfig] = None) -> None:
        self._config = config or ClientConfig()
        self._session: Optional["aiohttp.ClientSession"] = None

    # -- context manager -----------------------------------------------------

    async def __aenter__(self) -> "AsyncQgateClient":
        if not _AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp is required for the async Qgate client.  "
                "Install it with:  pip install qgate[qiskit]"
            )
        timeout = aiohttp.ClientTimeout(total=self._config.timeout_s)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    # -- core ----------------------------------------------------------------

    async def submit(self, payload: bytes) -> Dict[str, Any]:
        """POST *payload* to the backend and return the parsed JSON response.

        Parameters
        ----------
        payload:
            Gzip-compressed JSON bytes produced by
            :func:`qgate.transpiler.serialise_payload`.

        Returns
        -------
        dict
            Raw JSON body returned by the backend.

        Raises
        ------
        QgateClientError
            On HTTP / transport errors after exhausting retries.
        QgateBackendError
            If the backend returns a structured error response.
        """
        if self._session is None:
            raise RuntimeError("Client is not open.  Use `async with` context manager.")

        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Encoding": "gzip",
            "Accept": "application/json",
        }
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        last_exc: Optional[Exception] = None
        backoff = self._config.retry_backoff_s

        for attempt in range(1, self._config.retries + 1):
            try:
                logger.debug(
                    "POST %s (attempt %d/%d, %d bytes)",
                    self._config.endpoint,
                    attempt,
                    self._config.retries,
                    len(payload),
                )
                async with self._session.post(
                    self._config.endpoint, data=payload, headers=headers
                ) as resp:
                    body = await resp.read()

                    if resp.status == 200:
                        return self._parse_success(body)

                    # Structured backend error
                    if 400 <= resp.status < 500:
                        raise QgateBackendError(
                            f"Backend returned {resp.status}: {body[:500]}",
                            error_code=str(resp.status),
                        )

                    # Transient server error — retry
                    last_exc = QgateClientError(
                        f"Server error {resp.status}", status_code=resp.status
                    )
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = QgateClientError(str(exc))

            if attempt < self._config.retries:
                logger.warning("Retrying in %.1fs …", backoff)
                await asyncio.sleep(backoff)
                backoff *= 2

        raise QgateClientError(
            f"All {self._config.retries} attempts failed: {last_exc}"
        )

    # -- private helpers -----------------------------------------------------

    @staticmethod
    def _parse_success(body: bytes) -> Dict[str, Any]:
        """Parse a 200-OK response body into a dict."""
        try:
            return json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise QgateClientError(f"Malformed JSON response: {exc}") from exc


# ---------------------------------------------------------------------------
# Blind reconstruction — build Qiskit Result from backend response
# ---------------------------------------------------------------------------
def reconstruct_result(
    raw: Dict[str, Any],
    *,
    shots: int = 4096,
    backend_name: str = "qgate_remote",
) -> "QiskitResult":
    """Reconstruct a standard :class:`~qiskit.result.Result` from a backend response.

    The backend response is expected to contain (at minimum)::

        {
            "counts": {"0x0": 512, "0x1": 488, ...},
            "metadata": { ... }          # optional
        }

    Any backend-internal telemetry keys (prefixed ``_qg_``) are silently
    stripped so the user never sees mitigation artefacts.

    Parameters
    ----------
    raw:
        Parsed JSON dict from the backend.
    shots:
        The shot count to embed in the result header.
    backend_name:
        Name attributed to the result's backend field.

    Returns
    -------
    qiskit.result.Result

    Raises
    ------
    ImportError
        If Qiskit is not installed.
    KeyError
        If ``raw`` lacks required ``counts`` key.
    """
    if not _QISKIT_AVAILABLE:
        raise ImportError(
            "Qiskit is required for result reconstruction.  "
            "Install it with:  pip install qgate[qiskit]"
        )

    # Strip backend telemetry metadata
    counts = raw["counts"]
    user_meta = {
        k: v for k, v in raw.get("metadata", {}).items() if not k.startswith("_qg_")
    }

    exp_data = ExperimentResultData(counts=counts)
    exp_result = ExperimentResult(
        shots=shots,
        success=True,
        data=exp_data,
        header={"name": "qgate_executed", "metadata": user_meta},
    )

    result = QiskitResult(
        backend_name=backend_name,
        backend_version="0.0.1",
        qobj_id="qgate",
        job_id=raw.get("job_id", "qgate-local"),
        success=True,
        results=[exp_result],
    )
    return result


# ---------------------------------------------------------------------------
# Synchronous convenience wrapper
# ---------------------------------------------------------------------------
def post_payload(
    payload: bytes,
    *,
    config: Optional[ClientConfig] = None,
) -> Dict[str, Any]:
    """Synchronous wrapper around :meth:`AsyncQgateClient.submit`.

    Spins up an event loop, POSTs the payload, and returns the raw dict.
    Suitable for scripts / notebooks where ``await`` is inconvenient.
    """
    cfg = config or ClientConfig()

    async def _run() -> Dict[str, Any]:
        async with AsyncQgateClient(cfg) as client:
            return await client.submit(payload)

    return asyncio.run(_run())
