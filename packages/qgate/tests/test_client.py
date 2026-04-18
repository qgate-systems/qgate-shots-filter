"""Tests for qgate.client — async client, reconstruction, sync wrapper."""
from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")

from qiskit.result import Result as QiskitResult

from qgate.client import (
    AsyncQgateClient,
    ClientConfig,
    QgateBackendError,
    QgateClientError,
    reconstruct_result,
)


# ── reconstruct_result ────────────────────────────────────────────────────


class TestReconstructResult:
    def test_basic(self):
        raw = {"counts": {"0x0": 500, "0x3": 524}, "metadata": {"foo": "bar"}}
        result = reconstruct_result(raw, shots=1024, backend_name="ibm_fez")

        assert isinstance(result, QiskitResult)
        assert result.success is True
        assert result.backend_name == "ibm_fez"

        counts = result.get_counts()
        # Qiskit may normalise hex keys — just check values are present
        assert sum(counts.values()) == 1024

    def test_strips_qg_metadata(self):
        raw = {
            "counts": {"0x0": 1},
            "metadata": {"_qg_internal": "secret", "visible": "yes"},
        }
        result = reconstruct_result(raw, shots=1)
        header_meta = result.results[0].header["metadata"]
        assert "_qg_internal" not in header_meta
        assert header_meta["visible"] == "yes"

    def test_missing_counts_raises(self):
        with pytest.raises(KeyError):
            reconstruct_result({"metadata": {}})


# ── AsyncQgateClient unit tests (mocked transport) ───────────────────────


class TestAsyncQgateClient:
    @pytest.mark.asyncio
    async def test_submit_success(self):
        response_body = json.dumps({"counts": {"0x0": 42}}).encode()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.read = AsyncMock(return_value=response_body)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)

        config = ClientConfig(api_key="test-key", retries=1)
        client = AsyncQgateClient(config)
        client._session = mock_session

        result = await client.submit(b"compressed-payload")
        assert result == {"counts": {"0x0": 42}}

    @pytest.mark.asyncio
    async def test_submit_4xx_raises_backend_error(self):
        import aiohttp  # noqa: F811

        mock_resp = AsyncMock()
        mock_resp.status = 422
        mock_resp.read = AsyncMock(return_value=b"bad request")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)

        client = AsyncQgateClient(ClientConfig(retries=1))
        client._session = mock_session

        with pytest.raises(QgateBackendError, match="422"):
            await client.submit(b"payload")

    @pytest.mark.asyncio
    async def test_submit_retries_on_5xx(self):
        # First call: 503; second call: 200
        err_resp = AsyncMock()
        err_resp.status = 503
        err_resp.read = AsyncMock(return_value=b"unavailable")
        err_resp.__aenter__ = AsyncMock(return_value=err_resp)
        err_resp.__aexit__ = AsyncMock(return_value=False)

        ok_resp = AsyncMock()
        ok_resp.status = 200
        ok_resp.read = AsyncMock(return_value=json.dumps({"counts": {}}).encode())
        ok_resp.__aenter__ = AsyncMock(return_value=ok_resp)
        ok_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=[err_resp, ok_resp])

        client = AsyncQgateClient(ClientConfig(retries=2, retry_backoff_s=0))
        client._session = mock_session

        result = await client.submit(b"payload")
        assert result == {"counts": {}}
        assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_when_not_open(self):
        client = AsyncQgateClient()
        with pytest.raises(RuntimeError, match="not open"):
            await client.submit(b"payload")


# ── QgateClientError / QgateBackendError attributes ──────────────────────


class TestExceptions:
    def test_client_error_status(self):
        exc = QgateClientError("boom", status_code=503)
        assert exc.status_code == 503

    def test_backend_error_code(self):
        exc = QgateBackendError("nope", error_code="422")
        assert exc.error_code == "422"
