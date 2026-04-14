"""
Tests for qgate.cloud — QgateAdvantageClient.

All tests use a fake requests.Session to avoid any real network calls.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict
from unittest.mock import MagicMock, call, patch

import pytest
import requests

from qgate.cloud import (
    QgateAdvantageClient,
    QgateAPIError,
    QgateTaskError,
    QgateTimeoutError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(status_code: int, body: Dict[str, Any]) -> MagicMock:
    """Build a fake requests.Response-like MagicMock."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


def _make_client(session: MagicMock, **kwargs: Any) -> QgateAdvantageClient:
    return QgateAdvantageClient(
        api_key="test-key",
        base_url="http://localhost:8000",
        session=session,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestInit:
    def test_empty_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            QgateAdvantageClient(api_key="")

    def test_trailing_slash_stripped(self) -> None:
        c = QgateAdvantageClient(api_key="k", base_url="http://localhost:8000/")
        assert c.base_url == "http://localhost:8000"

    def test_custom_session_is_used(self) -> None:
        sess = MagicMock(spec=requests.Session)
        sess.headers = {}
        c = QgateAdvantageClient(api_key="k", session=sess)
        assert c._session is sess

    def test_default_session_created(self) -> None:
        c = QgateAdvantageClient(api_key="k")
        assert isinstance(c._session, requests.Session)


# ---------------------------------------------------------------------------
# price_asian_fbm — happy path
# ---------------------------------------------------------------------------


class TestPriceAsianFbmSuccess:
    def _make_happy_session(self) -> MagicMock:
        """Session that: (1) accepts submit, (2) returns 'processing', (3) returns 'completed'."""
        sess = MagicMock(spec=requests.Session)
        sess.headers = {}

        submit_resp = _make_response(202, {"task_id": "abc-123", "poll_interval_ms": 100})
        processing_resp = _make_response(200, {"status": "processing"})
        completed_resp = _make_response(
            200,
            {
                "status": "completed",
                "result": {"price": 5.42, "std_error": 0.03, "paths_used": 8000},
            },
        )

        sess.post.return_value = submit_resp
        sess.get.side_effect = [processing_resp, completed_resp]
        return sess

    @patch("qgate.cloud.time.sleep")  # prevent real sleeps
    def test_returns_result_dict(self, mock_sleep: MagicMock) -> None:
        sess = self._make_happy_session()
        client = _make_client(sess, max_interval_s=10.0)

        result = client.price_asian_fbm(
            spot=100.0, strike=105.0, vol=0.2, hurst=0.7, paths=50_000, steps=252
        )

        assert result == {"price": 5.42, "std_error": 0.03, "paths_used": 8000}

    @patch("qgate.cloud.time.sleep")
    def test_submit_payload_shape(self, mock_sleep: MagicMock) -> None:
        sess = self._make_happy_session()
        client = _make_client(sess)

        client.price_asian_fbm(
            spot=100.0, strike=105.0, vol=0.2, hurst=0.7, paths=50_000, steps=252
        )

        _, kwargs = sess.post.call_args
        payload = kwargs["json"]
        assert payload["spot_price"] == 100.0
        assert payload["strike_price"] == 105.0
        assert payload["volatility"] == 0.2
        assert payload["hurst_parameter"] == 0.7
        assert payload["requested_paths"] == 50_000
        assert payload["time_steps"] == 252

    @patch("qgate.cloud.time.sleep")
    def test_submit_url_includes_endpoint(self, mock_sleep: MagicMock) -> None:
        sess = self._make_happy_session()
        client = _make_client(sess)

        client.price_asian_fbm(
            spot=100.0, strike=105.0, vol=0.2, hurst=0.7, paths=1000, steps=10
        )

        submit_url = sess.post.call_args[0][0]
        assert submit_url.endswith("/v1/pricing/asian-fbm/submit")

    @patch("qgate.cloud.time.sleep")
    def test_poll_interval_from_server(self, mock_sleep: MagicMock) -> None:
        """Client should sleep for the server-suggested interval (clamped)."""
        sess = self._make_happy_session()
        client = _make_client(sess, max_interval_s=60.0)

        client.price_asian_fbm(
            spot=100.0, strike=105.0, vol=0.2, hurst=0.7, paths=1000, steps=10
        )

        # poll_interval_ms=100 → 0.1 s; sleep is called once per poll iteration
        for sleep_call in mock_sleep.call_args_list:
            assert sleep_call == call(0.1)

    @patch("qgate.cloud.time.sleep")
    def test_poll_interval_clamped_to_max(self, mock_sleep: MagicMock) -> None:
        """Server suggests 30 s but max_interval_s=5; client must clamp."""
        sess = MagicMock(spec=requests.Session)
        sess.headers = {}
        sess.post.return_value = _make_response(
            202, {"task_id": "xyz", "poll_interval_ms": 30_000}
        )
        sess.get.return_value = _make_response(
            200, {"status": "completed", "result": {"price": 1.0, "std_error": 0.0}}
        )

        client = _make_client(sess, max_interval_s=5.0)
        client.price_asian_fbm(
            spot=100.0, strike=105.0, vol=0.2, hurst=0.7, paths=1000, steps=10
        )

        assert mock_sleep.call_args_list[0] == call(5.0)

    @patch("qgate.cloud.time.sleep")
    def test_dynamic_server_interval_update(self, mock_sleep: MagicMock) -> None:
        """If the status response includes poll_interval_ms, client should adopt it."""
        sess = MagicMock(spec=requests.Session)
        sess.headers = {}
        sess.post.return_value = _make_response(
            202, {"task_id": "dyn", "poll_interval_ms": 500}
        )
        sess.get.side_effect = [
            _make_response(200, {"status": "processing", "poll_interval_ms": 2000}),
            _make_response(200, {"status": "completed", "result": {"price": 2.0}}),
        ]

        client = _make_client(sess, max_interval_s=60.0)
        client.price_asian_fbm(
            spot=100.0, strike=105.0, vol=0.2, hurst=0.7, paths=1000, steps=10
        )

        sleep_values = [c[0][0] for c in mock_sleep.call_args_list]
        assert sleep_values[0] == 0.5   # initial server hint
        assert sleep_values[1] == 2.0   # updated hint from first poll response


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestPriceAsianFbmErrors:
    @patch("qgate.cloud.time.sleep")
    def test_submit_non_202_raises_api_error(self, mock_sleep: MagicMock) -> None:
        sess = MagicMock(spec=requests.Session)
        sess.headers = {}
        sess.post.return_value = _make_response(400, {"detail": "bad request"})

        client = _make_client(sess)
        with pytest.raises(QgateAPIError) as exc_info:
            client.price_asian_fbm(
                spot=100.0, strike=105.0, vol=0.2, hurst=0.7, paths=1000, steps=10
            )

        assert exc_info.value.status_code == 400

    @patch("qgate.cloud.time.sleep")
    def test_failed_task_raises_task_error(self, mock_sleep: MagicMock) -> None:
        sess = MagicMock(spec=requests.Session)
        sess.headers = {}
        sess.post.return_value = _make_response(202, {"task_id": "fail-1", "poll_interval_ms": 50})
        sess.get.return_value = _make_response(
            200, {"status": "failed", "error": "out of memory"}
        )

        client = _make_client(sess)
        with pytest.raises(QgateTaskError, match="fail-1"):
            client.price_asian_fbm(
                spot=100.0, strike=105.0, vol=0.2, hurst=0.7, paths=1000, steps=10
            )

    @patch("qgate.cloud.time.sleep")
    def test_failed_task_includes_error_message(self, mock_sleep: MagicMock) -> None:
        sess = MagicMock(spec=requests.Session)
        sess.headers = {}
        sess.post.return_value = _make_response(202, {"task_id": "t", "poll_interval_ms": 50})
        sess.get.return_value = _make_response(
            200, {"status": "failed", "error": "divergence detected"}
        )

        client = _make_client(sess)
        with pytest.raises(QgateTaskError, match="divergence detected"):
            client.price_asian_fbm(
                spot=100.0, strike=105.0, vol=0.2, hurst=0.7, paths=1000, steps=10
            )

    @patch("qgate.cloud.time.sleep")
    def test_poll_non_200_raises_api_error(self, mock_sleep: MagicMock) -> None:
        sess = MagicMock(spec=requests.Session)
        sess.headers = {}
        sess.post.return_value = _make_response(202, {"task_id": "t2", "poll_interval_ms": 50})
        sess.get.return_value = _make_response(503, {"detail": "service unavailable"})

        client = _make_client(sess)
        with pytest.raises(QgateAPIError) as exc_info:
            client.price_asian_fbm(
                spot=100.0, strike=105.0, vol=0.2, hurst=0.7, paths=1000, steps=10
            )

        assert exc_info.value.status_code == 503

    @patch("qgate.cloud.time.monotonic")
    @patch("qgate.cloud.time.sleep")
    def test_timeout_raises_timeout_error(
        self, mock_sleep: MagicMock, mock_monotonic: MagicMock
    ) -> None:
        """Simulate the deadline being exceeded after the first poll sleep."""
        # monotonic() values: start=0, check after sleep=999 (past deadline of 30)
        mock_monotonic.side_effect = [0.0, 999.0]

        sess = MagicMock(spec=requests.Session)
        sess.headers = {}
        sess.post.return_value = _make_response(202, {"task_id": "slow", "poll_interval_ms": 100})
        # get should never be called — deadline exceeded before first poll
        sess.get.return_value = _make_response(200, {"status": "processing"})

        client = _make_client(sess, timeout_s=30.0)
        with pytest.raises(QgateTimeoutError, match="slow"):
            client.price_asian_fbm(
                spot=100.0, strike=105.0, vol=0.2, hurst=0.7, paths=1000, steps=10
            )

    @patch("qgate.cloud.time.sleep")
    def test_network_error_on_submit_raises_api_error(self, mock_sleep: MagicMock) -> None:
        sess = MagicMock(spec=requests.Session)
        sess.headers = {}
        sess.post.side_effect = requests.ConnectionError("refused")

        client = _make_client(sess)
        with pytest.raises(QgateAPIError, match="Network error submitting"):
            client.price_asian_fbm(
                spot=100.0, strike=105.0, vol=0.2, hurst=0.7, paths=1000, steps=10
            )

    @patch("qgate.cloud.time.sleep")
    def test_network_error_on_poll_raises_api_error(self, mock_sleep: MagicMock) -> None:
        sess = MagicMock(spec=requests.Session)
        sess.headers = {}
        sess.post.return_value = _make_response(202, {"task_id": "net", "poll_interval_ms": 50})
        sess.get.side_effect = requests.ConnectionError("timeout")

        client = _make_client(sess)
        with pytest.raises(QgateAPIError, match="Network error polling"):
            client.price_asian_fbm(
                spot=100.0, strike=105.0, vol=0.2, hurst=0.7, paths=1000, steps=10
            )


# ---------------------------------------------------------------------------
# Top-level re-export
# ---------------------------------------------------------------------------


def test_importable_from_qgate_top_level() -> None:
    from qgate import QgateAdvantageClient as C
    from qgate import QgateAPIError, QgateTaskError, QgateTimeoutError

    assert C is QgateAdvantageClient
    assert issubclass(QgateAPIError, RuntimeError)
    assert issubclass(QgateTaskError, RuntimeError)
    assert issubclass(QgateTimeoutError, TimeoutError)


# ---------------------------------------------------------------------------
# Vectorised strikes
# ---------------------------------------------------------------------------

_GREEK_ROW = {"price": 5.0, "delta": 0.4, "gamma": 0.03, "vega": 20.0,
              "compute_units_used": 100, "latency_ms": 50.0}


def _make_vector_session(strike_list: list, rows: list) -> MagicMock:
    """Session stub that accepts a vectorised submit and returns completed rows."""
    sess = MagicMock(spec=requests.Session)
    sess.headers = {}
    sess.post.return_value = _make_response(
        202, {"task_id": "vec-1", "poll_interval_ms": 50}
    )
    sess.get.return_value = _make_response(
        200, {"status": "completed", "result": rows}
    )
    return sess


class TestVectorisedStrikes:
    @patch("qgate.cloud.time.sleep")
    def test_list_strike_returns_dataframe(self, mock_sleep: MagicMock) -> None:
        import pandas as pd

        rows = [dict(_GREEK_ROW, strike_price=k) for k in [100.0, 105.0, 110.0]]
        sess = _make_vector_session([100.0, 105.0, 110.0], rows)
        client = _make_client(sess)

        result = client.price_asian_fbm(
            spot=100.0, strike=[100.0, 105.0, 110.0], vol=0.2, hurst=0.7,
            paths=1000, steps=10,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result["strike_price"]) == [100.0, 105.0, 110.0]
        assert "price" in result.columns
        assert "delta" in result.columns

    @patch("qgate.cloud.time.sleep")
    def test_ndarray_strike_returns_dataframe(self, mock_sleep: MagicMock) -> None:
        import numpy as np
        import pandas as pd

        strikes = np.array([95.0, 100.0, 105.0])
        rows = [dict(_GREEK_ROW, strike_price=float(k)) for k in strikes]
        sess = _make_vector_session(strikes.tolist(), rows)
        client = _make_client(sess)

        result = client.price_asian_fbm(
            spot=100.0, strike=strikes, vol=0.2, hurst=0.7,
            paths=1000, steps=10,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    @patch("qgate.cloud.time.sleep")
    def test_ndarray_converted_to_list_in_payload(self, mock_sleep: MagicMock) -> None:
        """ndarray strikes must be serialised as a plain Python list in the POST body."""
        import numpy as np

        strikes = np.linspace(90.0, 110.0, 3)
        rows = [dict(_GREEK_ROW, strike_price=float(k)) for k in strikes]
        sess = _make_vector_session(strikes.tolist(), rows)
        client = _make_client(sess)

        client.price_asian_fbm(
            spot=100.0, strike=strikes, vol=0.2, hurst=0.7,
            paths=1000, steps=10,
        )

        _, kwargs = sess.post.call_args
        sent_strike = kwargs["json"]["strike_price"]
        assert isinstance(sent_strike, list), "ndarray must be converted to list for JSON"
        assert sent_strike == strikes.tolist()

    @patch("qgate.cloud.time.sleep")
    def test_scalar_strike_returns_dict(self, mock_sleep: MagicMock) -> None:
        """Scalar strike must still return a plain dict, not a DataFrame."""
        sess = MagicMock(spec=requests.Session)
        sess.headers = {}
        sess.post.return_value = _make_response(
            202, {"task_id": "s-1", "poll_interval_ms": 50}
        )
        sess.get.return_value = _make_response(
            200, {"status": "completed", "result": dict(_GREEK_ROW)}
        )
        client = _make_client(sess)

        result = client.price_asian_fbm(
            spot=100.0, strike=105.0, vol=0.2, hurst=0.7,
            paths=1000, steps=10,
        )

        assert isinstance(result, dict)
        assert result["price"] == _GREEK_ROW["price"]

    @patch("qgate.cloud.time.sleep")
    def test_list_strike_sends_list_in_payload(self, mock_sleep: MagicMock) -> None:
        rows = [dict(_GREEK_ROW, strike_price=k) for k in [100.0, 110.0]]
        sess = _make_vector_session([100.0, 110.0], rows)
        client = _make_client(sess)

        client.price_asian_fbm(
            spot=100.0, strike=[100.0, 110.0], vol=0.2, hurst=0.7,
            paths=1000, steps=10,
        )

        _, kwargs = sess.post.call_args
        assert kwargs["json"]["strike_price"] == [100.0, 110.0]

    @patch("qgate.cloud.pd", None)           # simulate pandas not installed
    @patch("qgate.cloud._PD_AVAILABLE", False)
    @patch("qgate.cloud.time.sleep")
    def test_no_pandas_raises_import_error(self, mock_sleep: MagicMock) -> None:
        rows = [dict(_GREEK_ROW, strike_price=k) for k in [100.0, 105.0]]
        sess = _make_vector_session([100.0, 105.0], rows)
        client = _make_client(sess)

        with pytest.raises(ImportError, match="pandas"):
            client.price_asian_fbm(
                spot=100.0, strike=[100.0, 105.0], vol=0.2, hurst=0.7,
                paths=1000, steps=10,
            )


# ---------------------------------------------------------------------------
# price_european_heston
# ---------------------------------------------------------------------------

_HESTON_SCALAR = {"price": 8.1, "delta": 0.55, "gamma": 0.02, "vega": 18.0,
                  "compute_units_used": 200, "latency_ms": 60.0}

_HESTON_KWARGS = dict(
    spot=100.0, time_to_maturity=0.5, initial_vol=0.04,
    mean_reversion=2.0, long_term_var=0.04, vol_of_vol=0.3,
    correlation=-0.7, paths=5000, steps=100,
)


class TestPriceEuropeanHeston:
    def _make_scalar_session(self) -> MagicMock:
        sess = MagicMock(spec=requests.Session)
        sess.headers = {}
        sess.post.return_value = _make_response(
            202, {"task_id": "hes-1", "poll_interval_ms": 50}
        )
        sess.get.return_value = _make_response(
            200, {"status": "completed", "result": dict(_HESTON_SCALAR)}
        )
        return sess

    @patch("qgate.cloud.time.sleep")
    def test_scalar_strike_returns_dict(self, mock_sleep: MagicMock) -> None:
        sess = self._make_scalar_session()
        client = _make_client(sess)

        result = client.price_european_heston(strike=105.0, **_HESTON_KWARGS)

        assert isinstance(result, dict)
        assert result["price"] == _HESTON_SCALAR["price"]

    @patch("qgate.cloud.time.sleep")
    def test_submit_url(self, mock_sleep: MagicMock) -> None:
        sess = self._make_scalar_session()
        client = _make_client(sess)

        client.price_european_heston(strike=105.0, **_HESTON_KWARGS)

        url = sess.post.call_args[0][0]
        assert url.endswith("/v1/pricing/european-heston/submit")

    @patch("qgate.cloud.time.sleep")
    def test_payload_fields(self, mock_sleep: MagicMock) -> None:
        sess = self._make_scalar_session()
        client = _make_client(sess)

        client.price_european_heston(strike=105.0, **_HESTON_KWARGS)

        payload = sess.post.call_args[1]["json"]
        assert payload["spot_price"] == 100.0
        assert payload["strike_price"] == 105.0
        assert payload["time_to_maturity"] == 0.5
        assert payload["initial_vol"] == 0.04
        assert payload["mean_reversion"] == 2.0
        assert payload["long_term_var"] == 0.04
        assert payload["vol_of_vol"] == 0.3
        assert payload["correlation"] == -0.7
        assert payload["requested_paths"] == 5000
        assert payload["time_steps"] == 100

    @patch("qgate.cloud.time.sleep")
    def test_vector_list_strike_returns_dataframe(self, mock_sleep: MagicMock) -> None:
        import pandas as pd

        strikes = [100.0, 105.0, 110.0]
        rows = [dict(_HESTON_SCALAR, strike_price=k) for k in strikes]
        sess = _make_vector_session(strikes, rows)
        client = _make_client(sess)

        result = client.price_european_heston(strike=strikes, **_HESTON_KWARGS)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result["strike_price"]) == strikes

    @patch("qgate.cloud.time.sleep")
    def test_vector_ndarray_strike_returns_dataframe(self, mock_sleep: MagicMock) -> None:
        import numpy as np
        import pandas as pd

        strikes = np.array([95.0, 100.0, 105.0])
        rows = [dict(_HESTON_SCALAR, strike_price=float(k)) for k in strikes]
        sess = _make_vector_session(strikes.tolist(), rows)
        client = _make_client(sess)

        result = client.price_european_heston(strike=strikes, **_HESTON_KWARGS)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    @patch("qgate.cloud.time.sleep")
    def test_ndarray_strike_serialised_as_list(self, mock_sleep: MagicMock) -> None:
        import numpy as np

        strikes = np.array([100.0, 110.0])
        rows = [dict(_HESTON_SCALAR, strike_price=float(k)) for k in strikes]
        sess = _make_vector_session(strikes.tolist(), rows)
        client = _make_client(sess)

        client.price_european_heston(strike=strikes, **_HESTON_KWARGS)

        payload = sess.post.call_args[1]["json"]
        assert isinstance(payload["strike_price"], list)
        assert payload["strike_price"] == strikes.tolist()


# ---------------------------------------------------------------------------
# price_basket_fbm
# ---------------------------------------------------------------------------

_BASKET_SCALAR = {"price": 12.5, "delta": 0.6, "gamma": 0.01, "vega": 30.0,
                  "compute_units_used": 500, "latency_ms": 120.0}

_BASKET_KWARGS = dict(
    spots=[100.0, 95.0, 105.0],
    volatilities=[0.2, 0.25, 0.18],
    correlation_matrix=[[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]],
    hurst_parameters=[0.7, 0.6, 0.5],
    weights=[0.4, 0.3, 0.3],
    paths=5000,
    steps=100,
)


class TestPriceBasketFbm:
    def _make_scalar_session(self) -> MagicMock:
        sess = MagicMock(spec=requests.Session)
        sess.headers = {}
        sess.post.return_value = _make_response(
            202, {"task_id": "bsk-1", "poll_interval_ms": 50}
        )
        sess.get.return_value = _make_response(
            200, {"status": "completed", "result": dict(_BASKET_SCALAR)}
        )
        return sess

    @patch("qgate.cloud.time.sleep")
    def test_scalar_strike_returns_dict(self, mock_sleep: MagicMock) -> None:
        sess = self._make_scalar_session()
        client = _make_client(sess)

        result = client.price_basket_fbm(strike=100.0, **_BASKET_KWARGS)

        assert isinstance(result, dict)
        assert result["price"] == _BASKET_SCALAR["price"]

    @patch("qgate.cloud.time.sleep")
    def test_submit_url(self, mock_sleep: MagicMock) -> None:
        sess = self._make_scalar_session()
        client = _make_client(sess)

        client.price_basket_fbm(strike=100.0, **_BASKET_KWARGS)

        url = sess.post.call_args[0][0]
        assert url.endswith("/v1/pricing/basket-fbm/submit")

    @patch("qgate.cloud.time.sleep")
    def test_payload_fields(self, mock_sleep: MagicMock) -> None:
        sess = self._make_scalar_session()
        client = _make_client(sess)

        client.price_basket_fbm(strike=100.0, **_BASKET_KWARGS)

        payload = sess.post.call_args[1]["json"]
        assert payload["spot_prices"] == [100.0, 95.0, 105.0]
        assert payload["strike_price"] == 100.0
        assert payload["volatilities"] == [0.2, 0.25, 0.18]
        assert payload["correlation_matrix"] == _BASKET_KWARGS["correlation_matrix"]
        assert payload["hurst_parameters"] == [0.7, 0.6, 0.5]
        assert payload["weights"] == [0.4, 0.3, 0.3]
        assert payload["requested_paths"] == 5000
        assert payload["time_steps"] == 100

    @patch("qgate.cloud.time.sleep")
    def test_ndarray_inputs_serialised_as_lists(self, mock_sleep: MagicMock) -> None:
        """All ndarray basket inputs (spots, vols, corr matrix, hursts, weights)
        must be converted to plain lists before the POST."""
        import numpy as np

        sess = self._make_scalar_session()
        client = _make_client(sess)

        client.price_basket_fbm(
            spots=np.array([100.0, 95.0, 105.0]),
            strike=100.0,
            volatilities=np.array([0.2, 0.25, 0.18]),
            correlation_matrix=np.eye(3),
            hurst_parameters=np.array([0.7, 0.6, 0.5]),
            weights=np.array([0.4, 0.3, 0.3]),
            paths=5000,
            steps=100,
        )

        payload = sess.post.call_args[1]["json"]
        for key in ("spot_prices", "volatilities", "correlation_matrix",
                    "hurst_parameters", "weights"):
            assert isinstance(payload[key], list), f"{key} should be a list, not ndarray"

    @patch("qgate.cloud.time.sleep")
    def test_vector_strike_returns_dataframe(self, mock_sleep: MagicMock) -> None:
        import pandas as pd

        strikes = [98.0, 100.0, 102.0]
        rows = [dict(_BASKET_SCALAR, strike_price=k) for k in strikes]
        sess = _make_vector_session(strikes, rows)
        client = _make_client(sess)

        result = client.price_basket_fbm(strike=strikes, **_BASKET_KWARGS)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result["strike_price"]) == strikes

    @patch("qgate.cloud.time.sleep")
    def test_ndarray_strike_returns_dataframe(self, mock_sleep: MagicMock) -> None:
        import numpy as np
        import pandas as pd

        strikes = np.array([98.0, 100.0, 102.0])
        rows = [dict(_BASKET_SCALAR, strike_price=float(k)) for k in strikes]
        sess = _make_vector_session(strikes.tolist(), rows)
        client = _make_client(sess)

        result = client.price_basket_fbm(strike=strikes, **_BASKET_KWARGS)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
