"""
cloud.py — Qgate Advantage cloud API client.

Provides a synchronous-feeling SDK over the asynchronous Qgate Advantage
PPU (Processing Power Unit) REST API.  Heavy stochastic simulations are
submitted as background tasks; the client polls transparently so callers
see a simple blocking call.

Example — scalar strike (returns ``dict``)::

    from qgate.cloud import QgateAdvantageClient

    client = QgateAdvantageClient(api_key="your-key")
    result = client.price_asian_fbm(
        spot=100.0, strike=105.0, vol=0.2, hurst=0.7, paths=100_000, steps=252
    )
    print(result["price"], result["delta"])

Example — vectorised strikes (returns ``pandas.DataFrame``)::

    import numpy as np
    strikes = np.linspace(90, 110, 5)
    df = client.price_asian_fbm(
        spot=100.0, strike=strikes, vol=0.2, hurst=0.7, paths=100_000, steps=252
    )
    print(df[["strike_price", "price", "delta"]])
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

import requests

# numpy / pandas are optional heavy deps; import gracefully so the module
# still loads in environments that only have the core qgate install.
try:
    import numpy as np

    _NP_AVAILABLE = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _NP_AVAILABLE = False

try:
    import pandas as pd

    _PD_AVAILABLE = True
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    _PD_AVAILABLE = False

__all__ = ["QgateAdvantageClient", "QgateAPIError", "QgateTaskError", "QgateTimeoutError"]

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "https://api.qgate.systems"
_DEFAULT_TIMEOUT_S = 600.0  # 10-minute hard ceiling per call
_DEFAULT_MAX_INTERVAL_S = 10.0  # cap exponential back-off at 10 s


class QgateAPIError(RuntimeError):
    """Raised when the Qgate Advantage API returns a non-2xx response."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class QgateTaskError(RuntimeError):
    """Raised when the remote simulation task terminates in a failed state."""


class QgateTimeoutError(TimeoutError):
    """Raised when *timeout_s* elapses before the task completes."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class QgateAdvantageClient:
    """Client for the Qgate Advantage PPU API.

    Compresses heavy stochastic Monte Carlo simulations via trajectory
    filtering on Qgate hardware, returning statistically equivalent results
    with far fewer raw paths.

    Args:
        api_key:        Your Qgate Advantage API key.
        base_url:       Base URL of the API (default: production endpoint).
        timeout_s:      Hard timeout in seconds for any single blocking call
                        (default: 600 s / 10 min).
        max_interval_s: Upper bound on the polling interval in seconds;
                        the client uses the server-suggested interval but
                        never sleeps longer than this value (default: 10 s).
        session:        Optional pre-configured :class:`requests.Session` to
                        use (useful for injecting test fakes or proxies).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        max_interval_s: float = _DEFAULT_MAX_INTERVAL_S,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be a non-empty string")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_interval_s = max_interval_s

        self._session: requests.Session = session or requests.Session()
        self._session.headers.update(
            {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Public pricing methods
    # ------------------------------------------------------------------

    def price_asian_fbm(
        self,
        spot: float,
        strike: Union[float, List[float], "np.ndarray"],  # type: ignore[type-arg]
        vol: float,
        hurst: float,
        paths: int,
        steps: int,
    ) -> "Union[Dict[str, Any], pd.DataFrame]":  # type: ignore[type-arg]
        """Price a Fractional Brownian Motion (fBm) Asian option.

        Supports both scalar and vectorised strike inputs.

        Args:
            spot:   Current underlying spot price.
            strike: Option strike price(s).  Pass a single :class:`float`
                    to price one option; pass a :class:`list` or
                    :class:`numpy.ndarray` to price a strip of options in a
                    single API call.
            vol:    Annualised volatility (e.g. ``0.20`` for 20 %).
            hurst:  Hurst exponent H ∈ (0, 1).  H = 0.5 → standard
                    Brownian motion; H > 0.5 → persistent (trending);
                    H < 0.5 → anti-persistent (mean-reverting).
            paths:  Number of Monte Carlo paths requested.
            steps:  Number of time-steps per path.

        Returns:
            * **scalar strike** → ``dict`` with keys ``"price"``,
              ``"delta"``, ``"gamma"``, ``"vega"``,
              ``"compute_units_used"``, ``"latency_ms"``.
            * **vector strike** (``list`` / ``np.ndarray``) →
              :class:`pandas.DataFrame` with one row per strike and the
              same columns as the scalar result plus ``"strike_price"``.
              Requires ``pandas`` to be installed.

        Raises:
            QgateAPIError:     HTTP error from the API.
            QgateTaskError:    Remote task finished in a failed state.
            QgateTimeoutError: Task did not complete within *timeout_s*.
            ImportError:       ``pandas`` is not installed and a vectorised
                               strike was supplied.
        """
        strike_payload, vectorised = self._normalise_strike(strike)

        payload: Dict[str, Any] = {
            "spot_price": spot,
            "strike_price": strike_payload,
            "volatility": vol,
            "hurst_parameter": hurst,
            "requested_paths": paths,
            "time_steps": steps,
        }

        task_id, poll_interval_s = self._submit("/v1/pricing/asian-fbm/submit", payload)
        return self._shape_result(self._wait_for_result(task_id, poll_interval_s), vectorised)

    def price_european_heston(
        self,
        spot: float,
        strike: Union[float, List[float], "np.ndarray"],  # type: ignore[type-arg]
        time_to_maturity: float,
        initial_vol: float,
        mean_reversion: float,
        long_term_var: float,
        vol_of_vol: float,
        correlation: float,
        paths: int,
        steps: int,
    ) -> "Union[Dict[str, Any], pd.DataFrame]":  # type: ignore[type-arg]
        """Price a European option under the Heston stochastic-volatility model.

        The Heston model describes the underlying with two coupled SDEs:

        .. math::

            dS_t &= S_t \\sqrt{V_t}\\, dW^S_t \\\\
            dV_t &= \\kappa (\\theta - V_t)\\, dt + \\xi \\sqrt{V_t}\\, dW^V_t

        where :math:`\\langle dW^S_t, dW^V_t \\rangle = \\rho\\, dt`.

        Args:
            spot:             Current underlying spot price.
            strike:           Option strike price(s).  Scalar returns a
                              ``dict``; ``list`` / ``np.ndarray`` returns a
                              :class:`pandas.DataFrame`.
            time_to_maturity: Time to expiry in years (e.g. ``0.25`` for
                              3 months).
            initial_vol:      Initial instantaneous variance :math:`V_0`
                              (i.e. :math:`\\sigma_0^2`).
            mean_reversion:   Mean-reversion speed :math:`\\kappa` (κ).
            long_term_var:    Long-run variance :math:`\\theta` (θ).
            vol_of_vol:       Volatility of variance :math:`\\xi` (ξ).
            correlation:      Correlation :math:`\\rho` between the spot and
                              variance Brownian motions (ρ ∈ (−1, 1)).
            paths:            Number of Monte Carlo paths.
            steps:            Number of time-steps per path.

        Returns:
            * **scalar strike** → ``dict``
            * **vector strike** → :class:`pandas.DataFrame`

        Raises:
            QgateAPIError:     HTTP error from the API.
            QgateTaskError:    Remote task finished in a failed state.
            QgateTimeoutError: Task did not complete within *timeout_s*.
            ImportError:       ``pandas`` not installed and vector strike given.
        """
        strike_payload, vectorised = self._normalise_strike(strike)

        payload: Dict[str, Any] = {
            "spot_price": spot,
            "strike_price": strike_payload,
            "time_to_maturity": time_to_maturity,
            "initial_vol": initial_vol,
            "mean_reversion": mean_reversion,
            "long_term_var": long_term_var,
            "vol_of_vol": vol_of_vol,
            "correlation": correlation,
            "requested_paths": paths,
            "time_steps": steps,
        }

        task_id, poll_interval_s = self._submit("/v1/pricing/european-heston/submit", payload)
        return self._shape_result(self._wait_for_result(task_id, poll_interval_s), vectorised)

    def price_basket_fbm(
        self,
        spots: Union[List[float], "np.ndarray"],  # type: ignore[type-arg]
        strike: Union[float, List[float], "np.ndarray"],  # type: ignore[type-arg]
        volatilities: Union[List[float], "np.ndarray"],  # type: ignore[type-arg]
        correlation_matrix: Union[List[List[float]], "np.ndarray"],  # type: ignore[type-arg]
        hurst_parameters: Union[List[float], "np.ndarray"],  # type: ignore[type-arg]
        weights: Union[List[float], "np.ndarray"],  # type: ignore[type-arg]
        paths: int,
        steps: int,
    ) -> "Union[Dict[str, Any], pd.DataFrame]":  # type: ignore[type-arg]
        """Price a basket option under correlated Fractional Brownian Motion.

        Each asset :math:`i` follows an fBm with its own Hurst exponent
        :math:`H_i`, volatility :math:`\\sigma_i`, and spot :math:`S_i^0`.
        The basket payoff is computed from the weighted average:

        .. math:: B_T = \\sum_i w_i S_i^T

        Args:
            spots:              Spot prices for each asset in the basket
                                (length *n*).
            strike:             Basket strike price(s).  Scalar returns a
                                ``dict``; ``list`` / ``np.ndarray`` returns a
                                :class:`pandas.DataFrame`.
            volatilities:       Per-asset annualised volatilities (length *n*).
            correlation_matrix: *n × n* correlation matrix between assets.
                                Accepts a nested list or 2-D ``np.ndarray``.
            hurst_parameters:   Per-asset Hurst exponents :math:`H_i \\in (0,1)`
                                (length *n*).
            weights:            Basket weights :math:`w_i` (length *n*;
                                need not sum to 1 — normalisation is
                                server-side).
            paths:              Number of Monte Carlo paths.
            steps:              Number of time-steps per path.

        Returns:
            * **scalar strike** → ``dict``
            * **vector strike** → :class:`pandas.DataFrame`

        Raises:
            QgateAPIError:     HTTP error from the API.
            QgateTaskError:    Remote task finished in a failed state.
            QgateTimeoutError: Task did not complete within *timeout_s*.
            ImportError:       ``pandas`` not installed and vector strike given.
        """
        strike_payload, vectorised = self._normalise_strike(strike)

        payload: Dict[str, Any] = {
            "spot_prices": self._to_list(spots),
            "strike_price": strike_payload,
            "volatilities": self._to_list(volatilities),
            "correlation_matrix": self._to_list(correlation_matrix),
            "hurst_parameters": self._to_list(hurst_parameters),
            "weights": self._to_list(weights),
            "requested_paths": paths,
            "time_steps": steps,
        }

        task_id, poll_interval_s = self._submit("/v1/pricing/basket-fbm/submit", payload)
        return self._shape_result(self._wait_for_result(task_id, poll_interval_s), vectorised)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_list(value: Any) -> Any:
        """Convert a numpy ndarray to a plain Python list; leave everything else unchanged."""
        if _NP_AVAILABLE and isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def _normalise_strike(
        self,
        strike: Union[float, List[float], "np.ndarray"],  # type: ignore[type-arg]
    ) -> "tuple[Any, bool]":
        """Return ``(strike_payload, vectorised)``.

        *vectorised* is ``True`` when the caller passed a ``list`` or
        ``np.ndarray``; the returned *strike_payload* is always
        JSON-serialisable (ndarray converted to list).
        """
        vectorised = isinstance(strike, list) or (
            _NP_AVAILABLE and isinstance(strike, np.ndarray)
        )
        return self._to_list(strike), vectorised

    def _shape_result(
        self,
        raw: Any,
        vectorised: bool,
    ) -> "Union[Dict[str, Any], pd.DataFrame]":  # type: ignore[type-arg]
        """Return *raw* as-is for scalar calls or wrap it in a DataFrame for vector calls.

        Raises:
            ImportError: ``pandas`` is not installed and *vectorised* is ``True``.
        """
        if not vectorised:
            return raw  # type: ignore[return-value]

        if not _PD_AVAILABLE:
            raise ImportError(
                "pandas is required for vectorised strike inputs. "
                "Install it with: pip install pandas"
            )
        return pd.DataFrame(raw)  # type: ignore[return-value]

    def _submit(self, endpoint: str, payload: Dict[str, Any]) -> "tuple[str, float]":
        """POST *payload* to *endpoint* and return ``(task_id, poll_interval_s)``."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self._session.post(url, json=payload, timeout=30)
        except requests.RequestException as exc:
            raise QgateAPIError(f"Network error submitting task: {exc}") from exc

        if response.status_code != 202:
            raise QgateAPIError(
                f"Failed to submit task to {endpoint}: {response.text}",
                status_code=response.status_code,
            )

        data: Dict[str, Any] = response.json()
        task_id: str = data["task_id"]
        # Server hint: how long to sleep before the first poll.
        poll_interval_s: float = min(
            data.get("poll_interval_ms", 1000) / 1000.0,
            self.max_interval_s,
        )
        return task_id, poll_interval_s

    def _wait_for_result(self, task_id: str, poll_interval_s: float) -> Dict[str, Any]:
        """Poll ``/v1/tasks/{task_id}`` until the task completes or times out.

        Uses the server-suggested *poll_interval_s* clamped to
        ``max_interval_s``.

        Raises:
            QgateTaskError:    Remote task status is ``"failed"``.
            QgateTimeoutError: Deadline exceeded before completion.
            QgateAPIError:     Unexpected HTTP error while polling.
        """
        status_url = f"{self.base_url}/v1/tasks/{task_id}"
        deadline = time.monotonic() + self.timeout_s

        while True:
            # Respect the server hint, but never exceed our own ceiling.
            sleep_s = min(poll_interval_s, self.max_interval_s)
            time.sleep(sleep_s)

            if time.monotonic() >= deadline:
                raise QgateTimeoutError(
                    f"Task {task_id} did not complete within {self.timeout_s:.0f} s"
                )

            try:
                response = self._session.get(status_url, timeout=30)
            except requests.RequestException as exc:
                raise QgateAPIError(f"Network error polling task {task_id}: {exc}") from exc

            if response.status_code != 200:
                raise QgateAPIError(
                    f"Unexpected status while polling task {task_id}: {response.text}",
                    status_code=response.status_code,
                )

            data: Dict[str, Any] = response.json()
            status: str = data.get("status", "")

            if status == "completed":
                return data["result"]

            if status == "failed":
                error_msg: str = data.get("error", "no details provided")
                raise QgateTaskError(
                    f"Simulation failed on Qgate servers for task {task_id}: {error_msg}"
                )

            # "processing" | "queued" | any other in-progress state → keep polling.
            # If the server updates its suggested interval dynamically, respect it.
            if "poll_interval_ms" in data:
                poll_interval_s = min(
                    data["poll_interval_ms"] / 1000.0,
                    self.max_interval_s,
                )
