---
description: >-
  QgateAdvantageClient — Python SDK reference for the Qgate Advantage PPU cloud API.
  Price Asian fBM, European Heston, and Basket fBM options with vectorised NumPy support.
keywords: qgate advantage, cloud API, asian option, heston model, basket option, fractional brownian motion, monte carlo, python sdk
---

# Qgate Advantage Cloud Client

`QgateAdvantageClient` is a synchronous Python SDK over the **Qgate Advantage PPU REST API**.
Heavy stochastic Monte Carlo simulations run on Qgate hardware and are returned via a
transparent async-polling loop — callers see a simple blocking call.

## Installation

The cloud client lives in `qgate.cloud` and ships with the core `qgate` package.
Vectorised (NumPy / Pandas) inputs require the `quant` extras:

```bash
pip install qgate              # core — client available, scalar inputs only
pip install qgate[quant]       # + numpy + pandas for vectorised strikes & DataFrame output
```

## Quick Start

```python
from qgate.cloud import QgateAdvantageClient

client = QgateAdvantageClient(
    api_key="your-api-key",
    base_url="https://api.qgate.systems",   # default — can omit
)
```

---

## Pricing Methods

All three methods share the same **scalar / vectorised** contract for the `strike` argument:

| `strike` type | Return type |
|---|---|
| `float` | `dict` (keys: `price`, `delta`, `gamma`, `vega`, `compute_units_used`, `latency_ms`) |
| `list[float]` or `np.ndarray` | `pandas.DataFrame` (one row per strike, same columns + `strike_price`) |

### `price_asian_fbm` — Asian Option under Fractional BM

An Asian option whose payoff depends on the arithmetic average of the underlying.
The underlying follows a fractional Brownian motion (fBm) parameterised by the
Hurst exponent $H \in (0, 1)$.

$$
dS_t = S_t \, \sigma \, dB_t^H, \qquad \text{payoff} = \max\!\left(\bar{S} - K,\, 0\right)
$$

=== "Scalar strike"

    ```python
    result = client.price_asian_fbm(
        spot=100.0,
        strike=105.0,
        vol=0.20,
        hurst=0.70,    # persistent (trending) process
        paths=100_000,
        steps=252,
    )

    print(f"Price : {result['price']:.4f}")
    print(f"Delta : {result['delta']:.4f}")
    print(f"Gamma : {result['gamma']:.6f}")
    print(f"Vega  : {result['vega']:.4f}")
    ```

=== "Strike strip (NumPy)"

    ```python
    import numpy as np

    strikes = np.linspace(90, 110, 9)   # OTM … ATM … ITM
    df = client.price_asian_fbm(
        spot=100.0,
        strike=strikes,
        vol=0.20,
        hurst=0.70,
        paths=100_000,
        steps=252,
    )

    print(df[["strike_price", "price", "delta"]].to_string(index=False))
    ```

**Parameters**

| Name | Type | Description |
|---|---|---|
| `spot` | `float` | Current underlying spot price $S_0$ |
| `strike` | `float \| list \| np.ndarray` | Strike price(s) $K$ |
| `vol` | `float` | Annualised volatility $\sigma$ |
| `hurst` | `float` | Hurst exponent $H \in (0,1)$. $H = 0.5$ → standard BM; $H > 0.5$ → persistent; $H < 0.5$ → anti-persistent |
| `paths` | `int` | Monte Carlo path count (100 – 10,000,000) |
| `steps` | `int` | Time-steps per path, e.g. 252 for daily (1 – 10,000) |

---

### `price_european_heston` — European Option under Heston SV

A European call/put priced under the Heston stochastic-volatility model, where both
the spot and its instantaneous variance are stochastic:

$$
\begin{aligned}
dS_t &= S_t \sqrt{V_t}\, dW^S_t \\
dV_t &= \kappa(\theta - V_t)\, dt + \xi\sqrt{V_t}\, dW^V_t \\
\langle dW^S_t,\, dW^V_t \rangle &= \rho\, dt
\end{aligned}
$$

=== "Scalar strike"

    ```python
    result = client.price_european_heston(
        spot=100.0,
        strike=105.0,
        time_to_maturity=0.5,   # 6 months
        initial_vol=0.04,        # V₀ = σ₀² = 0.04  →  σ₀ = 20 %
        mean_reversion=2.0,      # κ
        long_term_var=0.04,      # θ (long-run variance)
        vol_of_vol=0.3,          # ξ
        correlation=-0.7,        # ρ  (leverage effect)
        paths=100_000,
        steps=100,
    )
    print(result["price"], result["delta"])
    ```

=== "Strike strip (list)"

    ```python
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    df = client.price_european_heston(
        spot=100.0,
        strike=strikes,
        time_to_maturity=0.5,
        initial_vol=0.04,
        mean_reversion=2.0,
        long_term_var=0.04,
        vol_of_vol=0.3,
        correlation=-0.7,
        paths=100_000,
        steps=100,
    )
    print(df[["strike_price", "price", "vega"]])
    ```

**Parameters**

| Name | Type | Description |
|---|---|---|
| `spot` | `float` | Current spot price $S_0$ |
| `strike` | `float \| list \| np.ndarray` | Strike price(s) $K$ |
| `time_to_maturity` | `float` | Time to expiry in years (e.g. `0.25` for 3 months) |
| `initial_vol` | `float` | Initial instantaneous **variance** $V_0$ (= $\sigma_0^2$) |
| `mean_reversion` | `float` | Mean-reversion speed $\kappa$ |
| `long_term_var` | `float` | Long-run variance $\theta$ |
| `vol_of_vol` | `float` | Volatility of variance $\xi$ |
| `correlation` | `float` | Spot–variance Brownian correlation $\rho \in (-1, 1)$ |
| `paths` | `int` | Monte Carlo path count |
| `steps` | `int` | Time-steps per path |

!!! tip "Feller condition"
    For the variance process to remain strictly positive, the Feller condition
    $2\kappa\theta \geq \xi^2$ should hold.  The server will still compute if
    it is violated, but numerical stability may be reduced.

---

### `price_basket_fbm` — Basket Option under Correlated fBM

Prices a basket option on $n$ assets where each asset $i$ follows its own fBm with
Hurst exponent $H_i$, volatility $\sigma_i$, and spot $S_i^0$.
The assets are correlated via the $n \times n$ correlation matrix $\rho$.

$$
\text{payoff} = \max\!\left(\sum_i w_i S_i^T - K,\; 0\right)
$$

=== "Scalar strike"

    ```python
    import numpy as np

    # 3-asset basket
    spots  = [100.0, 95.0, 105.0]
    vols   = [0.20,  0.25,  0.18]
    hursts = [0.70,  0.60,  0.50]
    weights = [0.40,  0.30,  0.30]

    corr = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0],
    ])

    result = client.price_basket_fbm(
        spots=spots,
        strike=100.0,
        volatilities=vols,
        correlation_matrix=corr,
        hurst_parameters=hursts,
        weights=weights,
        paths=100_000,
        steps=252,
    )
    print(result["price"], result["delta"])
    ```

=== "Strike strip (NumPy)"

    ```python
    import numpy as np

    strikes = np.arange(95.0, 110.0, 2.5)
    df = client.price_basket_fbm(
        spots=[100.0, 95.0, 105.0],
        strike=strikes,
        volatilities=[0.20, 0.25, 0.18],
        correlation_matrix=corr,
        hurst_parameters=[0.70, 0.60, 0.50],
        weights=[0.40, 0.30, 0.30],
        paths=100_000,
        steps=252,
    )
    print(df[["strike_price", "price", "delta"]])
    ```

**Parameters**

| Name | Type | Description |
|---|---|---|
| `spots` | `list \| np.ndarray` | Per-asset spot prices, length $n$ |
| `strike` | `float \| list \| np.ndarray` | Basket strike(s) $K$ |
| `volatilities` | `list \| np.ndarray` | Per-asset annualised volatilities, length $n$ |
| `correlation_matrix` | `list[list] \| np.ndarray` | $n \times n$ asset correlation matrix |
| `hurst_parameters` | `list \| np.ndarray` | Per-asset Hurst exponents $H_i \in (0,1)$, length $n$ |
| `weights` | `list \| np.ndarray` | Basket weights $w_i$, length $n$ (server normalises) |
| `paths` | `int` | Monte Carlo path count |
| `steps` | `int` | Time-steps per path |

!!! note "ndarray serialisation"
    All NumPy inputs are automatically converted to plain Python lists
    before the JSON POST.  You never need to call `.tolist()` manually.

---

## Return Value Shape

### Scalar strike → `dict`

```python
{
    "price": 5.42,
    "delta": 0.43,
    "gamma": 0.032,
    "vega": 22.6,
    "compute_units_used": 12600,
    "latency_ms": 9304.7,
}
```

### Vector strike → `pandas.DataFrame`

```
  strike_price  price  delta  gamma   vega  compute_units_used  latency_ms
0          90.0  12.31   0.71  0.021  25.1               12600      9210.3
1          95.0   8.84   0.58  0.030  23.9               12600      9310.1
2         100.0   5.88   0.44  0.034  22.0               12600      9290.8
3         105.0   3.55   0.31  0.029  19.4               12600      9330.5
4         110.0   1.92   0.20  0.021  15.8               12600      9280.0
```

---

## Error Handling

```python
from qgate.cloud import (
    QgateAdvantageClient,
    QgateAPIError,       # HTTP-level error (4xx / 5xx)
    QgateTaskError,      # Remote simulation failed
    QgateTimeoutError,   # timeout_s exceeded before completion
)

try:
    result = client.price_asian_fbm(...)
except QgateAPIError as e:
    print(f"HTTP {e.status_code}: {e}")   # e.g. 401 bad key, 422 validation
except QgateTaskError as e:
    print(f"Simulation failed: {e}")
except QgateTimeoutError as e:
    print(f"Timed out: {e}")
```

| Exception | Cause |
|---|---|
| `QgateAPIError` | Non-2xx HTTP response — includes `status_code` attribute |
| `QgateTaskError` | Backend task finished with `status == "failed"` |
| `QgateTimeoutError` | `timeout_s` (default 600 s) elapsed before task completed |
| `ImportError` | Vectorised strike given but `pandas` is not installed |

---

## Client Configuration

```python
client = QgateAdvantageClient(
    api_key="your-key",
    base_url="https://api.qgate.systems",  # default
    timeout_s=300.0,       # hard deadline per call (default: 600 s)
    max_interval_s=5.0,    # cap on polling sleep (default: 10 s)
)
```

| Parameter | Default | Description |
|---|---|---|
| `api_key` | — | Required. API key issued by Qgate Systems |
| `base_url` | `https://api.qgate.systems` | Override for staging / local dev |
| `timeout_s` | `600.0` | Hard deadline (seconds) for any single blocking call |
| `max_interval_s` | `10.0` | Upper bound on polling sleep; server hints are respected but clamped |
| `session` | `None` | Inject a custom `requests.Session` (useful for proxies or testing) |

---

## Full API Reference

::: qgate.cloud
