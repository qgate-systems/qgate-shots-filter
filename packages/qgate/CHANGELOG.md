# Changelog

All notable changes to **qgate** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.5.0] — 2026-02-22

### Added

- **Galton adaptive thresholding** — new `mode="galton"` in
  `DynamicThresholdConfig` provides distribution-aware, rolling-window
  gating inspired by diffusion / central-limit principles.  Targets a
  stable acceptance fraction under hardware drift.
  - **Quantile sub-mode** (default, `use_quantile=True`) — sets threshold
    at the empirical (1 − `target_acceptance`) quantile of the score window.
  - **Z-score sub-mode** (`use_quantile=False`) — estimates μ ± z·σ with
    optional robust statistics (median + MAD × 1.4826).
  - **Warmup period** — threshold falls back to `baseline` until
    `min_window_size` scores have been observed.
  - Galton telemetry (rolling mean, sigma, quantile, acceptance rate,
    window size) is logged to `FilterResult.metadata["galton"]`.
- **`GaltonAdaptiveThreshold`** class in `threshold.py` — standalone
  adaptive threshold with `observe()`, `observe_batch()`, `reset()`, and
  a `GaltonSnapshot` dataclass for telemetry.
- **`estimate_diffusion_width()`** utility — variance estimator with
  robust (MAD) and standard modes.
- **`ThresholdMode`** type alias — `Literal["fixed", "rolling_z", "galton"]`.
- **`TrajectoryFilter.galton_snapshot`** property — introspect the latest
  `GaltonSnapshot` when galton mode is active.
- **Config auto-enable** — setting `mode="galton"` or `mode="rolling_z"`
  automatically sets `enabled=True`.
- **27 new unit tests** for galton mode — quantile accuracy, robust stats
  under outliers, warmup, clamping, window management, integration with
  `TrajectoryFilter`, and `estimate_diffusion_width`.

### Changed

- `DynamicThresholdConfig` now accepts `mode`, `min_window_size`,
  `target_acceptance`, `robust_stats`, `use_quantile`, and `z_sigma`
  fields (all optional, backward-compatible defaults).
- `TrajectoryFilter.filter()` routes to `GaltonAdaptiveThreshold` when
  `mode="galton"`, feeding per-shot scores instead of batch means.

## [0.4.0] — 2026-02-19

### Added

- **Vectorised internals** — `ParityOutcome.parity_matrix` is now a
  `numpy.ndarray` (shape `(n_cycles, n_subsystems)`, dtype `int8`).
  All scoring, conditioning, and filtering hot-paths use NumPy instead of
  Python loops.  `score_batch()` stacks matrices into a single 3-D array
  for fully vectorised batch scoring.
- **`pass_rates` property** on `ParityOutcome` — returns per-cycle pass
  rates as an ndarray, used internally by scoring and decision functions.
- **`RunLogger` context-manager** — `with RunLogger("log.jsonl") as rl:`
  now supported; calls `close()` on exit.
- **Parquet buffered writes** — Parquet records accumulate in memory and
  flush on `close()`, avoiding per-shot file rewrites.
- **`TrajectoryFilter.__repr__`** — human-readable summary string.
- **CLI flags** — `--verbose`/`-v`, `--quiet`/`-q`, and
  `--error-rate`/`-e` (mock adapter error rate override).
- **stdlib `logging`** — `filter.py`, `threshold.py`, `run_logging.py`,
  `qiskit_adapter.py` emit structured log messages via
  `logging.getLogger("qgate.*")`.
- **22 new edge-case tests** — empty inputs, `n_subsystems=1`,
  ndarray coercion, frozen config, `filter_counts`, Parquet logging,
  CLI flags, Qiskit copy safety.
- **`csv` optional extra** — `pip install qgate[csv]` installs pandas
  for CSV logging without pulling the full `[all]` bundle.

### Changed

- **`GateConfig` is now frozen** — all Pydantic models use
  `ConfigDict(frozen=True, extra="forbid")`, preventing accidental
  mutation after construction.
- **pandas is no longer a core dependency** — moved to `[csv]` and
  `[parquet]` extras.  Lazy-imported at first use with a clear error
  message when absent.
- **`Dict[str, object]` → `Dict[str, Any]`** for `adapter_options` and
  `metadata` fields in `GateConfig` (fixes downstream typing issues).
- **Removed redundant `k_fraction` validator** — `Field(gt=0.0, le=1.0)`
  constraints are sufficient.
- **`compute_window_metric` de-duplicated** —
  `compat/monitors.py` now re-exports from `scoring.py` instead of
  carrying a copy.
- **`MultiRateMonitor` type hints** — `hf_scores` and `lf_scores` are
  typed as `list[float]`.
- **`threshold.py` docstring** — corrected formula to
  `rolling_mean + z_factor × rolling_std`.
- **MkDocs CDN** — replaced compromised `polyfill.io` with
  `cdnjs.cloudflare.com` for MathJax.

### Fixed

- **Qiskit adapter aliasing bug** — each shot from `parse_results()` now
  receives an independent ndarray copy of the parity matrix.  Previously,
  all shots sharing a bitstring shared the same object; mutating one
  silently corrupted the rest.
- **Empty outcomes** — `TrajectoryFilter.filter([])` returns a zeroed
  `FilterResult` instead of raising.
- **Unknown log extension** — `RunLogger("data.xyz")` now emits a
  warning and falls back to JSONL instead of silently misbehaving.

---

## [0.3.0] — 2025-07-07

### Added

- Multi-rate monitoring (`MultiRateMonitor`, HF / LF cycle partitions).
- Dynamic threshold adaptation with z-factor and rolling window.
- Probe-based early abort (`should_abort_batch`).
- Run logging to JSON-Lines, CSV, and Parquet (`RunLogger`).
- Deterministic SHA-256 run IDs for reproducibility.
- Adapter registry with `list_adapters()` / `load_adapter()`.
- Full Qiskit adapter with scramble layers and mid-circuit measurement.
- CLI (`qgate run`, `qgate validate`, `qgate schema`, `qgate adapters`, `qgate version`).
- MkDocs-based documentation site.
- Comprehensive test suite (152 tests).

### Changed

- Flat module layout refactored to `compat/` sub-package for backward
  compatibility.

---

## [0.2.0] — 2025-06-28

### Added

- Score-fusion conditioning variant.
- Configurable `alpha` blending parameter.
- `GateConfig` Pydantic model with validation.

---

## [0.1.0] — 2025-06-15

### Added

- Initial release.
- Global and hierarchical conditioning.
- `MockAdapter` for testing.
- Basic `TrajectoryFilter.run()` pipeline.
