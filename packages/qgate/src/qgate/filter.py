"""
filter.py — TrajectoryFilter: the main qgate API entry-point.

Orchestrates adapter → execute → score → threshold → accept/reject → log.

Usage::

    from qgate import TrajectoryFilter, GateConfig
    from qgate.adapters import MockAdapter

    config = GateConfig(n_subsystems=4, n_cycles=2, shots=1024)
    adapter = MockAdapter(error_rate=0.05, seed=42)
    tf = TrajectoryFilter(config, adapter)
    result = tf.run()
    print(result.acceptance_probability)

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from qgate.adapters.base import BaseAdapter
from qgate.conditioning import (
    ParityOutcome,
    decide_global,
    decide_hierarchical,
)
from qgate.config import ConditioningVariant, GateConfig
from qgate.run_logging import FilterResult, RunLogger, compute_run_id
from qgate.scoring import score_batch
from qgate.threshold import DynamicThreshold, GaltonAdaptiveThreshold

logger = logging.getLogger("qgate.filter")


class TrajectoryFilter:
    """Main API class — build, run, and filter quantum trajectories.

    Typical workflow::

        tf = TrajectoryFilter(config, adapter)
        result = tf.run()                  # build → execute → filter
        result = tf.filter(outcomes)       # filter pre-existing data
        result = tf.filter_counts(counts)  # filter from count dict

    The *adapter* argument accepts:
      - A :class:`BaseAdapter` **instance** (existing usage).
      - A :class:`BaseAdapter` **subclass** (instantiated with no args).
      - An adapter **name string** (e.g. ``"mock"``, ``"qiskit"``)
        resolved via entry-point discovery.

    Args:
        config:  :class:`~qgate.config.GateConfig` with all parameters.
        adapter: Backend adapter — instance, class, or registered name.
        logger:  Optional :class:`RunLogger` for structured output.
    """

    def __init__(
        self,
        config: GateConfig,
        adapter: BaseAdapter | type | str,
        logger: RunLogger | None = None,
    ) -> None:
        self.config = config
        self.adapter = _resolve_adapter(adapter)
        self.logger = logger
        self._dyn_threshold = DynamicThreshold(config.dynamic_threshold)
        self._galton_threshold: GaltonAdaptiveThreshold | None = None
        if config.dynamic_threshold.mode == "galton":
            self._galton_threshold = GaltonAdaptiveThreshold(
                config.dynamic_threshold
            )

    def __repr__(self) -> str:
        return (
            f"TrajectoryFilter(variant={self.config.variant.value!r}, "
            f"n_sub={self.config.n_subsystems}, n_cyc={self.config.n_cycles}, "
            f"shots={self.config.shots}, adapter={type(self.adapter).__name__})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> FilterResult:
        """Build circuit → execute → parse → filter → return result.

        This is the high-level "do everything" method.
        """
        outcomes = self.adapter.build_and_run(
            n_subsystems=self.config.n_subsystems,
            n_cycles=self.config.n_cycles,
            shots=self.config.shots,
        )
        return self.filter(outcomes)

    def filter(self, outcomes: Sequence[ParityOutcome]) -> FilterResult:
        """Apply the configured conditioning + thresholding to *outcomes*.

        Args:
            outcomes: List of ParityOutcome (one per shot).  May be empty.

        Returns:
            :class:`~qgate.run_logging.FilterResult`.
        """
        variant = self.config.variant
        total = len(outcomes)

        # Edge case: empty batch
        if total == 0:
            logger.warning("filter() called with zero outcomes")
            config_dump = self.config.model_dump_json(indent=2)
            adapter_name = type(self.adapter).__name__
            return FilterResult(
                run_id=compute_run_id(config_dump, adapter_name=adapter_name),
                variant=variant.value,
                total_shots=0,
                accepted_shots=0,
                acceptance_probability=0.0,
                tts=float("inf"),
                config_json=config_dump,
                metadata=dict(self.config.metadata),
            )

        logger.info(
            "Filtering %d outcomes — variant=%s, n_sub=%d, n_cyc=%d",
            total, variant.value, self.config.n_subsystems, self.config.n_cycles,
        )

        # Vectorised per-shot scoring
        scored = score_batch(
            outcomes,
            alpha=self.config.fusion.alpha,
            hf_cycles=self.config.fusion.hf_cycles,
            lf_cycles=self.config.fusion.lf_cycles,
        )
        combined_scores = [s[2] for s in scored]

        # --- Determine threshold to use --------------------------------
        dt_cfg = self.config.dynamic_threshold

        if dt_cfg.mode == "galton" and self._galton_threshold is not None:
            # Galton mode: feed per-shot scores into the adaptive window
            # and use the resulting threshold for score_fusion gating.
            self._galton_threshold.observe_batch(combined_scores)
            threshold = self._galton_threshold.current_threshold
            snap = self._galton_threshold.last_snapshot
            logger.debug(
                "Galton threshold → %.4f  (warmup=%s, window=%d)",
                threshold, snap.in_warmup, snap.window_size_current,
            )
        elif dt_cfg.enabled and dt_cfg.mode in ("rolling_z", "fixed") and combined_scores:
            batch_mean = float(np.mean(combined_scores))
            threshold = self._dyn_threshold.update(batch_mean)
            logger.debug("Dynamic threshold updated → %.4f (batch mean=%.4f)", threshold, batch_mean)
        else:
            threshold = self.config.fusion.threshold

        # Apply conditioning rule
        accepted_count = 0
        for i, outcome in enumerate(outcomes):
            if variant == ConditioningVariant.GLOBAL:
                if decide_global(outcome):
                    accepted_count += 1
            elif variant == ConditioningVariant.HIERARCHICAL:
                if decide_hierarchical(outcome, self.config.k_fraction):
                    accepted_count += 1
            elif (
                variant == ConditioningVariant.SCORE_FUSION
                and combined_scores[i] >= threshold
            ):
                accepted_count += 1

        acc_prob = accepted_count / total if total > 0 else 0.0
        tts = 1.0 / acc_prob if acc_prob > 0 else float("inf")

        config_dump = self.config.model_dump_json(indent=2)
        adapter_name = type(self.adapter).__name__

        # --- Build galton telemetry metadata --------------------------
        galton_meta: dict[str, object] = {}
        if dt_cfg.mode == "galton" and self._galton_threshold is not None:
            snap = self._galton_threshold.last_snapshot
            galton_meta = {
                "galton_rolling_mean": snap.rolling_mean,
                "galton_rolling_sigma": snap.rolling_sigma,
                "galton_rolling_quantile": snap.rolling_quantile,
                "galton_effective_threshold": snap.effective_threshold,
                "galton_window_size_current": snap.window_size_current,
                "galton_acceptance_rate_rolling": snap.acceptance_rate_rolling,
                "galton_in_warmup": snap.in_warmup,
            }

        # Merge galton telemetry into metadata
        result_metadata = dict(self.config.metadata)
        if galton_meta:
            result_metadata["galton"] = galton_meta

        # --- Determine dynamic threshold final value ------------------
        dyn_final: float | None = None
        if dt_cfg.mode == "galton" and self._galton_threshold is not None:
            dyn_final = self._galton_threshold.current_threshold
        elif dt_cfg.enabled:
            dyn_final = self._dyn_threshold.current_threshold

        result = FilterResult(
            run_id=compute_run_id(config_dump, adapter_name=adapter_name),
            variant=variant.value,
            total_shots=total,
            accepted_shots=accepted_count,
            acceptance_probability=acc_prob,
            tts=tts,
            mean_combined_score=float(np.mean(combined_scores)) if combined_scores else None,
            threshold_used=threshold,
            dynamic_threshold_final=dyn_final,
            scores=combined_scores,
            config_json=config_dump,
            metadata=result_metadata,
        )

        logger.info(
            "Result: %d/%d accepted (P=%.4f, TTS=%.2f)",
            accepted_count, total, acc_prob, tts,
        )

        if self.logger is not None:
            self.logger.log(result)

        return result

    def filter_counts(
        self,
        counts: dict,
        n_subsystems: int | None = None,
        n_cycles: int | None = None,
    ) -> FilterResult:
        """Filter from a pre-existing count dictionary.

        This is a convenience for working with raw Qiskit-style count
        dictionaries when you already have results.

        Args:
            counts:       ``{bitstring: count}`` mapping.
            n_subsystems: Override (defaults to ``config.n_subsystems``).
            n_cycles:     Override (defaults to ``config.n_cycles``).
        """
        n_sub = n_subsystems or self.config.n_subsystems
        n_cyc = n_cycles or self.config.n_cycles
        outcomes = self.adapter.parse_results(counts, n_sub, n_cyc)
        return self.filter(outcomes)

    # ------------------------------------------------------------------
    # Threshold introspection
    # ------------------------------------------------------------------

    @property
    def current_threshold(self) -> float:
        """The current effective threshold (may be dynamic)."""
        if self.config.dynamic_threshold.mode == "galton" and self._galton_threshold is not None:
            return self._galton_threshold.current_threshold
        if self.config.dynamic_threshold.enabled:
            return self._dyn_threshold.current_threshold
        return self.config.fusion.threshold

    def reset_threshold(self) -> None:
        """Reset the dynamic threshold to baseline."""
        self._dyn_threshold.reset()
        if self._galton_threshold is not None:
            self._galton_threshold.reset()

    @property
    def galton_snapshot(self) -> object | None:
        """The latest :class:`GaltonSnapshot`, or ``None`` if not in galton mode."""
        if self._galton_threshold is not None:
            return self._galton_threshold.last_snapshot
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_adapter(adapter: BaseAdapter | type | str) -> BaseAdapter:
    """Resolve an adapter argument to a :class:`BaseAdapter` instance.

    Accepts:
      - A :class:`BaseAdapter` instance (returned as-is).
      - A :class:`BaseAdapter` subclass (instantiated with no args).
      - A string name (resolved via entry-point discovery).
    """
    if isinstance(adapter, BaseAdapter):
        return adapter
    if isinstance(adapter, str):
        from qgate.adapters.registry import load_adapter

        cls = load_adapter(adapter)
        return cls()  # type: ignore[no-any-return]
    if isinstance(adapter, type) and issubclass(adapter, BaseAdapter):
        return adapter()
    raise TypeError(
        f"adapter must be a BaseAdapter instance, subclass, or registered name string, "
        f"got {type(adapter).__name__}"
    )
