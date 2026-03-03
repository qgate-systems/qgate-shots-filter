"""
config.py — Pydantic v2 configuration models for qgate.

All configuration objects are JSON-serialisable, immutable (``frozen=True``),
and carry field-level validation.

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ThresholdMode = Literal["fixed", "rolling_z", "galton"]
"""Threshold adaptation strategy.

- ``"fixed"``      — Static threshold (no adaptation).
- ``"rolling_z"``  — Legacy rolling z-score gating (existing behaviour).
- ``"galton"``     — Distribution-aware adaptive gating inspired by
                     diffusion / central-limit principles.  Supports both
                     empirical-quantile and z-score sub-modes.
"""

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConditioningVariant(str, Enum):
    """Supported conditioning strategies."""

    GLOBAL = "global"
    HIERARCHICAL = "hierarchical"
    SCORE_FUSION = "score_fusion"


class AdapterKind(str, Enum):
    """Known adapter back-ends."""

    QISKIT = "qiskit"
    CIRQ = "cirq"
    PENNYLANE = "pennylane"
    GROVER_TSVF = "grover_tsvf"
    QAOA_TSVF = "qaoa_tsvf"
    VQE_TSVF = "vqe_tsvf"
    QPE_TSVF = "qpe_tsvf"
    MOCK = "mock"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class FusionConfig(BaseModel):
    """Parameters for α-weighted LF / HF score fusion.

    Attributes:
        alpha:      Weight for the low-frequency component (0 ≤ α ≤ 1).
        threshold:  Accept if combined score ≥ threshold.
        hf_cycles:  Explicit list of cycle indices counted as HF
                    (``None`` → every cycle).
        lf_cycles:  Explicit list of cycle indices counted as LF
                    (``None`` → every 2nd cycle: 0, 2, 4, …).
    """

    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="LF weight (0–1)")
    threshold: float = Field(
        default=0.65, ge=0.0, le=1.0, description="Accept if combined ≥ threshold"
    )
    hf_cycles: Optional[List[int]] = Field(default=None, description="Override HF cycle indices")
    lf_cycles: Optional[List[int]] = Field(default=None, description="Override LF cycle indices")

    model_config = ConfigDict(frozen=True, extra="forbid")


class DynamicThresholdConfig(BaseModel):
    """Parameters for dynamic threshold gating.

    Supports three modes:

    ``"fixed"`` (default)
        No adaptation — uses ``baseline`` as a static threshold.

    ``"rolling_z"``
        Legacy rolling z-score gating:

        .. math:: \\theta_t = \\text{clamp}(\\mu_{\\text{roll}} + z \\cdot \\sigma_{\\text{roll}},\\; \\theta_{\\min},\\; \\theta_{\\max})

    ``"galton"``
        Distribution-aware adaptive gating inspired by diffusion /
        central-limit principles.  The algorithm maintains a rolling window
        of **per-shot** combined scores and sets the threshold so that a
        target fraction of future scores is expected to be accepted.

        Two sub-modes are available:

        * **Quantile** (``use_quantile=True``, recommended) — sets
          :math:`\\theta = Q_{1 - \\text{target\\_acceptance}}(\\text{window})`.
        * **Z-score** — estimates μ and σ from the window, then
          :math:`\\theta = \\mu + z_{\\sigma} \\cdot \\sigma`.  When
          ``robust_stats=True`` the median and MAD-based σ are used.

    Attributes:
        enabled:            Whether dynamic thresholding is active.
        mode:               Threshold strategy (``"fixed"`` | ``"rolling_z"``
                            | ``"galton"``).
        baseline:           Starting / fallback threshold.
        z_factor:           Std-dev multiplier for ``rolling_z`` mode.
        window_size:        Rolling window capacity (batches for rolling_z,
                            individual scores for galton).
        min_threshold:      Floor — threshold never drops below this.
        max_threshold:      Ceiling — threshold never exceeds this.
        min_window_size:    Galton mode: minimum observations before
                            adaptation kicks in (warmup).
        target_acceptance:  Galton quantile mode: target acceptance
                            fraction (one-sided tail).
        robust_stats:       Galton z-score mode: use median + MAD
                            instead of mean + std.
        use_quantile:       Galton mode: prefer empirical quantile
                            (True, default) over z-score.
        z_sigma:            Galton z-score mode: number of σ above
                            centre to place the gate.

    .. note::
       Setting ``mode="galton"`` automatically sets ``enabled=True``
       during validation.  You do **not** need to set both.
    """

    enabled: bool = Field(default=False, description="Enable dynamic thresholding")
    mode: ThresholdMode = Field(
        default="fixed",
        description="Threshold strategy: fixed | rolling_z | galton",
    )
    baseline: float = Field(default=0.65, ge=0.0, le=1.0)
    z_factor: float = Field(default=1.5, ge=0.0, description="Std-dev multiplier (rolling_z)")
    window_size: int = Field(default=10, ge=1, description="Rolling window size")
    min_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_threshold: float = Field(default=0.95, ge=0.0, le=1.0)

    # --- Galton-specific fields (ignored when mode != "galton") -----------
    min_window_size: int = Field(
        default=100,
        ge=1,
        description="Galton: minimum samples before adaptation (warmup)",
    )
    target_acceptance: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Galton quantile: target acceptance fraction",
    )
    robust_stats: bool = Field(
        default=True,
        description="Galton z-score: use median + MAD instead of mean + std",
    )
    use_quantile: bool = Field(
        default=True,
        description="Galton: use empirical quantile (True) or z-score (False)",
    )
    z_sigma: float = Field(
        default=1.645,
        ge=0.0,
        description="Galton z-score: number of σ above centre (~5 % one-sided)",
    )

    @model_validator(mode="after")
    def _min_le_max(self) -> DynamicThresholdConfig:
        if self.min_threshold > self.max_threshold:
            raise ValueError(
                f"min_threshold ({self.min_threshold}) must be ≤ "
                f"max_threshold ({self.max_threshold})"
            )
        return self

    @model_validator(mode="after")
    def _auto_enable_galton(self) -> DynamicThresholdConfig:
        """Automatically set enabled=True when mode is not 'fixed'."""
        if self.mode != "fixed" and not self.enabled:
            object.__setattr__(self, "enabled", True)
        return self

    model_config = ConfigDict(frozen=True, extra="forbid")


class ProbeConfig(BaseModel):
    """Probe-based batch abort configuration.

    Before running a full batch a small *probe* batch is executed.
    If the probe pass-rate is below ``theta`` the full batch is skipped.

    Attributes:
        enabled:     Whether probing is active.
        probe_shots: Number of probe shots.
        theta:       Minimum probe pass-rate to proceed.
    """

    enabled: bool = Field(default=False, description="Enable probe-based abort")
    probe_shots: int = Field(default=100, ge=1)
    theta: float = Field(default=0.65, ge=0.0, le=1.0)

    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Top-level gate configuration
# ---------------------------------------------------------------------------


class GateConfig(BaseModel):
    """Top-level configuration for a qgate trajectory-filter run.

    Compose this from the sub-configs above, or load from JSON / YAML:

        config = GateConfig.model_validate_json(path.read_text())

    Attributes:
        schema_version:  Configuration schema version (for forward compat).
        n_subsystems:    Number of Bell-pair subsystems.
        n_cycles:        Number of monitoring cycles per shot.
        shots:           Total shots to execute per configuration.
        variant:         Conditioning strategy to apply.
        k_fraction:      For hierarchical variant — required pass fraction.
        fusion:          Fusion scoring parameters.
        dynamic_threshold: Rolling z-score threshold adaptation.
        probe:           Probe-based batch abort.
        adapter:         Which adapter back-end to use.
        adapter_options: Arbitrary adapter-specific options (e.g. backend name).
        metadata:        Free-form metadata dict attached to run logs.
    """

    schema_version: str = Field(default="1", description="Config schema version")
    n_subsystems: int = Field(default=4, ge=1, description="Number of Bell-pair subsystems")
    n_cycles: int = Field(default=2, ge=1, description="Monitoring cycles per shot")
    shots: int = Field(default=1024, ge=1, description="Shots per configuration")
    variant: ConditioningVariant = Field(
        default=ConditioningVariant.SCORE_FUSION,
        description="Conditioning strategy",
    )
    k_fraction: float = Field(
        default=0.9, gt=0.0, le=1.0, description="Hierarchical k-of-N fraction"
    )
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    dynamic_threshold: DynamicThresholdConfig = Field(default_factory=DynamicThresholdConfig)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)
    adapter: AdapterKind = Field(default=AdapterKind.MOCK, description="Adapter back-end")
    adapter_options: Dict[str, Any] = Field(
        default_factory=dict, description="Adapter-specific options"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Free-form run metadata")

    model_config = ConfigDict(extra="forbid", frozen=True)
