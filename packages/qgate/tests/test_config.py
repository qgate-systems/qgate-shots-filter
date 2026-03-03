"""Tests for qgate.config — Pydantic v2 configuration models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from qgate.config import (
    AdapterKind,
    ConditioningVariant,
    DynamicThresholdConfig,
    FusionConfig,
    GateConfig,
    ProbeConfig,
)

# ---------------------------------------------------------------------------
# FusionConfig
# ---------------------------------------------------------------------------


class TestFusionConfig:
    def test_defaults(self):
        fc = FusionConfig()
        assert fc.alpha == 0.5
        assert fc.threshold == 0.65
        assert fc.hf_cycles is None
        assert fc.lf_cycles is None

    def test_custom_values(self):
        fc = FusionConfig(alpha=0.3, threshold=0.8, hf_cycles=[0, 1], lf_cycles=[0])
        assert fc.alpha == 0.3
        assert fc.threshold == 0.8

    def test_alpha_out_of_range(self):
        with pytest.raises(ValidationError):
            FusionConfig(alpha=1.5)
        with pytest.raises(ValidationError):
            FusionConfig(alpha=-0.1)

    def test_frozen(self):
        fc = FusionConfig()
        with pytest.raises(ValidationError):
            fc.alpha = 0.9  # type: ignore[misc]

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            FusionConfig(unknown_field=42)  # type: ignore[call-arg]

    def test_json_round_trip(self):
        fc = FusionConfig(alpha=0.7, threshold=0.55)
        data = fc.model_dump_json()
        fc2 = FusionConfig.model_validate_json(data)
        assert fc == fc2


# ---------------------------------------------------------------------------
# DynamicThresholdConfig
# ---------------------------------------------------------------------------


class TestDynamicThresholdConfig:
    def test_defaults(self):
        dtc = DynamicThresholdConfig()
        assert dtc.enabled is False
        assert dtc.baseline == 0.65
        assert dtc.z_factor == 1.5
        assert dtc.window_size == 10

    def test_min_gt_max_raises(self):
        with pytest.raises(ValidationError, match="min_threshold"):
            DynamicThresholdConfig(min_threshold=0.9, max_threshold=0.3)

    def test_valid_min_max(self):
        dtc = DynamicThresholdConfig(min_threshold=0.5, max_threshold=0.8)
        assert dtc.min_threshold == 0.5
        assert dtc.max_threshold == 0.8


# ---------------------------------------------------------------------------
# ProbeConfig
# ---------------------------------------------------------------------------


class TestProbeConfig:
    def test_defaults(self):
        pc = ProbeConfig()
        assert pc.enabled is False
        assert pc.probe_shots == 100
        assert pc.theta == 0.65

    def test_probe_shots_must_be_positive(self):
        with pytest.raises(ValidationError):
            ProbeConfig(probe_shots=0)


# ---------------------------------------------------------------------------
# GateConfig
# ---------------------------------------------------------------------------


class TestGateConfig:
    def test_defaults(self):
        gc = GateConfig()
        assert gc.schema_version == "1"
        assert gc.n_subsystems == 4
        assert gc.n_cycles == 2
        assert gc.shots == 1024
        assert gc.variant == ConditioningVariant.SCORE_FUSION
        assert gc.k_fraction == 0.9
        assert gc.adapter == AdapterKind.MOCK

    def test_custom(self):
        gc = GateConfig(
            n_subsystems=8,
            n_cycles=4,
            shots=2048,
            variant=ConditioningVariant.HIERARCHICAL,
            k_fraction=0.75,
            adapter=AdapterKind.QISKIT,
        )
        assert gc.n_subsystems == 8
        assert gc.variant == ConditioningVariant.HIERARCHICAL
        assert gc.adapter == AdapterKind.QISKIT

    def test_k_fraction_validation(self):
        with pytest.raises(ValidationError):
            GateConfig(k_fraction=0.0)
        with pytest.raises(ValidationError):
            GateConfig(k_fraction=1.5)

    def test_json_round_trip(self):
        gc = GateConfig(
            n_subsystems=6,
            variant="hierarchical",
            fusion=FusionConfig(alpha=0.3),
            metadata={"experiment": "test"},
        )
        j = gc.model_dump_json(indent=2)
        gc2 = GateConfig.model_validate_json(j)
        assert gc2.n_subsystems == 6
        assert gc2.variant == ConditioningVariant.HIERARCHICAL
        assert gc2.fusion.alpha == 0.3
        assert gc2.metadata["experiment"] == "test"

    def test_from_dict(self):
        d = {"n_subsystems": 2, "shots": 512, "variant": "global"}
        gc = GateConfig.model_validate(d)
        assert gc.n_subsystems == 2
        assert gc.variant == ConditioningVariant.GLOBAL

    def test_nested_sub_configs(self):
        gc = GateConfig(
            fusion=FusionConfig(alpha=0.8),
            dynamic_threshold=DynamicThresholdConfig(enabled=True, z_factor=2.0),
            probe=ProbeConfig(enabled=True, probe_shots=50),
        )
        assert gc.fusion.alpha == 0.8
        assert gc.dynamic_threshold.enabled is True
        assert gc.probe.probe_shots == 50


# ---------------------------------------------------------------------------
# Enum values
# ---------------------------------------------------------------------------


class TestEnums:
    def test_conditioning_variants(self):
        assert ConditioningVariant.GLOBAL.value == "global"
        assert ConditioningVariant.HIERARCHICAL.value == "hierarchical"
        assert ConditioningVariant.SCORE_FUSION.value == "score_fusion"

    def test_adapter_kinds(self):
        assert AdapterKind.QISKIT.value == "qiskit"
        assert AdapterKind.CIRQ.value == "cirq"
        assert AdapterKind.PENNYLANE.value == "pennylane"
        assert AdapterKind.MOCK.value == "mock"
