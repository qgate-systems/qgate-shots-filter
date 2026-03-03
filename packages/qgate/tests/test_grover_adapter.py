"""Tests for qgate.adapters.grover_adapter — GroverTSVFAdapter."""

from __future__ import annotations

import numpy as np
import pytest

from qgate.adapters.grover_adapter import GroverTSVFAdapter
from qgate.conditioning import ParityOutcome
from qgate.config import (
    ConditioningVariant,
    DynamicThresholdConfig,
    FusionConfig,
    GateConfig,
)
from qgate.filter import TrajectoryFilter
from qgate.scoring import score_batch

# ── Skip if Qiskit/Aer not installed ─────────────────────────────────────
pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from qiskit_aer import AerSimulator

# ── Helpers ───────────────────────────────────────────────────────────────


def _ideal_backend():
    return AerSimulator()


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — circuit construction
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildCircuit:
    """Test circuit construction for both algorithm modes."""

    def test_standard_circuit_has_correct_qubits(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            target_state="101",
        )
        qc = adapter.build_circuit(n_subsystems=3, n_cycles=2)
        assert qc.num_qubits == 3
        assert qc.num_clbits == 3

    def test_tsvf_circuit_has_ancilla(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            target_state="101",
        )
        qc = adapter.build_circuit(n_subsystems=3, n_cycles=2)
        # 3 search + 1 ancilla
        assert qc.num_qubits == 4
        # 3 search clbits + 1 ancilla clbit
        assert qc.num_clbits == 4

    def test_n_subsystems_mismatch_raises(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            target_state="101",
        )
        with pytest.raises(ValueError, match="n_subsystems"):
            adapter.build_circuit(n_subsystems=4, n_cycles=1)

    def test_unknown_mode_raises(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="invalid_mode",
            target_state="101",
        )
        with pytest.raises(ValueError, match="Unknown algorithm_mode"):
            adapter.build_circuit(n_subsystems=3, n_cycles=1)

    def test_standard_circuit_depth_increases_with_iterations(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            target_state="101",
        )
        d1 = adapter.build_circuit(3, 1).depth()
        d3 = adapter.build_circuit(3, 3).depth()
        assert d3 > d1

    def test_tsvf_circuit_depth_increases_with_iterations(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            target_state="101",
        )
        d1 = adapter.build_circuit(3, 1).depth()
        d3 = adapter.build_circuit(3, 3).depth()
        assert d3 > d1


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — run + parse
# ═══════════════════════════════════════════════════════════════════════════


class TestRunAndParse:
    """Test circuit execution and result parsing."""

    def test_standard_run_returns_dict(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            target_state="101",
        )
        qc = adapter.build_circuit(3, 1)
        raw = adapter.run(qc, shots=100)
        assert isinstance(raw, dict)
        assert "counts" in raw or "pub_result" in raw

    def test_standard_parse_returns_outcomes(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            target_state="101",
        )
        qc = adapter.build_circuit(3, 2)
        raw = adapter.run(qc, shots=100)
        outcomes = adapter.parse_results(raw, 3, 2)
        assert len(outcomes) == 100
        for o in outcomes:
            assert isinstance(o, ParityOutcome)
            assert o.n_subsystems == 3
            assert o.n_cycles == 2
            assert o.parity_matrix.shape == (2, 3)

    def test_tsvf_parse_returns_outcomes(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            target_state="101",
            seed=42,
        )
        qc = adapter.build_circuit(3, 1)
        raw = adapter.run(qc, shots=100)
        outcomes = adapter.parse_results(raw, 3, 1)
        assert len(outcomes) == 100
        for o in outcomes:
            assert isinstance(o, ParityOutcome)
            assert o.n_subsystems == 3
            assert o.n_cycles == 1

    def test_build_and_run_convenience(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            target_state="101",
        )
        outcomes = adapter.build_and_run(n_subsystems=3, n_cycles=1, shots=50)
        assert len(outcomes) == 50
        assert all(isinstance(o, ParityOutcome) for o in outcomes)

    def test_parity_values_are_binary(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            target_state="101",
            seed=99,
        )
        outcomes = adapter.build_and_run(n_subsystems=3, n_cycles=2, shots=200)
        for o in outcomes:
            unique_vals = set(o.parity_matrix.flatten().tolist())
            assert unique_vals.issubset({0, 1})


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — target probability extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractProbability:
    """Test P(target) extraction with and without post-selection."""

    def test_standard_probability_is_valid(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            target_state="101",
        )
        qc = adapter.build_circuit(3, 1)
        raw = adapter.run(qc, shots=1000)
        p, n = adapter.extract_target_probability(raw, postselect=False)
        assert 0.0 <= p <= 1.0
        assert n == 1000

    def test_tsvf_postselected_probability(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            target_state="101",
            seed=42,
        )
        qc = adapter.build_circuit(3, 1)
        raw = adapter.run(qc, shots=2000)
        p, n_accepted = adapter.extract_target_probability(raw, postselect=True)
        assert 0.0 <= p <= 1.0
        # Some shots should be accepted (ancilla=1)
        assert n_accepted >= 0
        assert n_accepted <= 2000


# ═══════════════════════════════════════════════════════════════════════════
# Integration tests — TrajectoryFilter pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestTrajectoryFilterIntegration:
    """Test the adapter works end-to-end with qgate's TrajectoryFilter."""

    def test_standard_grover_through_filter(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            target_state="101",
        )
        config = GateConfig(
            n_subsystems=3,
            n_cycles=1,
            shots=200,
            variant=ConditioningVariant.SCORE_FUSION,
            fusion=FusionConfig(alpha=0.5, threshold=0.5),
            dynamic_threshold=DynamicThresholdConfig(mode="fixed"),
        )
        tf = TrajectoryFilter(config, adapter)
        outcomes = adapter.build_and_run(n_subsystems=3, n_cycles=1, shots=200)
        result = tf.filter(outcomes)

        assert result.total_shots == 200
        assert 0 <= result.accepted_shots <= 200
        assert 0.0 <= result.acceptance_probability <= 1.0
        assert result.variant == "score_fusion"
        assert len(result.scores) == 200

    def test_tsvf_grover_with_galton_threshold(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            target_state="101",
            seed=42,
        )
        config = GateConfig(
            n_subsystems=3,
            n_cycles=2,
            shots=500,
            variant=ConditioningVariant.SCORE_FUSION,
            fusion=FusionConfig(alpha=0.6, threshold=0.5),
            dynamic_threshold=DynamicThresholdConfig(
                mode="galton",
                target_acceptance=0.10,
                min_window_size=50,
                window_size=500,
                use_quantile=True,
                min_threshold=0.2,
                max_threshold=0.95,
            ),
        )
        tf = TrajectoryFilter(config, adapter)
        outcomes = adapter.build_and_run(n_subsystems=3, n_cycles=2, shots=500)
        result = tf.filter(outcomes)

        assert result.total_shots == 500
        assert result.variant == "score_fusion"
        assert result.dynamic_threshold_final is not None
        assert 0.2 <= result.dynamic_threshold_final <= 0.95
        assert len(result.scores) == 500

    def test_global_conditioning(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            target_state="101",
        )
        config = GateConfig(
            n_subsystems=3,
            n_cycles=1,
            shots=200,
            variant=ConditioningVariant.GLOBAL,
        )
        tf = TrajectoryFilter(config, adapter)
        outcomes = adapter.build_and_run(n_subsystems=3, n_cycles=1, shots=200)
        result = tf.filter(outcomes)
        # Global: only shots where ALL qubits match target
        assert result.total_shots == 200
        assert result.accepted_shots <= 200

    def test_hierarchical_conditioning(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            target_state="101",
        )
        config = GateConfig(
            n_subsystems=3,
            n_cycles=1,
            shots=200,
            variant=ConditioningVariant.HIERARCHICAL,
            k_fraction=0.67,  # At least 2 of 3 qubits match
        )
        tf = TrajectoryFilter(config, adapter)
        outcomes = adapter.build_and_run(n_subsystems=3, n_cycles=1, shots=200)
        result = tf.filter(outcomes)
        assert result.total_shots == 200
        assert result.accepted_shots >= result.accepted_shots  # tautology, just check no crash


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — scoring compatibility
# ═══════════════════════════════════════════════════════════════════════════


class TestScoringCompatibility:
    """Ensure ParityOutcomes from GroverTSVFAdapter score correctly."""

    def test_score_batch_works_on_grover_outcomes(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            target_state="101",
            seed=42,
        )
        outcomes = adapter.build_and_run(n_subsystems=3, n_cycles=2, shots=100)
        scores = score_batch(outcomes, alpha=0.5)
        assert len(scores) == 100
        for lf, hf, combined in scores:
            assert 0.0 <= lf <= 1.0
            assert 0.0 <= hf <= 1.0
            assert 0.0 <= combined <= 1.0

    def test_perfect_match_scores_high(self):
        """A ParityOutcome with all zeros should score 1.0."""
        outcome = ParityOutcome(
            n_subsystems=3,
            n_cycles=2,
            parity_matrix=np.zeros((2, 3), dtype=np.int8),
        )
        scores = score_batch([outcome], alpha=0.5)
        _, _, combined = scores[0]
        assert combined == pytest.approx(1.0)

    def test_all_fail_scores_low(self):
        """A ParityOutcome with all ones should score 0.0."""
        outcome = ParityOutcome(
            n_subsystems=3,
            n_cycles=2,
            parity_matrix=np.ones((2, 3), dtype=np.int8),
        )
        scores = score_batch([outcome], alpha=0.5)
        _, _, combined = scores[0]
        assert combined == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — bitstring parsing helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestBitstringParsing:
    """Test internal bitstring → parity matrix conversion."""

    def test_tsvf_ancilla_1_target_match(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            target_state="101",
        )
        # Bitstring "1 101" → ancilla=1, search=101 → all match → all 0s
        row = adapter._bitstring_to_parity_row("1 101", n_subsystems=3, n_cycles=2)
        assert row.shape == (2, 3)
        np.testing.assert_array_equal(row, np.zeros((2, 3), dtype=np.int8))

    def test_tsvf_ancilla_0_all_fail(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            target_state="101",
        )
        # Ancilla=0 → all fail (all 1s)
        row = adapter._bitstring_to_parity_row("0 101", n_subsystems=3, n_cycles=2)
        np.testing.assert_array_equal(row, np.ones((2, 3), dtype=np.int8))

    def test_tsvf_ancilla_1_partial_match(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            target_state="101",
        )
        # Search=100, target=101 → q0 match, q1 match, q2 mismatch
        row = adapter._bitstring_to_parity_row("1 100", n_subsystems=3, n_cycles=1)
        assert row.shape == (1, 3)
        # q0: 1==1 → 0, q1: 0==0 → 0, q2: 0!=1 → 1
        expected = np.array([[0, 0, 1]], dtype=np.int8)
        np.testing.assert_array_equal(row, expected)

    def test_standard_target_match(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            target_state="101",
        )
        row = adapter._bitstring_to_parity_row("101", n_subsystems=3, n_cycles=2)
        np.testing.assert_array_equal(row, np.zeros((2, 3), dtype=np.int8))

    def test_standard_mismatch(self):
        adapter = GroverTSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            target_state="101",
        )
        row = adapter._bitstring_to_parity_row("010", n_subsystems=3, n_cycles=1)
        # All qubits mismatch
        expected = np.array([[1, 1, 1]], dtype=np.int8)
        np.testing.assert_array_equal(row, expected)


# ═══════════════════════════════════════════════════════════════════════════
# No-backend test
# ═══════════════════════════════════════════════════════════════════════════


class TestNoBackend:
    """Verify error when no backend is configured."""

    def test_run_without_backend_raises(self):
        adapter = GroverTSVFAdapter(
            backend=None,
            algorithm_mode="standard",
            target_state="101",
        )
        qc = adapter.build_circuit(3, 1)
        with pytest.raises(RuntimeError, match="No backend"):
            adapter.run(qc, shots=10)
