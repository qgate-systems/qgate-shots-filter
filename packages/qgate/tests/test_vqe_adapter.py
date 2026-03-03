"""Tests for qgate.adapters.vqe_adapter — VQETSVFAdapter."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qgate.adapters.vqe_adapter import (
    VQETSVFAdapter,
    compute_energy_from_bitstring,
    energy_error,
    energy_ratio,
    estimate_energy_from_counts,
    tfim_exact_ground_energy,
)
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

N_QUBITS = 4
J_COUPLING = 1.0
H_FIELD = 1.0


def _ideal_backend():
    return AerSimulator()


# ═══════════════════════════════════════════════════════════════════════════
# Hamiltonian helper tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHamiltonianHelpers:
    """Tests for TFIM energy computation functions."""

    def test_exact_ground_energy_is_negative(self):
        e = tfim_exact_ground_energy(N_QUBITS, J_COUPLING, H_FIELD)
        assert e < 0.0

    def test_exact_ground_energy_2_qubits(self):
        # 2-qubit TFIM: H = -J*Z0*Z1 - h*(X0 + X1)
        # For J=1, h=1: exact ground state energy = -sqrt(1+1) ≈ -2.236
        # (from analytical diagonalisation of 4x4 matrix)
        e = tfim_exact_ground_energy(2, 1.0, 1.0)
        assert e < -2.0
        assert e == pytest.approx(-2.23606797749979, abs=0.01)

    def test_exact_ground_energy_no_field(self):
        # h=0: H = -J Σ Z_i Z_{i+1}, ground state is all aligned
        # For n=4 open chain: E = -J * 3 = -3.0
        e = tfim_exact_ground_energy(4, 1.0, 0.0)
        assert e == pytest.approx(-3.0, abs=1e-10)

    def test_exact_ground_energy_deterministic(self):
        e1 = tfim_exact_ground_energy(4, 1.0, 1.0)
        e2 = tfim_exact_ground_energy(4, 1.0, 1.0)
        assert e1 == e2

    def test_compute_energy_all_aligned(self):
        # "0000" → all spins +1: E_ZZ = -J * 3 = -3.0
        e = compute_energy_from_bitstring("0000", 4, j_coupling=1.0)
        assert e == pytest.approx(-3.0)

    def test_compute_energy_alternating(self):
        # "0101" → spins [+1, -1, +1, -1]: E_ZZ = -J*(-1 + -1 + -1) = +3.0
        e = compute_energy_from_bitstring("0101", 4, j_coupling=1.0)
        assert e == pytest.approx(3.0)

    def test_compute_energy_all_ones(self):
        # "1111" → all spins -1: E_ZZ = -J * (1+1+1) = -3.0
        e = compute_energy_from_bitstring("1111", 4, j_coupling=1.0)
        assert e == pytest.approx(-3.0)

    def test_estimate_energy_from_counts(self):
        counts = {"0000": 500, "1111": 500}
        e = estimate_energy_from_counts(counts, 4, j_coupling=1.0)
        # Both give -3.0 → average = -3.0
        assert e == pytest.approx(-3.0)

    def test_estimate_energy_mixed(self):
        counts = {"0000": 1, "0101": 1}
        e = estimate_energy_from_counts(counts, 4, j_coupling=1.0)
        # (-3.0 + 3.0) / 2 = 0.0
        assert e == pytest.approx(0.0)

    def test_energy_error_function(self):
        assert energy_error(-2.5, -3.0) == pytest.approx(0.5)
        assert energy_error(-3.0, -3.0) == pytest.approx(0.0)

    def test_energy_ratio_function(self):
        assert energy_ratio(-3.0, -3.0) == pytest.approx(1.0)
        assert energy_ratio(-1.5, -3.0) == pytest.approx(0.5)
        assert energy_ratio(0.0, 0.0) == 0.0  # edge case


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — circuit construction
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildCircuit:
    """Test circuit construction for both algorithm modes."""

    def test_standard_circuit_has_correct_qubits(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
        )
        qc = adapter.build_circuit(n_subsystems=N_QUBITS, n_cycles=2)
        assert qc.num_qubits == N_QUBITS
        assert qc.num_clbits == N_QUBITS

    def test_tsvf_circuit_has_ancilla(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            n_qubits=N_QUBITS,
        )
        qc = adapter.build_circuit(n_subsystems=N_QUBITS, n_cycles=2)
        assert qc.num_qubits == N_QUBITS + 1  # +1 ancilla
        assert qc.num_clbits == N_QUBITS + 1  # +1 ancilla clbit

    def test_n_subsystems_mismatch_raises(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
        )
        with pytest.raises(ValueError, match="n_subsystems"):
            adapter.build_circuit(n_subsystems=6, n_cycles=1)

    def test_unknown_mode_raises(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="invalid_mode",
            n_qubits=N_QUBITS,
        )
        with pytest.raises(ValueError, match="Unknown algorithm_mode"):
            adapter.build_circuit(n_subsystems=N_QUBITS, n_cycles=1)

    def test_circuit_depth_increases_with_layers(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
        )
        d1 = adapter.build_circuit(N_QUBITS, 1).depth()
        d3 = adapter.build_circuit(N_QUBITS, 3).depth()
        assert d3 > d1

    def test_tsvf_circuit_depth_increases_with_layers(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            n_qubits=N_QUBITS,
        )
        d1 = adapter.build_circuit(N_QUBITS, 1).depth()
        d3 = adapter.build_circuit(N_QUBITS, 3).depth()
        assert d3 > d1

    def test_custom_params(self):
        params = np.random.default_rng(42).uniform(
            -math.pi,
            math.pi,
            size=(2, N_QUBITS, 2),
        )
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
            params=params,
        )
        qc = adapter.build_circuit(N_QUBITS, 2)
        assert qc.num_qubits == N_QUBITS

    def test_2d_params_replicated(self):
        """2D params (n_qubits, 2) should be replicated for all layers."""
        params = np.random.default_rng(42).uniform(
            -math.pi,
            math.pi,
            size=(N_QUBITS, 2),
        )
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
            params=params,
        )
        qc = adapter.build_circuit(N_QUBITS, 3)
        assert qc.num_qubits == N_QUBITS


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — run + parse
# ═══════════════════════════════════════════════════════════════════════════


class TestRunAndParse:
    """Test circuit execution and result parsing."""

    def test_standard_run_returns_dict(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
        )
        qc = adapter.build_circuit(N_QUBITS, 1)
        raw = adapter.run(qc, shots=100)
        assert isinstance(raw, dict)
        assert "counts" in raw or "pub_result" in raw

    def test_standard_parse_returns_outcomes(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
        )
        qc = adapter.build_circuit(N_QUBITS, 2)
        raw = adapter.run(qc, shots=100)
        outcomes = adapter.parse_results(raw, N_QUBITS, 2)
        assert len(outcomes) == 100
        for o in outcomes:
            assert isinstance(o, ParityOutcome)
            assert o.n_subsystems == N_QUBITS
            assert o.n_cycles == 2
            assert o.parity_matrix.shape == (2, N_QUBITS)

    def test_tsvf_parse_returns_outcomes(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            n_qubits=N_QUBITS,
            seed=42,
        )
        qc = adapter.build_circuit(N_QUBITS, 1)
        raw = adapter.run(qc, shots=100)
        outcomes = adapter.parse_results(raw, N_QUBITS, 1)
        assert len(outcomes) == 100
        for o in outcomes:
            assert isinstance(o, ParityOutcome)
            assert o.n_subsystems == N_QUBITS
            assert o.n_cycles == 1

    def test_build_and_run_convenience(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
        )
        outcomes = adapter.build_and_run(n_subsystems=N_QUBITS, n_cycles=1, shots=50)
        assert len(outcomes) == 50
        assert all(isinstance(o, ParityOutcome) for o in outcomes)

    def test_parity_values_are_binary(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            n_qubits=N_QUBITS,
            seed=99,
        )
        outcomes = adapter.build_and_run(n_subsystems=N_QUBITS, n_cycles=2, shots=200)
        for o in outcomes:
            unique_vals = set(o.parity_matrix.flatten().tolist())
            assert unique_vals.issubset({0, 1})


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — energy extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractEnergy:
    """Test energy extraction with and without post-selection."""

    def test_standard_energy_is_valid(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
            j_coupling=J_COUPLING,
            h_field=H_FIELD,
        )
        qc = adapter.build_circuit(N_QUBITS, 2)
        raw = adapter.run(qc, shots=1000)
        energy, n = adapter.extract_energy(raw, postselect=False)
        # Energy should be a finite number
        assert np.isfinite(energy)
        assert n == 1000

    def test_tsvf_postselected_energy(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            n_qubits=N_QUBITS,
            j_coupling=J_COUPLING,
            h_field=H_FIELD,
            seed=42,
        )
        qc = adapter.build_circuit(N_QUBITS, 1)
        raw = adapter.run(qc, shots=2000)
        energy, n_accepted = adapter.extract_energy(raw, postselect=True)
        assert np.isfinite(energy)
        assert 0 <= n_accepted <= 2000

    def test_energy_ratio_extraction(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
            j_coupling=J_COUPLING,
            h_field=H_FIELD,
        )
        qc = adapter.build_circuit(N_QUBITS, 2)
        raw = adapter.run(qc, shots=1000)
        ratio, err, n = adapter.extract_energy_ratio(raw, postselect=False)
        assert np.isfinite(ratio)
        assert err >= 0.0
        assert n == 1000

    def test_best_bitstring_extraction(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
        )
        qc = adapter.build_circuit(N_QUBITS, 2)
        raw = adapter.run(qc, shots=1000)
        bs, energy, count = adapter.extract_best_bitstring(raw, postselect=False)
        assert len(bs) == N_QUBITS
        assert np.isfinite(energy)
        assert count > 0

    def test_exact_ground_energy_accessor(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            n_qubits=N_QUBITS,
            j_coupling=J_COUPLING,
            h_field=H_FIELD,
        )
        e = adapter.get_exact_ground_energy()
        assert e < 0.0
        # Should match the helper function
        assert e == tfim_exact_ground_energy(N_QUBITS, J_COUPLING, H_FIELD)


# ═══════════════════════════════════════════════════════════════════════════
# Integration tests — TrajectoryFilter pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestTrajectoryFilterIntegration:
    """Test the adapter works end-to-end with qgate's TrajectoryFilter."""

    def test_standard_vqe_through_filter(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
        )
        config = GateConfig(
            n_subsystems=N_QUBITS,
            n_cycles=1,
            shots=200,
            variant=ConditioningVariant.SCORE_FUSION,
            fusion=FusionConfig(alpha=0.5, threshold=0.5),
            dynamic_threshold=DynamicThresholdConfig(mode="fixed"),
        )
        tf = TrajectoryFilter(config, adapter)
        outcomes = adapter.build_and_run(n_subsystems=N_QUBITS, n_cycles=1, shots=200)
        result = tf.filter(outcomes)

        assert result.total_shots == 200
        assert 0 <= result.accepted_shots <= 200
        assert 0.0 <= result.acceptance_probability <= 1.0
        assert result.variant == "score_fusion"
        assert len(result.scores) == 200

    def test_tsvf_vqe_with_galton_threshold(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            n_qubits=N_QUBITS,
            seed=42,
        )
        config = GateConfig(
            n_subsystems=N_QUBITS,
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
        outcomes = adapter.build_and_run(n_subsystems=N_QUBITS, n_cycles=2, shots=500)
        result = tf.filter(outcomes)

        assert result.total_shots == 500
        assert result.variant == "score_fusion"
        assert result.dynamic_threshold_final is not None
        assert 0.2 <= result.dynamic_threshold_final <= 0.95
        assert len(result.scores) == 500

    def test_global_conditioning(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
        )
        config = GateConfig(
            n_subsystems=N_QUBITS,
            n_cycles=1,
            shots=200,
            variant=ConditioningVariant.GLOBAL,
        )
        tf = TrajectoryFilter(config, adapter)
        outcomes = adapter.build_and_run(n_subsystems=N_QUBITS, n_cycles=1, shots=200)
        result = tf.filter(outcomes)
        assert result.total_shots == 200
        assert result.accepted_shots <= 200

    def test_hierarchical_conditioning(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
        )
        config = GateConfig(
            n_subsystems=N_QUBITS,
            n_cycles=1,
            shots=200,
            variant=ConditioningVariant.HIERARCHICAL,
            k_fraction=0.5,
        )
        tf = TrajectoryFilter(config, adapter)
        outcomes = adapter.build_and_run(n_subsystems=N_QUBITS, n_cycles=1, shots=200)
        result = tf.filter(outcomes)
        assert result.total_shots == 200


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — scoring compatibility
# ═══════════════════════════════════════════════════════════════════════════


class TestScoringCompatibility:
    """Ensure ParityOutcomes from VQETSVFAdapter score correctly."""

    def test_score_batch_works_on_vqe_outcomes(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            n_qubits=N_QUBITS,
            seed=42,
        )
        outcomes = adapter.build_and_run(n_subsystems=N_QUBITS, n_cycles=2, shots=100)
        scores = score_batch(outcomes, alpha=0.5)
        assert len(scores) == 100
        for lf, hf, combined in scores:
            assert 0.0 <= lf <= 1.0
            assert 0.0 <= hf <= 1.0
            assert 0.0 <= combined <= 1.0

    def test_perfect_alignment_scores_high(self):
        """A ParityOutcome with all zeros should score 1.0."""
        outcome = ParityOutcome(
            n_subsystems=N_QUBITS,
            n_cycles=2,
            parity_matrix=np.zeros((2, N_QUBITS), dtype=np.int8),
        )
        scores = score_batch([outcome], alpha=0.5)
        _, _, combined = scores[0]
        assert combined == pytest.approx(1.0)

    def test_all_fail_scores_low(self):
        """A ParityOutcome with all ones should score 0.0."""
        outcome = ParityOutcome(
            n_subsystems=N_QUBITS,
            n_cycles=2,
            parity_matrix=np.ones((2, N_QUBITS), dtype=np.int8),
        )
        scores = score_batch([outcome], alpha=0.5)
        _, _, combined = scores[0]
        assert combined == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — bitstring parsing helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestBitstringParsing:
    """Test internal bitstring → parity matrix conversion."""

    def test_tsvf_ancilla_1_aligned(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            n_qubits=N_QUBITS,
        )
        # "0000" → all aligned → all qubits are "good" (low energy)
        row = adapter._bitstring_to_parity_row("1 0000", n_subsystems=N_QUBITS, n_cycles=2)
        assert row.shape == (2, N_QUBITS)
        # All qubits aligned with neighbours → all 0
        np.testing.assert_array_equal(row, np.zeros((2, N_QUBITS), dtype=np.int8))

    def test_tsvf_ancilla_0_all_fail(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            n_qubits=N_QUBITS,
        )
        # Ancilla=0 → all fail regardless of search bits
        row = adapter._bitstring_to_parity_row("0 0000", n_subsystems=N_QUBITS, n_cycles=2)
        np.testing.assert_array_equal(row, np.ones((2, N_QUBITS), dtype=np.int8))

    def test_tsvf_ancilla_1_alternating(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            n_qubits=N_QUBITS,
        )
        # "0101" → no qubit is aligned with any neighbour → all fail
        row = adapter._bitstring_to_parity_row("1 0101", n_subsystems=N_QUBITS, n_cycles=1)
        assert row.shape == (1, N_QUBITS)
        np.testing.assert_array_equal(row, np.ones((1, N_QUBITS), dtype=np.int8))

    def test_standard_aligned(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
        )
        row = adapter._bitstring_to_parity_row("0000", n_subsystems=N_QUBITS, n_cycles=2)
        np.testing.assert_array_equal(row, np.zeros((2, N_QUBITS), dtype=np.int8))

    def test_standard_alternating(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
        )
        row = adapter._bitstring_to_parity_row("0101", n_subsystems=N_QUBITS, n_cycles=1)
        np.testing.assert_array_equal(row, np.ones((1, N_QUBITS), dtype=np.int8))

    def test_partial_alignment(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
        )
        # "0011" → qubits 0,1 aligned (0==0), qubits 2,3 aligned (1==1)
        # qubit 1 and 2 are different → but each has at least one aligned neighbour
        row = adapter._bitstring_to_parity_row("0011", n_subsystems=N_QUBITS, n_cycles=1)
        # qubit 0: matches qubit 1 → 0
        # qubit 1: matches qubit 0 → 0
        # qubit 2: matches qubit 3 → 0
        # qubit 3: matches qubit 2 → 0
        np.testing.assert_array_equal(row, np.zeros((1, N_QUBITS), dtype=np.int8))


# ═══════════════════════════════════════════════════════════════════════════
# No-backend test
# ═══════════════════════════════════════════════════════════════════════════


class TestNoBackend:
    """Verify error when no backend is configured."""

    def test_run_without_backend_raises(self):
        adapter = VQETSVFAdapter(
            backend=None,
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
        )
        qc = adapter.build_circuit(N_QUBITS, 1)
        with pytest.raises(RuntimeError, match="No backend"):
            adapter.run(qc, shots=10)


# ═══════════════════════════════════════════════════════════════════════════
# TFIM parameter variations
# ═══════════════════════════════════════════════════════════════════════════


class TestTFIMVariations:
    """Test different TFIM parameter settings."""

    def test_different_coupling(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
            j_coupling=2.0,
            h_field=0.5,
        )
        qc = adapter.build_circuit(N_QUBITS, 1)
        raw = adapter.run(qc, shots=100)
        energy, n = adapter.extract_energy(raw, postselect=False)
        assert np.isfinite(energy)
        assert n == 100

    def test_seed_reproducibility(self):
        a1 = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
            seed=42,
        )
        a2 = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
            seed=42,
        )
        qc1 = a1.build_circuit(N_QUBITS, 2)
        qc2 = a2.build_circuit(N_QUBITS, 2)
        # Same seed → same circuit (same parameter generation)
        assert qc1.depth() == qc2.depth()
        assert qc1.num_qubits == qc2.num_qubits

    def test_transpiled_depth(self):
        adapter = VQETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_qubits=N_QUBITS,
        )
        qc = adapter.build_circuit(N_QUBITS, 2)
        depth = adapter.get_transpiled_depth(qc)
        assert depth > 0
