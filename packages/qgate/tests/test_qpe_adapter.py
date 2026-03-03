"""Tests for qgate.adapters.qpe_adapter — QPETSVFAdapter."""

from __future__ import annotations

import numpy as np
import pytest

from qgate.adapters.qpe_adapter import (
    QPETSVFAdapter,
    binary_fraction_to_phase,
    histogram_entropy,
    mean_phase_error,
    phase_error,
    phase_fidelity,
    phase_to_binary_fraction,
)
from qgate.conditioning import ParityOutcome
from qgate.config import (
    ConditioningVariant,
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
# Unit tests — helper functions
# ═══════════════════════════════════════════════════════════════════════════


class TestPhaseHelpers:
    """Test standalone phase utility functions."""

    def test_phase_to_binary_fraction_exact(self):
        # φ = 0.5 → "1" for 1 bit, "10" for 2 bits
        assert phase_to_binary_fraction(0.5, 1) == "1"
        assert phase_to_binary_fraction(0.5, 2) == "10"
        assert phase_to_binary_fraction(0.5, 3) == "100"

    def test_phase_to_binary_fraction_quarter(self):
        # φ = 0.25 → "01" for 2 bits, "010" for 3 bits
        assert phase_to_binary_fraction(0.25, 2) == "01"
        assert phase_to_binary_fraction(0.25, 3) == "010"

    def test_phase_to_binary_fraction_three_eighths(self):
        # φ = 0.375 = 3/8 → "011" for 3 bits
        assert phase_to_binary_fraction(0.375, 3) == "011"

    def test_phase_to_binary_fraction_zero(self):
        assert phase_to_binary_fraction(0.0, 3) == "000"

    def test_phase_to_binary_fraction_irrational(self):
        # φ = 1/3 → nearest 3-bit approximation
        # 1/3 * 8 = 2.666... → round to 3 → "011"
        result = phase_to_binary_fraction(1.0 / 3.0, 3)
        assert len(result) == 3
        assert all(c in "01" for c in result)

    def test_binary_fraction_to_phase_round_trip(self):
        for phi in [0.0, 0.25, 0.375, 0.5, 0.75]:
            bits = phase_to_binary_fraction(phi, 4)
            recovered = binary_fraction_to_phase(bits)
            assert abs(recovered - phi) < 1e-10

    def test_binary_fraction_to_phase_values(self):
        assert binary_fraction_to_phase("100") == 0.5
        assert binary_fraction_to_phase("010") == 0.25
        assert binary_fraction_to_phase("011") == 0.375
        assert binary_fraction_to_phase("000") == 0.0

    def test_phase_error_basic(self):
        assert abs(phase_error(0.3, 0.3)) < 1e-10
        assert abs(phase_error(0.5, 0.0) - 0.5) < 1e-10

    def test_phase_error_wraparound(self):
        # |0.9 - 0.1| should wrap to 0.2
        assert abs(phase_error(0.9, 0.1) - 0.2) < 1e-10
        assert abs(phase_error(0.05, 0.95) - 0.1) < 1e-10

    def test_histogram_entropy_delta(self):
        # Perfect: all shots on one outcome
        counts = {"010": 1000}
        assert abs(histogram_entropy(counts)) < 1e-10

    def test_histogram_entropy_uniform(self):
        # Uniform over 8 outcomes → entropy = 3.0 bits
        counts = {format(i, "03b"): 1000 for i in range(8)}
        assert abs(histogram_entropy(counts) - 3.0) < 1e-6

    def test_histogram_entropy_empty(self):
        assert abs(histogram_entropy({})) < 1e-10

    def test_phase_fidelity_perfect(self):
        counts = {"011": 500, "010": 300, "100": 200}
        assert abs(phase_fidelity(counts, "011") - 0.5) < 1e-10

    def test_phase_fidelity_zero(self):
        counts = {"011": 500, "010": 300}
        assert abs(phase_fidelity(counts, "111")) < 1e-10

    def test_mean_phase_error_perfect(self):
        # All shots on the correct phase
        counts = {"010": 1000}
        err = mean_phase_error(counts, 0.25, 3)
        assert err < 1e-10

    def test_mean_phase_error_half_wrong(self):
        # Half correct, half maximally wrong
        counts = {"010": 500, "110": 500}
        # 010 → 0.25, correct
        # 110 → 0.75, error = 0.5
        err = mean_phase_error(counts, 0.25, 3)
        assert abs(err - 0.25) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — circuit construction
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildCircuit:
    """Test circuit construction for both algorithm modes."""

    def test_standard_circuit_has_correct_qubits(self):
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=0.25,
        )
        qc = adapter.build_circuit(n_subsystems=3, n_cycles=1)
        # 3 precision + 1 eigenstate = 4 qubits
        assert qc.num_qubits == 4
        # 3 classical bits for phase
        assert qc.num_clbits == 3

    def test_tsvf_circuit_has_ancilla(self):
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            eigenphase=0.25,
        )
        qc = adapter.build_circuit(n_subsystems=3, n_cycles=1)
        # 3 precision + 1 eigenstate + 1 ancilla = 5 qubits
        assert qc.num_qubits == 5
        # 3 phase clbits + 1 ancilla clbit = 4
        assert qc.num_clbits == 4

    def test_unknown_mode_raises(self):
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="invalid_mode",
            eigenphase=0.25,
        )
        with pytest.raises(ValueError, match="Unknown algorithm_mode"):
            adapter.build_circuit(n_subsystems=3, n_cycles=1)

    def test_standard_depth_increases_with_precision(self):
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=0.25,
        )
        d3 = adapter.build_circuit(3, 1).depth()
        d5 = adapter.build_circuit(5, 1).depth()
        assert d5 > d3

    def test_tsvf_depth_increases_with_precision(self):
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            eigenphase=0.25,
        )
        d3 = adapter.build_circuit(3, 1).depth()
        d5 = adapter.build_circuit(5, 1).depth()
        assert d5 > d3

    def test_tsvf_deeper_than_standard(self):
        """TSVF circuit should be deeper due to chaotic ansatz + probe."""
        std_adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=0.25,
        )
        tsvf_adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            eigenphase=0.25,
        )
        d_std = std_adapter.build_circuit(4, 1).depth()
        d_tsvf = tsvf_adapter.build_circuit(4, 1).depth()
        assert d_tsvf > d_std

    def test_eigenstate_qubit_prepared(self):
        """The eigenstate register should have an X gate (|1⟩ prep)."""
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=0.25,
        )
        qc = adapter.build_circuit(3, 1)
        # Count X gates — should have at least 1 for eigenstate prep
        x_count = sum(1 for inst in qc.data if inst.operation.name == "x")
        assert x_count >= 1

    def test_circuit_has_h_gates(self):
        """Precision qubits should have Hadamard gates."""
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=0.25,
        )
        qc = adapter.build_circuit(4, 1)
        h_count = sum(1 for inst in qc.data if inst.operation.name == "h")
        # At least 4 for initial superposition + some from iQFT
        assert h_count >= 4

    def test_circuit_has_cp_gates(self):
        """Circuit should contain controlled-phase gates."""
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=0.25,
        )
        qc = adapter.build_circuit(3, 1)
        cp_count = sum(1 for inst in qc.data if inst.operation.name == "cp")
        # 3 from controlled-U^{2^k} + some from iQFT
        assert cp_count >= 3


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — run + parse
# ═══════════════════════════════════════════════════════════════════════════


class TestRunAndParse:
    """Test circuit execution and result parsing."""

    def test_standard_run_returns_dict(self):
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=0.25,
        )
        qc = adapter.build_circuit(3, 1)
        raw = adapter.run(qc, shots=100)
        assert isinstance(raw, dict)
        assert "counts" in raw or "pub_result" in raw

    def test_tsvf_run_returns_dict(self):
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            eigenphase=0.25,
        )
        qc = adapter.build_circuit(3, 1)
        raw = adapter.run(qc, shots=100)
        assert isinstance(raw, dict)

    def test_standard_parse_returns_parity_outcomes(self):
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=0.25,
        )
        qc = adapter.build_circuit(3, 1)
        raw = adapter.run(qc, shots=200)
        outcomes = adapter.parse_results(raw, 3, 1)
        assert len(outcomes) == 200
        assert all(isinstance(o, ParityOutcome) for o in outcomes)

    def test_tsvf_parse_returns_parity_outcomes(self):
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            eigenphase=0.25,
        )
        qc = adapter.build_circuit(3, 1)
        raw = adapter.run(qc, shots=200)
        outcomes = adapter.parse_results(raw, 3, 1)
        assert len(outcomes) == 200
        assert all(isinstance(o, ParityOutcome) for o in outcomes)

    def test_parity_matrix_shape(self):
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=0.25,
        )
        qc = adapter.build_circuit(4, 1)
        raw = adapter.run(qc, shots=50)
        outcomes = adapter.parse_results(raw, 4, 1)
        for o in outcomes:
            assert o.parity_matrix.shape == (1, 4)

    def test_parity_values_binary(self):
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=0.5,
        )
        qc = adapter.build_circuit(3, 1)
        raw = adapter.run(qc, shots=100)
        outcomes = adapter.parse_results(raw, 3, 1)
        for o in outcomes:
            assert set(np.unique(o.parity_matrix)).issubset({0, 1})

    def test_no_backend_raises(self):
        adapter = QPETSVFAdapter(
            backend=None,
            algorithm_mode="standard",
            eigenphase=0.25,
        )
        qc = adapter.build_circuit(3, 1)
        with pytest.raises(RuntimeError, match="No backend"):
            adapter.run(qc, shots=100)


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — phase estimation accuracy (ideal simulator)
# ═══════════════════════════════════════════════════════════════════════════


class TestPhaseAccuracy:
    """Verify QPE finds the correct phase on an ideal simulator."""

    @pytest.mark.parametrize("phi", [0.25, 0.5, 0.75, 0.125])
    def test_standard_exact_phase(self, phi):
        """Standard QPE should exactly resolve phases that are exact
        binary fractions."""
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=phi,
        )
        n_prec = 4
        qc = adapter.build_circuit(n_prec, 1)
        raw = adapter.run(qc, shots=1024)
        metrics = adapter.extract_phase_metrics(raw, n_prec, postselect=False)

        # Fidelity should be very high for exact fractions
        assert metrics["fidelity"] > 0.90
        assert metrics["mean_phase_error"] < 0.05

    def test_standard_irrational_phase(self):
        """For φ=1/3 (not exact binary), standard QPE should still
        find a close approximation with the peak on the nearest fraction."""
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=1.0 / 3.0,
        )
        n_prec = 5
        qc = adapter.build_circuit(n_prec, 1)
        raw = adapter.run(qc, shots=2048)
        _best_bs, best_phase, _best_count = adapter.extract_best_phase(
            raw,
            n_prec,
            postselect=False,
        )
        # Phase error should be small (≤ 1/2^5 = 0.03125)
        err = phase_error(best_phase, 1.0 / 3.0)
        assert err < 0.05

    def test_tsvf_finds_correct_phase(self):
        """TSVF-QPE should also find the correct phase (exact fraction)."""
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            eigenphase=0.25,
        )
        n_prec = 4
        qc = adapter.build_circuit(n_prec, 1)
        raw = adapter.run(qc, shots=4096)
        metrics = adapter.extract_phase_metrics(raw, n_prec, postselect=True)
        # Should still find the correct phase, though acceptance rate < 1
        assert metrics["total_shots"] > 0
        # The measured phase should be within 0.3 (chaotic ansatz perturbs)
        assert phase_error(metrics["measured_phase"], 0.25) < 0.30


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — phase metrics extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestPhaseMetrics:
    """Test extract_phase_metrics and extract_best_phase."""

    def test_metrics_keys(self):
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=0.5,
        )
        qc = adapter.build_circuit(3, 1)
        raw = adapter.run(qc, shots=100)
        metrics = adapter.extract_phase_metrics(raw, 3, postselect=False)
        expected_keys = {
            "fidelity",
            "mean_phase_error",
            "entropy",
            "measured_phase",
            "true_phase",
            "total_shots",
            "acceptance_rate",
        }
        assert set(metrics.keys()) == expected_keys

    def test_metrics_true_phase(self):
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=0.375,
        )
        qc = adapter.build_circuit(3, 1)
        raw = adapter.run(qc, shots=100)
        metrics = adapter.extract_phase_metrics(raw, 3, postselect=False)
        assert metrics["true_phase"] == 0.375

    def test_tsvf_acceptance_rate_less_than_one(self):
        """TSVF post-selection should discard some shots."""
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            eigenphase=0.25,
        )
        qc = adapter.build_circuit(3, 1)
        raw = adapter.run(qc, shots=2048)
        metrics = adapter.extract_phase_metrics(raw, 3, postselect=True)
        # Some shots should be discarded
        assert 0.0 < metrics["acceptance_rate"] < 1.0

    def test_standard_acceptance_rate_is_one(self):
        """Standard QPE has no post-selection → acceptance_rate = 1."""
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=0.5,
        )
        qc = adapter.build_circuit(3, 1)
        raw = adapter.run(qc, shots=100)
        metrics = adapter.extract_phase_metrics(raw, 3, postselect=False)
        assert metrics["acceptance_rate"] == 1.0

    def test_extract_best_phase_returns_tuple(self):
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=0.5,
        )
        qc = adapter.build_circuit(3, 1)
        raw = adapter.run(qc, shots=500)
        bs, phase_val, count = adapter.extract_best_phase(
            raw,
            3,
            postselect=False,
        )
        assert isinstance(bs, str)
        assert isinstance(phase_val, float)
        assert isinstance(count, int)
        assert count > 0

    def test_get_correct_phase_bits(self):
        adapter = QPETSVFAdapter(eigenphase=0.5)
        assert adapter.get_correct_phase_bits(3) == "100"
        adapter2 = QPETSVFAdapter(eigenphase=0.25)
        assert adapter2.get_correct_phase_bits(3) == "010"

    def test_get_transpiled_depth(self):
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=0.25,
        )
        qc = adapter.build_circuit(3, 1)
        depth = adapter.get_transpiled_depth(qc)
        assert isinstance(depth, int)
        assert depth > 0


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — TSVF post-selection
# ═══════════════════════════════════════════════════════════════════════════


class TestPostSelection:
    """Test the TSVF post-selection mechanics."""

    def test_postselect_with_synthetic_counts(self):
        """Verify _postselect_phase_counts with known input."""
        adapter = QPETSVFAdapter(
            algorithm_mode="tsvf",
            eigenphase=0.25,
        )
        # Simulate space-separated keys: "anc_bit phase_bits"
        counts = {
            "1 010": 300,  # accepted, correct phase
            "0 010": 200,  # rejected
            "1 110": 100,  # accepted, wrong phase
            "0 110": 400,  # rejected
        }
        phase_counts, total_orig, accepted = adapter._postselect_phase_counts(counts, 3)
        assert total_orig == 1000
        assert accepted == 400  # 300 + 100
        assert phase_counts["010"] == 300
        assert phase_counts["110"] == 100

    def test_postselect_with_concatenated_keys(self):
        """Verify with concatenated (non-spaced) bitstring keys."""
        adapter = QPETSVFAdapter(
            algorithm_mode="tsvf",
            eigenphase=0.25,
        )
        counts = {
            "1010": 300,  # anc=1, phase=010
            "0010": 200,  # anc=0
            "1110": 100,  # anc=1, phase=110
        }
        phase_counts, total_orig, accepted = adapter._postselect_phase_counts(counts, 3)
        assert total_orig == 600
        assert accepted == 400
        assert phase_counts.get("010", 0) == 300

    def test_split_ancilla_phase_space(self):
        adapter = QPETSVFAdapter(eigenphase=0.25)
        anc, phase = adapter._split_ancilla_phase("1 010", 3)
        assert anc == "1"
        assert phase == "010"

    def test_split_ancilla_phase_concat(self):
        adapter = QPETSVFAdapter(eigenphase=0.25)
        anc, phase = adapter._split_ancilla_phase("1010", 3)
        assert anc == "1"
        assert phase == "010"


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — parity matrix generation
# ═══════════════════════════════════════════════════════════════════════════


class TestParityMatrix:
    """Test _bitstring_to_parity_row and _compute_phase_match."""

    def test_compute_phase_match_all_correct(self):
        adapter = QPETSVFAdapter(eigenphase=0.25)
        match = adapter._compute_phase_match("010", 3, "010")
        np.testing.assert_array_equal(match, [0, 0, 0])

    def test_compute_phase_match_all_wrong(self):
        adapter = QPETSVFAdapter(eigenphase=0.25)
        match = adapter._compute_phase_match("101", 3, "010")
        np.testing.assert_array_equal(match, [1, 1, 1])

    def test_compute_phase_match_partial(self):
        adapter = QPETSVFAdapter(eigenphase=0.25)
        match = adapter._compute_phase_match("011", 3, "010")
        np.testing.assert_array_equal(match, [0, 0, 1])

    def test_parity_row_standard(self):
        adapter = QPETSVFAdapter(
            algorithm_mode="standard",
            eigenphase=0.25,
        )
        correct = "010"
        row = adapter._bitstring_to_parity_row("010", 3, 1, correct)
        assert row.shape == (1, 3)
        np.testing.assert_array_equal(row[0], [0, 0, 0])

    def test_parity_row_tsvf_accepted(self):
        adapter = QPETSVFAdapter(
            algorithm_mode="tsvf",
            eigenphase=0.25,
        )
        correct = "010"
        row = adapter._bitstring_to_parity_row("1 010", 3, 1, correct)
        assert row.shape == (1, 3)
        np.testing.assert_array_equal(row[0], [0, 0, 0])

    def test_parity_row_tsvf_rejected(self):
        adapter = QPETSVFAdapter(
            algorithm_mode="tsvf",
            eigenphase=0.25,
        )
        correct = "010"
        row = adapter._bitstring_to_parity_row("0 010", 3, 1, correct)
        assert row.shape == (1, 3)
        np.testing.assert_array_equal(row[0], [1, 1, 1])


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — integration with qgate pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestQGatePipelineIntegration:
    """Test that QPETSVFAdapter works with qgate's TrajectoryFilter."""

    def test_standard_with_trajectory_filter(self):
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=0.25,
        )
        config = GateConfig(
            n_subsystems=3,
            n_cycles=1,
            shots=200,
            variant=ConditioningVariant.GLOBAL,
        )
        tf = TrajectoryFilter(config, adapter)
        result = tf.run()
        assert result.acceptance_probability >= 0.0
        assert result.total_shots == 200

    def test_tsvf_with_trajectory_filter(self):
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            eigenphase=0.25,
        )
        config = GateConfig(
            n_subsystems=3,
            n_cycles=1,
            shots=200,
            variant=ConditioningVariant.GLOBAL,
        )
        tf = TrajectoryFilter(config, adapter)
        result = tf.run()
        assert result.acceptance_probability >= 0.0

    def test_score_batch_integration(self):
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=0.5,
        )
        qc = adapter.build_circuit(3, 1)
        raw = adapter.run(qc, shots=100)
        outcomes = adapter.parse_results(raw, 3, 1)
        scores = score_batch(outcomes)
        assert len(scores) == 100
        # score_batch returns (lf, hf, combined) tuples
        assert all(isinstance(s, tuple) and len(s) == 3 for s in scores)
        assert all(0.0 <= s[2] <= 1.0 for s in scores)


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — import / registration
# ═══════════════════════════════════════════════════════════════════════════


class TestImportAndRegistration:
    """Test that QPETSVFAdapter is properly registered."""

    def test_import_from_qgate(self):
        from qgate import QPETSVFAdapter as Cls

        assert Cls is QPETSVFAdapter

    def test_adapter_kind_has_qpe(self):
        from qgate.config import AdapterKind

        assert hasattr(AdapterKind, "QPE_TSVF")
        assert AdapterKind.QPE_TSVF.value == "qpe_tsvf"

    def test_listed_in_adapters(self):
        from qgate.adapters import list_adapters

        adapters = list_adapters()
        assert "qpe_tsvf" in adapters


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_precision_qubit(self):
        """QPE with just 1 precision qubit (binary: 0 or 1)."""
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=0.5,
        )
        qc = adapter.build_circuit(1, 1)
        raw = adapter.run(qc, shots=500)
        metrics = adapter.extract_phase_metrics(raw, 1, postselect=False)
        # φ=0.5 → should measure "1"
        assert metrics["fidelity"] > 0.90

    def test_phase_zero(self):
        """QPE for eigenphase = 0 → should measure all zeros."""
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=0.0,
        )
        qc = adapter.build_circuit(3, 1)
        raw = adapter.run(qc, shots=500)
        metrics = adapter.extract_phase_metrics(raw, 3, postselect=False)
        assert metrics["fidelity"] > 0.90

    def test_different_seeds_give_different_circuits(self):
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            eigenphase=0.25,
            seed=1,
        )
        qc1 = adapter.build_circuit(3, 1, seed_offset=0)
        qc2 = adapter.build_circuit(3, 1, seed_offset=1)
        # Different seed offsets → different chaotic ansatz
        assert qc1 != qc2

    def test_empty_counts_handling(self):
        """Metrics should handle empty count dicts gracefully."""
        QPETSVFAdapter(eigenphase=0.25)
        phase_counts: dict[str, int] = {}
        assert phase_fidelity(phase_counts, "010") == 0.0
        assert histogram_entropy(phase_counts) == 0.0

    def test_parse_with_n_cycles_zero(self):
        """n_cycles=0 should be handled (effective=1)."""
        adapter = QPETSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            eigenphase=0.25,
        )
        qc = adapter.build_circuit(3, 0)
        raw = adapter.run(qc, shots=50)
        outcomes = adapter.parse_results(raw, 3, 0)
        assert len(outcomes) == 50
        for o in outcomes:
            assert o.parity_matrix.shape == (1, 3)
