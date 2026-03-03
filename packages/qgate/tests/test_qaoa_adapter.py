"""Tests for qgate.adapters.qaoa_adapter — QAOATSVFAdapter."""
from __future__ import annotations

import math

import numpy as np
import pytest

from qgate.adapters.qaoa_adapter import (
    QAOATSVFAdapter,
    best_maxcut,
    maxcut_value,
    random_regular_graph,
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

from qiskit_aer import AerSimulator  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────

# Small 4-node graph for tests:  0-1, 1-2, 2-3, 0-3  (a cycle)
SMALL_EDGES = [(0, 1), (1, 2), (2, 3), (0, 3)]
N_NODES = 4


def _ideal_backend():
    return AerSimulator()


# ═══════════════════════════════════════════════════════════════════════════
# Graph helper tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGraphHelpers:
    """Tests for the graph utility functions."""

    def test_random_regular_graph_node_count(self):
        edges = random_regular_graph(6, degree=3, seed=42)
        nodes = set()
        for a, b in edges:
            nodes.add(a)
            nodes.add(b)
        # All 6 nodes should be reachable (connected)
        assert len(nodes) == 6

    def test_random_regular_graph_edges_sorted(self):
        edges = random_regular_graph(5, degree=2, seed=0)
        for a, b in edges:
            assert a < b

    def test_random_regular_graph_deterministic(self):
        e1 = random_regular_graph(5, degree=3, seed=99)
        e2 = random_regular_graph(5, degree=3, seed=99)
        assert e1 == e2

    def test_maxcut_value_cycle_graph(self):
        # Cycle graph: 0-1, 1-2, 2-3, 0-3
        # Bitstring "0101" should cut all 4 edges
        assert maxcut_value("0101", SMALL_EDGES) == 4

    def test_maxcut_value_all_same(self):
        # All same partition — no edges cut
        assert maxcut_value("0000", SMALL_EDGES) == 0
        assert maxcut_value("1111", SMALL_EDGES) == 0

    def test_best_maxcut_cycle(self):
        bs, val = best_maxcut(4, SMALL_EDGES)
        assert val == 4  # Cycle graph best cut is all edges
        assert maxcut_value(bs, SMALL_EDGES) == 4


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — circuit construction
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildCircuit:
    """Test circuit construction for both algorithm modes."""

    def test_standard_circuit_has_correct_qubits(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
        )
        qc = adapter.build_circuit(n_subsystems=N_NODES, n_cycles=2)
        assert qc.num_qubits == N_NODES
        assert qc.num_clbits == N_NODES

    def test_tsvf_circuit_has_ancilla(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
        )
        qc = adapter.build_circuit(n_subsystems=N_NODES, n_cycles=2)
        assert qc.num_qubits == N_NODES + 1  # +1 ancilla
        assert qc.num_clbits == N_NODES + 1  # +1 ancilla clbit

    def test_n_subsystems_mismatch_raises(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
        )
        with pytest.raises(ValueError, match="n_subsystems"):
            adapter.build_circuit(n_subsystems=6, n_cycles=1)

    def test_unknown_mode_raises(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="invalid_mode",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
        )
        with pytest.raises(ValueError, match="Unknown algorithm_mode"):
            adapter.build_circuit(n_subsystems=N_NODES, n_cycles=1)

    def test_circuit_depth_increases_with_layers(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
        )
        d1 = adapter.build_circuit(N_NODES, 1).depth()
        d3 = adapter.build_circuit(N_NODES, 3).depth()
        assert d3 > d1

    def test_tsvf_circuit_depth_increases_with_layers(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
        )
        d1 = adapter.build_circuit(N_NODES, 1).depth()
        d3 = adapter.build_circuit(N_NODES, 3).depth()
        assert d3 > d1

    def test_custom_gammas_betas(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
            gammas=[0.5, 1.0],
            betas=[0.3, 0.6],
        )
        qc = adapter.build_circuit(N_NODES, 2)
        assert qc.num_qubits == N_NODES

    def test_scalar_gammas_betas(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
            gammas=0.5,
            betas=0.3,
        )
        qc = adapter.build_circuit(N_NODES, 3)
        assert qc.num_qubits == N_NODES


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — run + parse
# ═══════════════════════════════════════════════════════════════════════════


class TestRunAndParse:
    """Test circuit execution and result parsing."""

    def test_standard_run_returns_dict(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
        )
        qc = adapter.build_circuit(N_NODES, 1)
        raw = adapter.run(qc, shots=100)
        assert isinstance(raw, dict)
        assert "counts" in raw or "pub_result" in raw

    def test_standard_parse_returns_outcomes(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
        )
        qc = adapter.build_circuit(N_NODES, 2)
        raw = adapter.run(qc, shots=100)
        outcomes = adapter.parse_results(raw, N_NODES, 2)
        assert len(outcomes) == 100
        for o in outcomes:
            assert isinstance(o, ParityOutcome)
            assert o.n_subsystems == N_NODES
            assert o.n_cycles == 2
            assert o.parity_matrix.shape == (2, N_NODES)

    def test_tsvf_parse_returns_outcomes(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
            seed=42,
        )
        qc = adapter.build_circuit(N_NODES, 1)
        raw = adapter.run(qc, shots=100)
        outcomes = adapter.parse_results(raw, N_NODES, 1)
        assert len(outcomes) == 100
        for o in outcomes:
            assert isinstance(o, ParityOutcome)
            assert o.n_subsystems == N_NODES
            assert o.n_cycles == 1

    def test_build_and_run_convenience(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
        )
        outcomes = adapter.build_and_run(n_subsystems=N_NODES, n_cycles=1, shots=50)
        assert len(outcomes) == 50
        assert all(isinstance(o, ParityOutcome) for o in outcomes)

    def test_parity_values_are_binary(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
            seed=99,
        )
        outcomes = adapter.build_and_run(n_subsystems=N_NODES, n_cycles=2, shots=200)
        for o in outcomes:
            unique_vals = set(o.parity_matrix.flatten().tolist())
            assert unique_vals.issubset({0, 1})


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — cut quality extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractCutQuality:
    """Test cut quality extraction with and without post-selection."""

    def test_standard_cut_quality_is_valid(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
        )
        qc = adapter.build_circuit(N_NODES, 2)
        raw = adapter.run(qc, shots=1000)
        cut_ratio, approx_ratio, n = adapter.extract_cut_quality(
            raw, postselect=False,
        )
        assert 0.0 <= cut_ratio <= 1.0
        assert approx_ratio >= 0.0
        assert n == 1000

    def test_tsvf_postselected_cut_quality(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
            seed=42,
        )
        qc = adapter.build_circuit(N_NODES, 1)
        raw = adapter.run(qc, shots=2000)
        cut_ratio, approx_ratio, n_accepted = adapter.extract_cut_quality(
            raw, postselect=True,
        )
        assert 0.0 <= cut_ratio <= 1.0
        assert n_accepted >= 0
        assert n_accepted <= 2000

    def test_best_bitstring_extraction(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
        )
        qc = adapter.build_circuit(N_NODES, 2)
        raw = adapter.run(qc, shots=1000)
        bs, cv, count = adapter.extract_best_bitstring(raw, postselect=False)
        assert len(bs) == N_NODES
        assert cv >= 0
        assert count > 0


# ═══════════════════════════════════════════════════════════════════════════
# Integration tests — TrajectoryFilter pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestTrajectoryFilterIntegration:
    """Test the adapter works end-to-end with qgate's TrajectoryFilter."""

    def test_standard_qaoa_through_filter(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
        )
        config = GateConfig(
            n_subsystems=N_NODES,
            n_cycles=1,
            shots=200,
            variant=ConditioningVariant.SCORE_FUSION,
            fusion=FusionConfig(alpha=0.5, threshold=0.5),
            dynamic_threshold=DynamicThresholdConfig(mode="fixed"),
        )
        tf = TrajectoryFilter(config, adapter)
        outcomes = adapter.build_and_run(n_subsystems=N_NODES, n_cycles=1, shots=200)
        result = tf.filter(outcomes)

        assert result.total_shots == 200
        assert 0 <= result.accepted_shots <= 200
        assert 0.0 <= result.acceptance_probability <= 1.0
        assert result.variant == "score_fusion"
        assert len(result.scores) == 200

    def test_tsvf_qaoa_with_galton_threshold(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
            seed=42,
        )
        config = GateConfig(
            n_subsystems=N_NODES,
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
        outcomes = adapter.build_and_run(n_subsystems=N_NODES, n_cycles=2, shots=500)
        result = tf.filter(outcomes)

        assert result.total_shots == 500
        assert result.variant == "score_fusion"
        assert result.dynamic_threshold_final is not None
        assert 0.2 <= result.dynamic_threshold_final <= 0.95
        assert len(result.scores) == 500

    def test_global_conditioning(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
        )
        config = GateConfig(
            n_subsystems=N_NODES,
            n_cycles=1,
            shots=200,
            variant=ConditioningVariant.GLOBAL,
        )
        tf = TrajectoryFilter(config, adapter)
        outcomes = adapter.build_and_run(n_subsystems=N_NODES, n_cycles=1, shots=200)
        result = tf.filter(outcomes)
        assert result.total_shots == 200
        assert result.accepted_shots <= 200

    def test_hierarchical_conditioning(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
        )
        config = GateConfig(
            n_subsystems=N_NODES,
            n_cycles=1,
            shots=200,
            variant=ConditioningVariant.HIERARCHICAL,
            k_fraction=0.5,
        )
        tf = TrajectoryFilter(config, adapter)
        outcomes = adapter.build_and_run(n_subsystems=N_NODES, n_cycles=1, shots=200)
        result = tf.filter(outcomes)
        assert result.total_shots == 200


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — scoring compatibility
# ═══════════════════════════════════════════════════════════════════════════


class TestScoringCompatibility:
    """Ensure ParityOutcomes from QAOATSVFAdapter score correctly."""

    def test_score_batch_works_on_qaoa_outcomes(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
            seed=42,
        )
        outcomes = adapter.build_and_run(n_subsystems=N_NODES, n_cycles=2, shots=100)
        scores = score_batch(outcomes, alpha=0.5)
        assert len(scores) == 100
        for lf, hf, combined in scores:
            assert 0.0 <= lf <= 1.0
            assert 0.0 <= hf <= 1.0
            assert 0.0 <= combined <= 1.0

    def test_perfect_cut_scores_high(self):
        """A ParityOutcome with all zeros should score 1.0."""
        outcome = ParityOutcome(
            n_subsystems=N_NODES,
            n_cycles=2,
            parity_matrix=np.zeros((2, N_NODES), dtype=np.int8),
        )
        scores = score_batch([outcome], alpha=0.5)
        _, _, combined = scores[0]
        assert combined == pytest.approx(1.0)

    def test_all_fail_scores_low(self):
        """A ParityOutcome with all ones should score 0.0."""
        outcome = ParityOutcome(
            n_subsystems=N_NODES,
            n_cycles=2,
            parity_matrix=np.ones((2, N_NODES), dtype=np.int8),
        )
        scores = score_batch([outcome], alpha=0.5)
        _, _, combined = scores[0]
        assert combined == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — bitstring parsing helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestBitstringParsing:
    """Test internal bitstring → parity matrix conversion."""

    def test_tsvf_ancilla_1_good_cut(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
        )
        # "0101" cuts all 4 edges in cycle graph — all qubits contribute
        row = adapter._bitstring_to_parity_row("1 0101", n_subsystems=N_NODES, n_cycles=2)
        assert row.shape == (2, N_NODES)
        # All qubits contribute to at least one cut edge
        np.testing.assert_array_equal(row, np.zeros((2, N_NODES), dtype=np.int8))

    def test_tsvf_ancilla_0_all_fail(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
        )
        # Ancilla=0 → all fail
        row = adapter._bitstring_to_parity_row("0 0101", n_subsystems=N_NODES, n_cycles=2)
        np.testing.assert_array_equal(row, np.ones((2, N_NODES), dtype=np.int8))

    def test_tsvf_ancilla_1_no_cut(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="tsvf",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
        )
        # "0000" cuts nothing — all qubits fail
        row = adapter._bitstring_to_parity_row("1 0000", n_subsystems=N_NODES, n_cycles=1)
        assert row.shape == (1, N_NODES)
        np.testing.assert_array_equal(row, np.ones((1, N_NODES), dtype=np.int8))

    def test_standard_good_cut(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
        )
        row = adapter._bitstring_to_parity_row("0101", n_subsystems=N_NODES, n_cycles=2)
        np.testing.assert_array_equal(row, np.zeros((2, N_NODES), dtype=np.int8))

    def test_standard_no_cut(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
        )
        row = adapter._bitstring_to_parity_row("0000", n_subsystems=N_NODES, n_cycles=1)
        np.testing.assert_array_equal(row, np.ones((1, N_NODES), dtype=np.int8))


# ═══════════════════════════════════════════════════════════════════════════
# No-backend test
# ═══════════════════════════════════════════════════════════════════════════


class TestNoBackend:
    """Verify error when no backend is configured."""

    def test_run_without_backend_raises(self):
        adapter = QAOATSVFAdapter(
            backend=None,
            algorithm_mode="standard",
            edges=SMALL_EDGES,
            n_nodes=N_NODES,
        )
        qc = adapter.build_circuit(N_NODES, 1)
        with pytest.raises(RuntimeError, match="No backend"):
            adapter.run(qc, shots=10)


# ═══════════════════════════════════════════════════════════════════════════
# Default graph generation test
# ═══════════════════════════════════════════════════════════════════════════


class TestDefaultGraph:
    """Test that adapter generates a graph when edges are not provided."""

    def test_auto_generated_graph(self):
        adapter = QAOATSVFAdapter(
            backend=_ideal_backend(),
            algorithm_mode="standard",
            n_nodes=5,
            seed=42,
        )
        assert len(adapter.edges) > 0
        for a, b in adapter.edges:
            assert 0 <= a < 5
            assert 0 <= b < 5
            assert a < b

    def test_auto_graph_deterministic(self):
        a1 = QAOATSVFAdapter(
            backend=_ideal_backend(), algorithm_mode="standard",
            n_nodes=5, seed=42,
        )
        a2 = QAOATSVFAdapter(
            backend=_ideal_backend(), algorithm_mode="standard",
            n_nodes=5, seed=42,
        )
        assert a1.edges == a2.edges
