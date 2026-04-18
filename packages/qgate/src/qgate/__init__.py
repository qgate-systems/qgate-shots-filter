"""
qgate — Quantum Trajectory Filter.

Runtime post-selection conditioning for quantum circuits.  This package
explores trajectory filtering concepts from US Patent Application
pending patent applications (see LICENSE).
The underlying invention is patent pending.

Quick start::

    from qgate import TrajectoryFilter, GateConfig
    from qgate.adapters import MockAdapter

    config = GateConfig(n_subsystems=4, n_cycles=2, shots=1024)
    adapter = MockAdapter(error_rate=0.05, seed=42)
    tf = TrajectoryFilter(config, adapter)
    result = tf.run()
    print(result.acceptance_probability)

Install extras for specific backends::

    pip install qgate[qiskit]       # IBM Qiskit
    pip install qgate[cirq]         # Google Cirq (stub)
    pip install qgate[pennylane]    # PennyLane (stub)
    pip install qgate[all]          # Everything

License: QGATE SOURCE AVAILABLE EVALUATION LICENSE v1.2 — see LICENSE
"""

__version__ = "0.6.0"

# ── Primary public API ────────────────────────────────────────────────────
from qgate.adapters.base import BaseAdapter
from qgate.adapters.grover_adapter import GroverTSVFAdapter
from qgate.adapters.qaoa_adapter import QAOATSVFAdapter
from qgate.adapters.qpe_adapter import QPETSVFAdapter
from qgate.adapters.registry import list_adapters, load_adapter
from qgate.adapters.vqe_adapter import VQETSVFAdapter

# ── Legacy / backward-compatible re-exports ──────────────────────────────
# These symbols were available in qgate <= 0.2; kept for compatibility.
from qgate.conditioning import (
    ConditioningStats,
    ParityOutcome,
    apply_rule_to_batch,
    decide_global,
    decide_hierarchical,
    decide_score_fusion,
)
from qgate.config import (
    AdapterKind,
    ConditioningVariant,
    DynamicThresholdConfig,
    FusionConfig,
    GateConfig,
    ProbeConfig,
    ThresholdMode,
)
from qgate.filter import TrajectoryFilter
from qgate.monitors import (
    MultiRateMonitor,
    compute_window_metric,
    score_fusion,
)
from qgate.run_logging import FilterResult, RunLogger, compute_run_id

# ── QgateSampler OS layer ─────────────────────────────────────────────────
# Transparent drop-in SamplerV2 replacement with autonomous probe injection
# and Galton-filtered result reconstruction.
# Patent pending.
from qgate.sampler import QgateSampler, SamplerConfig
from qgate.scoring import (
    fuse_scores,
    score_batch,
    score_outcome,
)
from qgate.threshold import (
    GaltonAdaptiveThreshold,
    GaltonSnapshot,
    estimate_diffusion_width,
)

# ── Qgate Advantage cloud client ──────────────────────────────────────────
from qgate.cloud import QgateAdvantageClient, QgateAPIError, QgateTaskError, QgateTimeoutError

# ── Qgate Execute pipeline (circuit → API → Result) ──────────────────────
from qgate.client import (
    AsyncQgateClient,
    ClientConfig,
    QgateBackendError,
    QgateClientError,
    reconstruct_result,
)
from qgate.execute import ExecutionContext, execute
from qgate.transpiler import (
    circuit_to_dag,
    dag_to_circuit,
    deserialise_payload,
    inject_telemetry,
    serialise_payload,
    strip_telemetry_registers,
)

__all__ = [
    "AdapterKind",
    # Execute pipeline
    "AsyncQgateClient",
    # Primary API
    "BaseAdapter",
    "ClientConfig",
    # Legacy (backward compat)
    "ConditioningStats",
    "ConditioningVariant",
    "DynamicThresholdConfig",
    "ExecutionContext",
    "FilterResult",
    "FusionConfig",
    # Galton adaptive threshold
    "GaltonAdaptiveThreshold",
    "GaltonSnapshot",
    "GateConfig",
    # Grover/TSVF adapter
    "GroverTSVFAdapter",
    "MultiRateMonitor",
    "ParityOutcome",
    "ProbeConfig",
    # QAOA/TSVF adapter
    "QAOATSVFAdapter",
    # QPE/TSVF adapter
    "QPETSVFAdapter",
    # QgateSampler OS
    "QgateSampler",
    # Qgate Advantage cloud client
    "QgateAdvantageClient",
    "QgateAPIError",
    "QgateBackendError",
    "QgateClientError",
    "QgateTaskError",
    "QgateTimeoutError",
    "RunLogger",
    "SamplerConfig",
    "ThresholdMode",
    "TrajectoryFilter",
    # VQE/TSVF adapter
    "VQETSVFAdapter",
    "apply_rule_to_batch",
    # Transpiler functions
    "circuit_to_dag",
    "compute_run_id",
    "compute_window_metric",
    "dag_to_circuit",
    "decide_global",
    "decide_hierarchical",
    "decide_score_fusion",
    "deserialise_payload",
    "estimate_diffusion_width",
    # Execute context manager
    "execute",
    "fuse_scores",
    "inject_telemetry",
    "list_adapters",
    "load_adapter",
    # Client functions
    "reconstruct_result",
    "score_batch",
    "score_fusion",
    "score_outcome",
    "serialise_payload",
    "strip_telemetry_registers",
]
