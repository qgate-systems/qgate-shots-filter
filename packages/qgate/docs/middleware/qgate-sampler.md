# QgateSampler — Transparent SamplerV2 Middleware# QgateSampler — Transparent SamplerV2 Middleware



!!! abstract "One-line swap, zero circuit changes, measurable physics improvement."!!! abstract "One-line swap, zero circuit changes, measurable physics improvement."



`QgateSampler` is a **transparent drop-in replacement** for Qiskit's `SamplerV2``QgateSampler` is a **transparent drop-in replacement** for Qiskit's `SamplerV2`

primitive. Pass it any IBM backend (or Aer simulator) and it autonomouslyprimitive. It wraps any existing sampler (Aer simulator or IBM hardware) and

injects lightweight probe qubits, applies Galton-filtered post-selection,autonomously injects lightweight probe qubits, applies Galton-filtered

and returns standard `PrimitiveResult` objects — all without touching yourpost-selection, and reconstructs clean `PrimitiveResult` objects — all without

circuits.modifying user circuits.



------



## Installation## Installation



```bash```bash

pip install qgate[qiskit]pip install qgate[qiskit]

``````



Requires:Requires:



- Python ≥ 3.9- Python ≥ 3.9

- `qiskit >= 1.0`- `qiskit >= 1.0`

- `qiskit-aer >= 0.13`- `qiskit-aer >= 0.13`

- `qiskit-ibm-runtime >= 0.20`- `qiskit-ibm-runtime >= 0.20`

- `pydantic >= 2.0`- `pydantic >= 2.0`



------



## Quick Start## Quick Start



### 5 lines to filtered results### Simulator



```python```python

from qiskit.circuit import QuantumCircuitfrom qiskit.circuit import QuantumCircuit

from qiskit_ibm_runtime import QiskitRuntimeServicefrom qiskit.primitives import StatevectorSampler

from qgate import QgateSamplerfrom qgate import QgateSampler, SamplerConfig



# 1. Connect to IBM Quantum# Build any circuit

service = QiskitRuntimeService()qc = QuantumCircuit(2)

backend = service.backend("ibm_fez")qc.h(0)

qc.cx(0, 1)

# 2. Build your circuit (unchanged)qc.measure_all()

qc = QuantumCircuit(2)

qc.h(0)# Wrap the sampler — that's it

qc.cx(0, 1)sampler = QgateSampler(

qc.measure_all()    inner=StatevectorSampler(),

    config=SamplerConfig(),        # sensible defaults

# 3. Wrap the backend — that's it)

sampler = QgateSampler(backend=backend)result = sampler.run([qc])

counts = result[0].data.meas.get_counts()

# 4. Run exactly like SamplerV2print(counts)  # {'00': ..., '11': ...}

job = sampler.run([(qc,)])```

result = job.result()

### IBM Hardware

# 5. Use the result exactly like before

counts = result[0].data.meas.get_counts()```python

print(counts)  # {'00': ..., '11': ...}  — higher-fidelity shots onlyfrom qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

```

service = QiskitRuntimeService()

!!! tip "The key difference"backend = service.least_busy(operational=True, simulator=False)

    With a standard `SamplerV2` you get **all** shots, including noise-corrupted

    ones. With `QgateSampler` you get only the **high-fidelity** shots — thesampler = QgateSampler(

    probe-injection and Galton filtering happen transparently.    inner=SamplerV2(mode=backend),

    config=SamplerConfig(

### Simulator (Aer)        probe_ry_angle=0.25,

        galton_quantile=0.75,

```python    ),

from qiskit_aer import AerSimulator)

from qgate import QgateSampler

# Use exactly like SamplerV2

backend = AerSimulator()result = sampler.run([(qc,)])

sampler = QgateSampler(backend=backend)counts = result[0].data.meas.get_counts()

```

job = sampler.run([(qc,)])

result = job.result()---

counts = result[0].data.meas.get_counts()

```## How It Works



### With a pre-configured SamplerV2```mermaid

flowchart LR

If you already have a `SamplerV2` with custom options, pass it via the    A["User Circuit"] --> B["Probe Injection"]

`sampler=` parameter:    B --> C["Inner SamplerV2"]

    C --> D["Galton Filter"]

```python    D --> E["Clean PrimitiveResult"]

from qiskit_ibm_runtime import SamplerV2

    style A fill:#e3f2fd

inner = SamplerV2(mode=backend, options={"default_shots": 4096})    style B fill:#fff3e0

sampler = QgateSampler(backend=backend, sampler=inner)    style C fill:#e8f5e9

```    style D fill:#fce4ec

    style E fill:#e3f2fd

---```



## How It Works### 1. Probe Injection



```mermaid`QgateSampler` adds a single ancilla qubit (`qgate_anc`) and a classical

flowchart LRregister (`qgate_probe`) to the user circuit. Controlled-RY gates are placed

    A["Your Circuit"] --> B["① Probe Injection"]on nearest-neighbour qubit pairs. The probe rotation angle controls the

    B --> C["② SamplerV2.run()"]sensitivity of the fidelity signal.

    C --> D["③ Galton Filter"]

    D --> E["④ Clean PrimitiveResult"]- **Zero depth overhead:** the probe gates are single-layer, controlled-RY

  operations that do not increase the effective circuit depth.

    style A fill:#e3f2fd- **Transparent:** the probe qubit and classical register are automatically

    style B fill:#fff3e0  stripped from the returned results.

    style C fill:#e8f5e9

    style D fill:#fce4ec### 2. Inner Execution

    style E fill:#e3f2fd

```The augmented circuit is transpiled and executed via the wrapped `SamplerV2`.

This can be any Qiskit-compatible sampler — `StatevectorSampler`,

### ① Probe Injection`AerSimulator`, or a real IBM backend.



`QgateSampler` adds a single ancilla qubit (`qgate_anc`) and a classical### 3. Galton Filtering

register (`qgate_probe`) to each circuit. Controlled-RY gates are placed

on nearest-neighbour qubit pairs. The probe rotation angle controls theEach shot receives a fidelity score based on the probe outcome. The

sensitivity of the fidelity signal.**Galton adaptive threshold** (a self-contained quantile estimator inspired

by the Galton board) maintains a rolling window of recent scores and

- **Minimal depth increase:** single-layer CRY gates — typically < 1% circuitdynamically sets the acceptance cutoff:

  depth overhead.

- **Fully transparent:** the probe qubit and classical register are stripped- Shots with probe scores **above the threshold** are kept.

  from the returned results — your downstream code never sees them.- Shots **below the threshold** are discarded as noise-corrupted.

- The threshold adapts automatically — no hand-tuning required.

### ② Inner Execution

### 4. Result Reconstruction

The augmented circuit is transpiled and executed via a standard `SamplerV2`.

This can target any Qiskit-compatible backend — `AerSimulator` or real IBMAccepted shots are reassembled into standard Qiskit data structures:

hardware.

- `BitArray` with correct shape and number of shots

### ③ Galton Filtering- `DataBin` with all user classical registers preserved

- `PubResult` and `PrimitiveResult` fully compatible with downstream code

Each shot receives a fidelity score based on the probe outcome. The

**Galton adaptive threshold** — a self-contained quantile estimator inspired!!! tip "Downstream compatibility"

by the Galton board — maintains a rolling window of recent scores and    Any code that works with `SamplerV2` results (`result[0].data.meas.get_counts()`,

dynamically sets the acceptance cutoff:    `result[0].data.meas.bitcount()`, etc.) works unchanged with `QgateSampler`.



- Shots with probe scores **above the threshold** → kept ✅---

- Shots **below the threshold** → discarded as noise-corrupted ❌

- The threshold **adapts automatically** — no hand-tuning required.## API Reference



### ④ Result Reconstruction### `QgateSampler`



Accepted shots are reassembled into standard Qiskit data structures:```python

from qgate import QgateSampler

- `BitArray` with correct shape and number of shots

- `DataBin` with all user classical registers preservedsampler = QgateSampler(

- `PubResult` and `PrimitiveResult` fully compatible with downstream code    inner: SamplerV2,            # any Qiskit SamplerV2 instance

    config: SamplerConfig = ..., # optional configuration

!!! success "Downstream compatibility")

    Any code that works with `SamplerV2` results — `result[0].data.meas.get_counts()`,```

    `result[0].data.meas.bitcount()`, etc. — works unchanged with `QgateSampler`.

#### Methods

---

| Method | Signature | Description |

## API Reference|---|---|---|

| `run` | `run(pubs, **kwargs) → PrimitiveResult` | Execute pubs through the filter pipeline |

### `QgateSampler`

The `run()` method accepts the same arguments as `SamplerV2.run()`:

```python

from qgate import QgateSampler, SamplerConfig- A list of PUBs (Primitive Unified Blocs): `QuantumCircuit`, `(circuit,)`,

  `(circuit, param_values)`, or `(circuit, param_values, shots)`

sampler = QgateSampler(- Any additional keyword arguments are forwarded to the inner sampler.

    backend,                         # any IBM/Aer backend

    config=SamplerConfig(),          # optional tuning (see below)### `SamplerConfig`

    sampler=None,                    # optional pre-built SamplerV2

)```python

```from qgate import SamplerConfig



| Argument | Type | Required | Description |config = SamplerConfig(

|---|---|---|---|    probe_ry_angle=0.25,

| `backend` | Backend | ✅ | IBM Runtime backend or `AerSimulator` |    galton_quantile=0.75,

| `config` | `SamplerConfig` | ❌ | Filtering configuration (defaults are sensible) |    galton_window=64,

| `sampler` | `SamplerV2` | ❌ | Pre-initialised sampler; if `None`, one is created from `backend` |    galton_warmup=32,

    probe_pairs="nn",

#### Methods)

```

| Method | Returns | Description |

|---|---|---|`SamplerConfig` is a **Pydantic v2 frozen model** — immutable after creation,

| `sampler.run(pubs, *, shots=None)` | `QgateSamplerResult` | Execute PUBs through the filter pipeline |with full validation and serialisation support.

| `job.result()` | `PrimitiveResult` | Retrieve the filtered results (lazy evaluation) |

#### Parameters

#### Properties

| Parameter | Type | Default | Description |

| Property | Type | Description ||---|---|---|---|

|---|---|---|| `probe_ry_angle` | `float` | `0.25` | Controlled-RY rotation angle in radians. Higher values produce a stronger probe signal but may perturb the circuit state slightly. Recommended range: 0.1–0.5. |

| `sampler.config` | `SamplerConfig` | The active configuration (immutable) || `galton_quantile` | `float` | `0.75` | Acceptance quantile (0–1). A value of 0.75 keeps the top 25% of shots by fidelity score. Higher values are more selective. |

| `sampler.backend` | Backend | The underlying backend || `galton_window` | `int` | `64` | Rolling window size for the adaptive threshold estimator. Larger windows produce smoother thresholds but adapt more slowly. |

| `sampler.current_threshold` | `float` | Current Galton adaptive threshold value || `galton_warmup` | `int` | `32` | Minimum number of shots collected before the Galton threshold activates. During warmup, all shots are accepted. |

| `sampler.in_warmup` | `bool` | `True` if the Galton window is still warming up || `probe_pairs` | `str` | `"nn"` | Probe qubit pairing strategy. `"nn"` places probes on nearest-neighbour pairs; `"all"` places probes on all pairs (higher overhead, stronger signal). |



#### PUB formats---



The `run()` method accepts the same PUB formats as `SamplerV2.run()`:## Advanced Usage



```python### Custom configuration for high-noise environments

sampler.run([qc])                              # bare circuit

sampler.run([(qc,)])                           # 1-tuple```python

sampler.run([(qc, param_values)])              # with parametersconfig = SamplerConfig(

sampler.run([(qc, param_values, 4096)])        # with parameters + shots    probe_ry_angle=0.4,        # stronger probe signal

```    galton_quantile=0.85,      # keep only top 15%

    galton_window=128,         # larger window for stability

### `SamplerConfig`)

```

```python

from qgate import SamplerConfig### Accessing filter metadata



config = SamplerConfig(```python

    probe_angle=0.5,              # stronger probe signalresult = sampler.run([qc])

    target_acceptance=0.10,       # keep top 10%pub_result = result[0]

    window_size=2048,             # smaller rolling window

)# Standard Qiskit access

```counts = pub_result.data.meas.get_counts()

bitarray = pub_result.data.meas

`SamplerConfig` is a **Pydantic v2 frozen model** — immutable after creation,

with full validation and serialisation support.# Filter metadata (when available)

metadata = pub_result.metadata

#### Parametersprint(f"Shots accepted: {bitarray.num_shots}")

```

| Parameter | Type | Default | Range | Description |

|---|---|---|---|---|### Using with VQE / QAOA / any variational algorithm

| `probe_angle` | `float` | `π/6 ≈ 0.524` | `(0, π]` | Controlled-RY rotation angle in radians. Higher values produce a stronger probe signal but may perturb the circuit state. |

| `target_acceptance` | `float` | `0.05` | `(0, 1)` | Target fraction of shots to accept. `0.05` keeps the top 5%. Lower values = stricter filtering = higher fidelity. |```python

| `window_size` | `int` | `4096` | `≥ 64` | Rolling window capacity for the Galton adaptive threshold. Larger windows → smoother thresholds, slower adaptation. |from qiskit.circuit import QuantumCircuit, ParameterVector

| `min_window_size` | `int` | `100` | `≥ 1` | Minimum observations before the adaptive threshold activates. During warmup, `baseline_threshold` is used. |from qiskit_ibm_runtime import EstimatorV2, SamplerV2

| `baseline_threshold` | `float` | `0.65` | `[0, 1]` | Fallback threshold used during the warmup phase. |

| `min_threshold` | `float` | `0.3` | `[0, 1]` | Floor — threshold never drops below this value. |# QgateSampler works with parameterized circuits

| `max_threshold` | `float` | `0.95` | `[0, 1]` | Ceiling — threshold never exceeds this value. |theta = ParameterVector("θ", 4)

| `use_quantile` | `bool` | `True` | — | If `True`, use empirical quantile for threshold. If `False`, use z-score mode. |qc = QuantumCircuit(4)

| `robust_stats` | `bool` | `True` | — | If `True` and `use_quantile=False`, use median + MAD instead of mean + std. |for i in range(4):

| `z_sigma` | `float` | `1.645` | `≥ 0` | Number of σ above centre for z-score mode (only when `use_quantile=False`). |    qc.ry(theta[i], i)

| `optimization_level` | `int` | `1` | `[0, 3]` | Qiskit transpiler optimization level. |for i in range(3):

| `oversample_factor` | `float` | `1.0` | `[1, 20]` | Request extra shots from backend to compensate for filtering. `1.0` = no oversampling. |    qc.cx(i, i + 1)

qc.measure_all()

!!! example "Recommended presets"

    === "Conservative (default)"sampler = QgateSampler(

        ```python    inner=SamplerV2(mode=backend),

        SamplerConfig()  # probe_angle=π/6, target_acceptance=5%    config=SamplerConfig(),

        ```)

        Good for most circuits. Accepts ~5% of shots with highest fidelity.

# Parameters are bound normally via PUBs

    === "Aggressive filtering"result = sampler.run([(qc, [0.1, 0.2, 0.3, 0.4])])

        ```python```

        SamplerConfig(target_acceptance=0.02, probe_angle=0.8)

        ```---

        For very noisy backends. Keeps only top 2% — fewer shots, much higher quality.

## Validation Results

    === "Gentle filtering"

        ```python### Real IBM Hardware

        SamplerConfig(target_acceptance=0.20, probe_angle=0.3)

        ```| Backend | Architecture | Qubits | Protocol | Key Result |

        For shallow circuits where most shots are already decent. Keeps top 20%.|---|---|---|---|---|

| **IBM Fez** | Heron r2 | 156 | 2Q Bell state, 100 shots | **95% Bell fidelity** on filtered shots; probe stripped cleanly |

---| **IBM Torino** | Heron r2 | 133 | Utility-scale TFIM | Galton acceptance ~9.7%; cooling Δ = −0.080 |

| **IBM Brisbane** | Eagle r3 | 127 | 8Q TFIM VQE | **6.6% acceptance** (vs 0% raw post-selection) |

## Advanced Usage

### E2E Physics Validation (Simulator)

### Using with VQE / QAOA / any variational algorithm

10-trial paired experiment on 8-qubit TFIM at the quantum critical point,

```pythonusing an IBM Heron-class noise model ($T_1 = 300\mu s$, $T_2 = 150\mu s$,

from qiskit.circuit import QuantumCircuit, ParameterVector1Q depolarising $= 10^{-3}$, 2Q depolarising $= 10^{-2}$):



# Parameterized ansatz| Metric | Value |

theta = ParameterVector("θ", 4)|---|---|

qc = QuantumCircuit(4)| Mean MSE reduction | **+0.69%** |

for i in range(4):| Paired t-test | **p = 1.26 × 10⁻⁴** |

    qc.ry(theta[i], i)| Trials improved | **9 / 10** |

for i in range(3):| Validation protocol | VQE warm-up (ZZ-only Hamiltonian) → QgateSampler vs raw SamplerV2 |

    qc.cx(i, i + 1)

qc.measure_all()### Statistical Bias Study (15 independent trials × 100K shots)



sampler = QgateSampler(backend=backend)| Experiment | Key Finding |

|---|---|

# Parameters are bound normally via PUBs| Noise robustness | MSE↓ **13.6% → 20.7%** as noise increases |

job = sampler.run([(qc, [0.1, 0.2, 0.3, 0.4])])| Qubit scaling (8–16Q) | Stable MSE↓ **14.5–16.5%**, variance↓ up to **5,360×** |

result = job.result()| Cross-algorithm | VQE **14.8%**, QAOA **48.8%**, Grover **24.4%** MSE reduction |

counts = result[0].data.meas.get_counts()| Train/test split | **14.7%** MSE↓ on blind test (p = 0.001), frozen threshold σ = 0.000 |

```

---

### Inspecting the adaptive threshold

## Frequently Asked Questions

```python

sampler = QgateSampler(backend=backend)??? question "Does QgateSampler modify my circuit?"

    Only at the probe-injection layer. Your user-visible qubits and classical

# After first run    registers are untouched. The probe ancilla and probe classical register are

job = sampler.run([(qc,)])    added automatically and stripped from the returned results.

result = job.result()

??? question "How many shots do I lose to filtering?"

print(f"Warmup: {sampler.in_warmup}")    It depends on the noise environment and `galton_quantile` setting. Typical

print(f"Threshold: {sampler.current_threshold:.4f}")    acceptance rates are 10–30% on real hardware. The accepted shots have

print(f"Shots accepted: {result[0].data.meas.num_shots}")    significantly higher fidelity, improving downstream expectation values.

```

??? question "Can I use QgateSampler with Estimator?"

### Oversampling to maintain shot count    `QgateSampler` wraps `SamplerV2` specifically. For `EstimatorV2` workflows,

    use the `TrajectoryFilter` API directly, or sample bitstrings via

By default, filtering reduces the number of returned shots. Use    `QgateSampler` and compute expectation values classically.

`oversample_factor` to automatically request extra shots from the backend:

??? question "What is the overhead?"

```python    - **Quantum overhead:** 1 additional qubit + single-layer CRY gates (negligible depth increase)

config = SamplerConfig(    - **Classical overhead:** probe scoring and threshold computation (< 1 ms for typical shot counts)

    target_acceptance=0.10,       # keep top 10%    - **Shot overhead:** you need ~3–10× more shots to compensate for filtering, but the filtered

    oversample_factor=10.0,       # request 10× more → ~same final count      shots are far more accurate, often yielding a net TTS improvement.

)

sampler = QgateSampler(backend=backend, config=config)??? question "Is QgateSampler compatible with error mitigation (ZNE, PEC, etc.)?"

```    Yes. QgateSampler operates at the shot-filtering layer and is orthogonal to

    circuit-level error mitigation techniques. You can compose them: run

---    QgateSampler for shot filtering, then apply ZNE or PEC on the filtered results.



## Validation Results---



### Real IBM Hardware## Patent Notice



| Backend | Architecture | Qubits | Protocol | Key Result |The algorithms implemented in `QgateSampler` are covered by pending patent

|---|---|---|---|---|applications:

| **IBM Fez** | Heron r2 | 156 | 2Q Bell state, 100 shots | **95% Bell fidelity** on filtered shots; probe stripped cleanly |

| **IBM Torino** | Heron r2 | 133 | Utility-scale TFIM | Galton acceptance ~9.7%; cooling Δ = −0.080 |- **U.S.** App. Nos. 63/983,831 & 63/989,632

| **IBM Brisbane** | Eagle r3 | 127 | 8Q TFIM VQE | **6.6% acceptance** (vs 0% raw post-selection) |- **Israel** App. No. 326915 (Paris Convention, priority date 27/02/2026)



### E2E Physics Validation (Simulator)Licensed under the **QGATE Source Available Evaluation License v1.2**.

Academic research, internal evaluation, and peer review are freely permitted.

10-trial paired experiment on 8-qubit TFIM at the quantum critical point,Commercial deployment requires a separate license — contact

using an IBM Heron-class noise model ($T_1 = 300\mu s$, $T_2 = 150\mu s$,[ranbuch@gmail.com](mailto:ranbuch@gmail.com).

1Q depolarising $= 10^{-3}$, 2Q depolarising $= 10^{-2}$):

| Metric | Value |
|---|---|
| Mean MSE reduction | **+0.69%** |
| Paired t-test | **p = 1.26 × 10⁻⁴** |
| Trials improved | **9 / 10** |
| Validation protocol | VQE warm-up (ZZ-only Hamiltonian) → QgateSampler vs raw SamplerV2 |

### Statistical Bias Study (15 independent trials × 100K shots)

| Experiment | Key Finding |
|---|---|
| Noise robustness | MSE↓ **13.6% → 20.7%** as noise increases |
| Qubit scaling (8–16Q) | Stable MSE↓ **14.5–16.5%**, variance↓ up to **5,360×** |
| Cross-algorithm | VQE **14.8%**, QAOA **48.8%**, Grover **24.4%** MSE reduction |
| Train/test split | **14.7%** MSE↓ on blind test (p = 0.001), frozen threshold σ = 0.000 |

---

## Frequently Asked Questions

??? question "Does QgateSampler modify my circuit?"
    Only at the probe-injection layer. Your user-visible qubits and classical
    registers are untouched. The probe ancilla and probe classical register are
    added automatically and stripped from the returned results.

??? question "How many shots do I lose to filtering?"
    It depends on the noise environment and `target_acceptance` setting.
    With the default `target_acceptance=0.05`, about 5% of shots are kept.
    Use `oversample_factor` to request extra shots and maintain your target
    count. The accepted shots have significantly higher fidelity, improving
    downstream expectation values.

??? question "Can I use QgateSampler with EstimatorV2?"
    `QgateSampler` wraps `SamplerV2` specifically. For `EstimatorV2` workflows,
    sample bitstrings via `QgateSampler` and compute expectation values
    classically, or use the `TrajectoryFilter` API directly.

??? question "What is the overhead?"
    - **Quantum overhead:** 1 ancilla qubit + single-layer CRY gates (< 1% depth increase)
    - **Classical overhead:** probe scoring + threshold computation (< 1 ms for typical shot counts)
    - **Shot overhead:** you need more shots to compensate for filtering — use `oversample_factor`
      to automate this. The filtered shots are far more accurate, often yielding a net TTS improvement.

??? question "Is QgateSampler compatible with error mitigation (ZNE, PEC, etc.)?"
    Yes. QgateSampler operates at the shot-filtering layer and is orthogonal to
    circuit-level error mitigation. You can compose them: run QgateSampler for
    shot filtering, then apply ZNE or PEC on the filtered results.

??? question "Do I need an IBM Quantum account?"
    For real hardware, yes — you need a free [IBM Quantum](https://quantum.ibm.com/)
    account. For local simulation, use `AerSimulator` with no account required.

---

## Patent Notice

The algorithms implemented in `QgateSampler` are covered by pending patent
applications:

- **U.S.** App. Nos. 63/983,831 & 63/989,632
- **Israel** App. No. 326915 (Paris Convention, priority date 27/02/2026)

Licensed under the **QGATE Source Available Evaluation License v1.2**.
Academic research, internal evaluation, and peer review are freely permitted.
Commercial deployment requires a separate license — contact
[ranbuch@gmail.com](mailto:ranbuch@gmail.com).
