# Quantum Error Suppression via Post-Selection Conditioning on Bell-Pair Subsystems

**Patent Reference:** US Patent Application Nos. 63/983,831 & 63/989,632 | IL Patent Application No. 326915

## Overview

This experiment implements and benchmarks **quantum error suppression via
post-selection conditioning** on IBM Quantum dynamic circuits. The method
monitors Bell-pair subsystems using mid-circuit Z-parity measurements and
applies three conditioning strategies:

| Strategy | Rule | Key Parameter |
|---|---|---|
| **Global** | All subsystems must pass every cycle | — |
| **Hierarchical k-of-N** | ≥ ⌈k·N⌉ subsystems pass each cycle | k ∈ {0.8, 0.9} |
| **Score Fusion** | α·score_LF + (1−α)·score_HF ≥ θ | α ∈ {0.3, 0.7}, θ = 0.65 |

### Multi-Rate Monitoring

- **HF (high-frequency):** Z-parity measured every cycle
- **LF (low-frequency):** Z-parity measured every 2nd cycle (0, 2, 4, …)
- **Score fusion** blends HF and LF scores with weight α

### Batch-Level Abort

A short **probe circuit** (W=1) is run first with 1000 shots. If the
pass-rate falls below θ_batch = 0.65, the full circuit is skipped — saving
hardware time on configurations unlikely to yield useful data.

## Parameter Sweep

| Parameter | Values | Description |
|---|---|---|
| N | 1, 2, 4, 8 | Number of Bell-pair subsystems (2N qubits) |
| W | 2, 4 | Monitoring cycles per run |
| D | 2, 4, 8 | Scramble depth (random rotations + barriers) |
| k | 0.8, 0.9 | Hierarchical fraction |
| α | 0.3, 0.7 | Fusion weight (LF vs HF) |
| Shots | 5000 | Per circuit configuration |

**Total:** 4 × 2 × 3 = 24 circuit configs × 5 variants = **120 result rows**

## Setup

### Prerequisites

- Python ≥ 3.9
- An IBM Quantum account (for real hardware only)

### Installation

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install Qiskit core + Aer simulator
pip install qiskit qiskit-aer

# For IBM Quantum hardware (optional)
pip install qiskit-ibm-runtime

# Plotting dependencies (likely already installed)
pip install matplotlib pandas numpy
```

Or using `uv`:

```bash
uv pip install qiskit qiskit-aer qiskit-ibm-runtime matplotlib pandas numpy
```

### Verify Installation

```bash
python -c "import qiskit; print(f'Qiskit {qiskit.__version__}')"
python -c "from qiskit_aer import AerSimulator; print('Aer OK')"
```

## Results Summary (IBM Marrakesh — 2026-02-16)

The experiment was executed on **IBM Marrakesh** (156 qubits) with the
following results across **120 rows** (24 circuit configs × 5 variants).
Total execution time: **≈ 6 minutes** on hardware.

### Acceptance probability by variant

| Variant | Mean acceptance | Best acceptance | Zero-acceptance configs |
|---|---|---|---|
| Global | Low at N ≥ 2 | High only at N = 1 | Majority at N ≥ 4 |
| Hierarchical (k=0.8) | Moderate | Sustained at larger N | Fewer than global |
| Hierarchical (k=0.9) | Slightly below k=0.8 | — | — |
| Score fusion (α=0.3) | **Highest overall** | Robust across D | Fewest |
| Score fusion (α=0.7) | High | — | — |

### Key findings on real hardware

1. **Score fusion dramatically outperforms global and hierarchical conditioning**
   on noisy IBM hardware — the soft threshold absorbs device-specific noise
   fluctuations that cause hard binary rules to over-reject.

2. **Global conditioning collapses at N ≥ 4** on real hardware, consistent with
   QuTiP simulation predictions.

3. **Batch-abort probe** (θ_batch was set to 0.0 for this run to force all configs)
   correctly identifies low-fidelity configurations — useful for future runs
   with θ_batch > 0.

4. **Hardware backend:** IBM Marrakesh (`ibm_marrakesh`), 156 qubits,
   Eagle r3 processor.

### Generated figures

| File | Description |
|---|---|
| `acceptance_vs_N.png` | Acceptance probability vs N, faceted by scramble depth D |
| `tts_vs_N.png` | Time-to-solution vs N for each variant |
| `fusion_vs_alpha.png` | Score fusion acceptance vs α at high D |
| `probe_heatmap.png` | Batch-abort probe pass-rate heatmap |
| `acceptance_vs_depth.png` | Acceptance vs scramble depth D, faceted by N |

---

## Running the Experiment

### 1. Local Noisy Simulation (Recommended First Step)

```bash
python qiskit_experiment/run_ibm_experiment.py --mode aer
```

This uses `AerSimulator` with a depolarising noise model:
- 1-qubit gate error: 0.1%
- 2-qubit gate error: 1%
- Measurement error: 2%

### 2. Ideal (Noiseless) Simulation

```bash
python qiskit_experiment/run_ibm_experiment.py --mode aer --no-noise
```

### 3. Real IBM Quantum Hardware

```bash
# Option A: Pass token directly
python qiskit_experiment/run_ibm_experiment.py --mode ibm --token YOUR_TOKEN

# Option B: Set environment variable
export IBMQ_TOKEN="YOUR_TOKEN"
python qiskit_experiment/run_ibm_experiment.py --mode ibm
```

The script automatically selects the **least-busy dynamic-capable backend**
with ≥ 17 qubits (16 data + 1 ancilla for N=8).

### 4. Custom Sweep

```bash
python qiskit_experiment/run_ibm_experiment.py --mode aer \
    --N 1 2 4 \
    --D 2 4 \
    --W 2 4 \
    --shots 2000 \
    --output-dir my_results
```

### CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--mode` | `aer` | `aer` (simulator) or `ibm` (real hardware) |
| `--token` | env var | IBM Quantum API token |
| `--no-noise` | false | Use ideal simulator |
| `--N` | 1 2 4 8 | System sizes |
| `--W` | 2 4 | Monitoring cycles |
| `--D` | 2 4 8 | Scramble depths |
| `--k` | 0.8 0.9 | Hierarchical fractions |
| `--alpha` | 0.3 0.7 | Fusion weights |
| `--threshold` | 0.65 | Fusion combined threshold |
| `--shots` | 5000 | Shots per circuit |
| `--probe-shots` | 1000 | Shots for probe circuit |
| `--theta-batch` | 0.65 | Batch-abort threshold |
| `--output-dir` | `patent_appendix` | Output directory |
| `--no-plots` | false | Skip figure generation |
| `--quiet` | false | Suppress progress output |

## Output Files

All outputs are saved to `patent_appendix/` (or `--output-dir`):

```
patent_appendix/
├── results.csv              # Full results (120+ rows)
├── run_log.json             # Execution metadata
├── acceptance_vs_N.png      # Fig 1: Acceptance prob. vs N, faceted by D
├── tts_vs_N.png             # Fig 2: Time-to-solution vs N
├── fusion_vs_alpha.png      # Fig 3: Fusion acceptance vs α at high D
├── probe_heatmap.png        # Fig 4: Batch-abort probe pass-rate
└── acceptance_vs_depth.png  # Fig 5: Acceptance vs D, faceted by N
```

### CSV Columns

| Column | Description |
|---|---|
| `variant` | `global`, `hierarchical`, or `score_fusion` |
| `N` | Number of subsystems |
| `D` | Scramble depth |
| `W` | Monitoring cycles |
| `k_fraction` | Hierarchical threshold (if applicable) |
| `alpha` | Fusion weight (if applicable) |
| `threshold_combined` | Fusion threshold (if applicable) |
| `total_shots` | Total shots executed |
| `accepted_shots` | Shots passing conditioning |
| `acceptance_probability` | accepted / total |
| `TTS` | 1 / acceptance_probability |
| `probe_pass_rate` | W=1 probe batch pass rate |
| `batch_aborted` | True if probe failed θ_batch |
| `mean_score` | Average fusion score (score_fusion only) |

## Circuit Architecture

```
         ┌──────────┐   ┌──────────────┐   ┌────────────┐
|0⟩──H── ┤          ├───┤  Scramble ×D  ├───┤  Z-parity  ├──── ... (W cycles)
|0⟩───── ┤  Bell    ├───┤  (Rx,Ry,Rz   ├───┤  mid-circ   ├────
         ┤  Pair    ├   ┤   +barrier)   ├   ┤  measure    ├
|0⟩──H── ┤  ×N     ├───┤              ├───┤  (ancilla)  ├────
|0⟩───── ┤          ├───┤              ├───┤             ├────
         └──────────┘   └──────────────┘   └────────────┘

Ancilla (reused): CNOT(qa, anc); CNOT(qb, anc); Measure(anc); Reset(anc)
Even parity (anc=0) → subsystem PASS
Odd parity  (anc=1) → subsystem FAIL
```

## Module Structure

```
qiskit_experiment/
├── __init__.py              # Package metadata
├── circuits.py              # Circuit construction (Bell pairs, scramble, parity)
├── conditioning.py          # Post-selection rules (global, hierarchical, fusion)
├── sweep.py                 # Parameter sweep engine with batch-abort
├── plots.py                 # Patent-appendix figure generation
├── run_ibm_experiment.py    # Main CLI entry point
└── README.md                # This file
```

## Relation to Patent Claims

This experiment provides **empirical evidence** for the claims in
US Patent Application Nos. 63/983,831 & 63/989,632 | IL Patent Application No. 326915:

1. **Claim 1 (Hierarchical conditioning):** Demonstrated via the
   k-of-N rule — acceptance probability remains high even at large N
   and moderate noise, where global conditioning fails.

2. **Claim 2 (Multi-rate monitoring):** HF and LF parity checks at
   different cadences enable score fusion, which provides a continuous
   quality metric superior to binary pass/fail.

3. **Claim 3 (Batch-level abort):** The probe circuit mechanism saves
   hardware time by early-rejecting configurations with low pass rates.

4. **Claim 4 (Score fusion):** The α-weighted blend of HF/LF scores
   enables tuning the sensitivity–specificity trade-off.

## Reproducibility

- All scramble rotations use **deterministic seeding** (SHA-256 hash of
  circuit parameters) for exact reproducibility.
- The Aer noise model parameters are fixed in the source code.
- For real hardware, results depend on the specific backend and
  calibration state — record the backend name from `run_log.json`.

## License

QGATE Source Available Evaluation License v1.2 — see [`LICENSE`](../../packages/qgate/LICENSE).

Academic research, peer review, and internal corporate evaluation are permitted.
Commercial deployment requires a separate license.
