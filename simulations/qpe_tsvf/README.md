# QPE vs TSVF-QPE Phase Estimation (IBM Fez)

> **Patent notice:** US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915

## Objective

Test whether TSVF trajectory filtering can "anchor" the phase estimate in
Quantum Phase Estimation, keeping a sharp probability spike on the correct
phase binary fraction despite hardware noise as precision qubits increase.

## Setup

| Parameter | Value |
|---|---|
| **Backend** | IBM Fez (156 qubits) |
| **Algorithm** | QPE for $U = R_z(2\pi\phi)$ with eigenphase $\phi = 1/3$ |
| **Precision qubits** | t = 3–7 |
| **Total qubits** | t + 1 (eigenstate) + 1 (ancilla) = 5–9 |
| **Shots** | 8,192 per configuration |
| **TSVF variant** | Mild perturbation + phase probe ancilla |
| **Date** | March 2026 |

## Why φ = 1/3?

The eigenphase $\phi = 1/3 = 0.\overline{01}$ in binary is irrational —
it cannot be exactly represented in any finite binary fraction. This makes
it a good stress test: even a perfect QPE circuit will have inherent
approximation error that shrinks as $2^{-t}$, and hardware noise
compounds on top of that.

## TSVF Approach

1. **Standard QPE:** Hadamard on precision register → controlled-$U^{2^k}$
   kicks → inverse QFT → measure precision qubits
2. **TSVF-QPE:** Same + mild chaotic perturbation (Rz/Ry with scale
   $\pi/(6\sqrt{t})$, sparse CZ ring) + ancilla phase probe
   (2-controlled-Ry gates rewarding correct binary fraction bits)
3. **Post-selection:** Accept only shots where ancilla measures `|1⟩`

## Key Results (IBM Fez Hardware)

| t | Fid(std) | Fid(TSVF) | Err(std) | Err(TSVF) | Ent(std) | Ent(TSVF) | Accept% |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 3 | **0.582** | 0.064 | **0.105** | 0.326 | **1.97** | 2.48 | 35.8% |
| 4 | **0.551** | 0.015 | **0.095** | 0.267 | **2.49** | 3.71 | 32.0% |
| 5 | **0.369** | 0.035 | **0.148** | 0.284 | **3.68** | 4.76 | 50.5% |
| 6 | **0.343** | 0.012 | **0.110** | 0.261 | **4.05** | 5.87 | 60.1% |
| 7 | **0.157** | 0.008 | **0.157** | 0.245 | **5.65** | 6.83 | 49.6% |

### Phase Identification

| t | Correct bits | Std best | ✓ | TSVF best | ✓ |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 3 | `011` | `011` | ✅ | `110` | ❌ |
| 4 | `0101` | `0101` | ✅ | `0010` | ❌ |
| 5 | `01011` | `01011` | ✅ | `11010` | ❌ |
| 6 | `010101` | `010101` | ✅ | `000010` | ❌ |
| 7 | `0101011` | `0101011` | ✅ | `0001110` | ❌ |

### Result: **TSVF does NOT help QPE** ❌

Standard QPE correctly identifies φ ≈ 1/3 at **all precision levels** on
IBM Fez — the hardware is good enough to preserve the phase structure even
at depth 411 (t=7). The TSVF perturbation, even at the mild scale of
$\pi/(6\sqrt{t})$, destroys the delicate phase coherence that the inverse
QFT relies on, producing near-uniform random output.

### Why TSVF Fails for QPE

QPE encodes its answer in the **phase coherence** of the precision register
after the inverse QFT. The controlled-$U^{2^k}$ gates establish precise
phase relationships between qubits, and the inverse QFT converts these into
a probability peak at the correct binary fraction.

Any unitary perturbation on the precision register — even small rotations —
disrupts these phase relationships. The inverse QFT then produces a diffuse
distribution instead of a sharp peak. Post-selection on the ancilla cannot
recover the destroyed phase information because it was lost before
measurement.

This is fundamentally different from Grover/QAOA/VQE, where the answer is
encoded in **amplitude patterns** that are robust to small perturbations.

## Reproduction

```bash
# Aer (local simulator with noise model)
python simulations/qpe_tsvf/run_qpe_tsvf_experiment.py \
    --mode aer --min-precision 3 --max-precision 7 --shots 8192

# IBM Hardware
python simulations/qpe_tsvf/run_qpe_tsvf_experiment.py \
    --mode ibm --min-precision 3 --max-precision 7 --shots 8192
```

Requires `.secrets.json` with `ibmq_token` for IBM hardware runs.

## Results

- **IBM Fez results:** `results_20260302_082720/`
- **Aer smoke test:** `results_20260302_082638/`
- Plots: `fidelity_vs_precision.png`, `phase_error_vs_precision.png`,
  `entropy_vs_precision.png`, `depth_vs_precision.png`,
  `acceptance_rate.png`, `fidelity_ratio_vs_precision.png`
- Telemetry: `qpe_tsvf_telemetry.jsonl` (per-run Galton threshold data)
