# VQE vs TSVF-VQE for TFIM (IBM Fez)

> **Patent notice:** US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915

## Objective

Test whether TSVF trajectory filtering helps VQE avoid the **barren plateau**
phenomenon — the catastrophic loss of gradient signal at deeper ansatz depths
that plagues variational quantum eigensolvers on real hardware.

## Setup

| Parameter | Value |
|---|---|
| **Backend** | IBM Fez (156 qubits) |
| **Algorithm** | VQE for 4-qubit Transverse-Field Ising Model (TFIM) |
| **Hamiltonian** | $H = -\sum_i Z_i Z_{i+1} - h\sum_i X_i$, $h=1.0$ |
| **Exact ground energy** | −4.0000 |
| **Ansatz layers** | L = 1–6 (hardware-efficient: Ry + CX ladder) |
| **Shots** | 4,000 per configuration |
| **TSVF variant** | Chaotic perturbation + energy probe ancilla |
| **Date** | March 2026 |

## TSVF Approach

1. **Standard VQE:** Hardware-efficient ansatz (Ry rotations + CX entangling
   ladder), L layers, random initial parameters
2. **TSVF-VQE:** Same + chaotic perturbation on ansatz qubits + ancilla
   energy probe (controlled rotations that reward low-energy bitstrings)
3. **Post-selection:** Accept only shots where ancilla measures `|1⟩`

## Key Results (IBM Fez Hardware)

| L (layers) | Energy std | Energy TSVF | Gap std | Gap TSVF | Δ Gap |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | −2.921 | −2.977 | 1.079 | 1.023 | 0.056 |
| 2 | −2.804 | −2.880 | 1.196 | 1.120 | 0.076 |
| 3 | −1.602 | −2.709 | **2.398** | **1.291** | **1.107** |
| 4 | −1.468 | −2.501 | 2.532 | 1.499 | 1.033 |
| 5 | −1.321 | −2.389 | 2.679 | 1.611 | 1.068 |
| 6 | −1.198 | −2.254 | 2.802 | 1.746 | 1.056 |

### Headline: **Barren Plateau Avoidance at L=3**

Standard VQE hits a dramatic barren plateau at L=3 — energy jumps from
−2.804 (L=2) to −1.602 (L=3), a loss of ~1.2 energy units as the gradient
signal vanishes in the deeper circuit. TSVF-VQE maintains smooth energy
descent through L=3 (−2.880 → −2.709), demonstrating that trajectory
filtering selects for low-energy execution paths even when the average
trajectory has lost gradient information.

The energy gap between standard and TSVF at L=3 is **1.107 units** — the
largest advantage point, confirming that TSVF is most beneficial exactly
where standard VQE fails.

## Reproduction

```bash
# Aer (local simulator with noise model)
python simulations/vqe_tsvf/run_vqe_tsvf_experiment.py \
    --mode aer --max-layers 6 --shots 4000

# IBM Hardware
python simulations/vqe_tsvf/run_vqe_tsvf_experiment.py \
    --mode ibm --max-layers 6 --shots 4000
```

Requires `.secrets.json` with `ibmq_token` for IBM hardware runs.

## Results

- **IBM Fez results:** `results_20260302_080312/`
- **Aer smoke test:** `results_20260302_080003/`
- Plots: `energy_vs_layers.png`, `energy_gap_vs_layers.png`, `variance_vs_layers.png`
