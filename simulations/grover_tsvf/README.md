# Grover vs TSVF-Chaotic Grover (IBM Fez)

> **Patent notice:** The underlying methods are covered by pending patent applications.

## Objective

Test whether TSVF trajectory filtering can rescue Grover search from
hardware noise degradation at higher iteration counts, where standard
Grover's success probability collapses on real NISQ devices.

## Setup

| Parameter | Value |
|---|---|
| **Backend** | IBM Fez (156 qubits) |
| **Algorithm** | 5-qubit Grover search (marked state `|10101⟩`) |
| **Iterations** | 1–10 |
| **Shots** | 8,192 per configuration |
| **TSVF variant** | Chaotic perturbation + parity probe ancilla |
| **Date** | February 2026 |

## TSVF Approach

1. **Standard Grover:** Oracle + Diffusion operator, iterated 1–10 times
2. **TSVF-Grover:** Same + chaotic layer (random Rz/Ry/CX) + ancilla
   parity probe (controlled rotations rewarding marked-state bit pattern)
3. **Post-selection:** Accept only shots where ancilla measures `|1⟩`

## Key Results (IBM Fez Hardware)

| Iteration | P(success) std | P(success) TSVF | Ratio | Accept% |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 0.2131 | 0.1953 | 0.92× | 29.1% |
| 2 | 0.4329 | 0.3618 | 0.84× | 31.4% |
| 3 | 0.1801 | 0.4764 | 2.65× | 28.7% |
| 4 | 0.0830 | 0.6105 | **7.36×** | 25.3% |
| 5 | 0.0552 | 0.4318 | 7.82× | 22.8% |

### Headline: **7.3× TSVF advantage at iteration 4**

At low iterations (1–2), standard Grover still has strong signal and TSVF
adds overhead. At iteration 3+, hardware noise degrades the Grover
amplitude pattern, and TSVF post-selection filters for trajectories where
the marked-state amplitude survived — yielding dramatic improvement.

## Reproduction

```bash
# Aer (local simulator with noise model)
python simulations/grover_tsvf/run_grover_tsvf_experiment.py \
    --mode aer --max-iter 10 --shots 8192

# IBM Hardware
python simulations/grover_tsvf/run_grover_tsvf_experiment.py \
    --mode ibm --max-iter 10 --shots 8192
```

Requires `.secrets.json` with `ibmq_token` for IBM hardware runs.

## Results

- **IBM Fez results:** `results_20260302_064427/`
- **Aer smoke test:** `results_20260302_064201/`
- Plots: `fidelity_vs_iteration.png`, `success_ratio.png`, `depth_vs_iteration.png`
