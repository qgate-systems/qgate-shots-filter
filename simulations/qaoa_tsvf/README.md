# QAOA vs TSVF-QAOA MaxCut (IBM Torino)

> **Patent notice:** US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915

## Objective

Test whether TSVF trajectory filtering improves QAOA MaxCut performance
on real hardware, particularly at shallow circuit depths (low p) where
hardware noise has the most severe impact on variational quality.

## Setup

| Parameter | Value |
|---|---|
| **Backend** | IBM Torino (133 qubits) |
| **Algorithm** | QAOA for MaxCut on a 6-node random graph |
| **Layers** | p = 1–5 |
| **Shots** | 2,000 per configuration |
| **TSVF variant** | Chaotic perturbation + cut-quality probe ancilla |
| **Date** | February 2026 |

## TSVF Approach

1. **Standard QAOA:** Cost layer (ZZ interactions from graph edges) +
   Mixer layer (Rx rotations), repeated p times
2. **TSVF-QAOA:** Same + chaotic perturbation + ancilla probe that
   rewards bitstrings with high cut fractions via controlled-Ry gates
3. **Post-selection:** Accept only shots where ancilla measures `|1⟩`

## Key Results (IBM Torino Hardware)

| p (layers) | AR std | AR TSVF | Ratio | Accept% |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 0.4268 | 0.8029 | **1.88×** | 33.5% |
| 2 | 0.7036 | 0.7024 | 1.00× | 32.0% |
| 3 | 0.6975 | 0.6987 | 1.00× | 35.2% |
| 4 | 0.6841 | 0.6912 | 1.01× | 34.8% |
| 5 | 0.6753 | 0.6802 | 1.01× | 36.1% |

### Headline: **1.88× TSVF advantage at p=1**

At p=1 (shallowest depth), hardware noise most severely degrades the
single QAOA layer. TSVF post-selection nearly doubles the approximation
ratio. At higher p, the variational ansatz has enough expressivity to
partially self-correct, so the TSVF advantage narrows.

## Reproduction

```bash
# Aer (local simulator with noise model)
python simulations/qaoa_tsvf/run_qaoa_tsvf_experiment.py \
    --mode aer --max-layers 5 --shots 4000

# IBM Hardware
python simulations/qaoa_tsvf/run_qaoa_tsvf_experiment.py \
    --mode ibm --max-layers 5 --shots 2000
```

Requires `.secrets.json` with `ibmq_token` for IBM hardware runs.

## Results

- **IBM Torino results:** `results_20260302_074221/`
- **Aer smoke test:** `results_20260302_074143/`
- Plots: `approx_ratio_vs_layers.png`, `cut_value_vs_layers.png`, `depth_vs_layers.png`
