# E2b — Does the latent thought encode the purchase motive? (linear probe)

Back to [main plan](../../main.md). Date: 2026-11-13. Code: `python pilot/pilot_bc.py probe`. Output:
`<ckpt>/<ds>/MotiveRec_pilot/pilot_b.json`.

## Setup
- **Probe:** a linear layer with 13 sigmoid outputs (the motive facets), trained on standardised features with
  5-fold GroupKFold by user. Weight decay is the best of {1e-4, 1e-3, 1e-2}.
- **Metric:** macro ROC-AUC over facets.
- **Features:** Gaussian noise (floor), the gte-large history embedding, each model's thought, and history + thought.

## Results (macro AUC)
| representation | CDs | Toys |
|---|---|---|
| noise (floor) | 0.5031 | 0.5084 |
| history text (gte) | 0.6237 | 0.6632 |
| IntuRec thought | 0.5855 | 0.6627 |
| LatentR³-SFT thought | 0.5372 | 0.4997 |
| LatentR³-RL thought | 0.5178 | 0.5026 |
| history + IntuRec thought | 0.6224 | 0.6649 |
| history + R³-SFT thought | 0.6224 | 0.6601 |
| history + R³-RL thought | 0.6175 | 0.6606 |

## Reading
- R³ thoughts carry almost no motive information.
- IntuRec's thought, despite rank ≈ 1, can be decoded to about the history's level. That information sits in
  tiny-variance directions (the probe standardises its inputs).
- No thought adds information beyond the history (Δ −0.0062 to +0.0017).
- The history alone predicts the motive only weakly (0.62–0.66), which fits the retriever ceiling.
- **TODO:** run the same probe on the MotiveRec thoughts (`thoughts_test.npz` of the MotiveRec runs).
