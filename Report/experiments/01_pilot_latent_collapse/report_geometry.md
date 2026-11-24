# E1a — Latent collapse: geometry of test thoughts (motivation)

Back to [main plan](../../main.md). Date: 2026-11-13.

## Question
Are the latent thoughts of existing latent-reasoning recommenders diverse across users?

## Setup
- **Models:** the original codebases and checkpoints, all Qwen2.5-1.5B on the same test split.
  - LatentR³ SFT: `R3_ori/output_dir/<ds>/latent`
  - LatentR³ RL: `R3_ori/output_dir/<ds>/nomul_attn_lr_5e-5`
  - IntuRec F2 micro 2: `IntuRec_original/IntuRecF2_micro2_job349506` (CDs), `…_job348597` (Toys)
- **Thoughts:**
  - IntuRec: `diagnostics/thoughts_test.npy`, from `collab/inturec_latent_diagnostics.py`.
  - R³: `MotiveRec_pilot/r3/<run>/evidence/<ds>/pilot/test/thoughts.pt`, extracted by `pilot/pilot_c_r3.py`, which
    wraps R³'s unmodified `experiment.py`.
- **Measures** (`pilot/common.collapse_stats`):
  - effective rank: the entropy-exponential of the normalised singular values of the centred thought matrix;
  - participation ratio (PR);
  - PC1 variance share;
  - number of PCs needed for 90% of the variance;
  - mean cosine of random pairs (uncentred);
  - cosine to the mean thought.
- **Command:** `python pilot/pilot_bc.py collapse`. Output: `<ckpt>/<ds>/MotiveRec_pilot/pilot_c.json`.

## Results
| Data | Model | eff. rank | PR | PC1 var | PCs 90% | pair cos | cos to mean |
|---|---|---|---|---|---|---|---|
| CDs | LatentR³-SFT | 9.01 | 6.23 | 0.307 | 8 | 0.9444 | 0.9718 |
| CDs | LatentR³-RL | 1.55 | 1.22 | 0.905 | 1 | 0.9947 | 0.9974 |
| CDs | IntuRec | 1.02 | 1.00 | 0.998 | 1 | 0.9524 | 0.9686 |
| Toys | LatentR³-SFT | 2.27 | — | 0.631 | — | 1.0000 | — |
| Toys | LatentR³-RL | 2.30 | — | 0.607 | — | 1.0000 | — |
| Toys | IntuRec | 1.05 | 1.02 | 0.992 | 1 | 0.9955 | 0.9976 |

Reference, the backbone hidden state at the thought position (history query, measured on the MotiveRec-T backbone):

| Data | eff. rank | PC1 var | pair cos |
|---|---|---|---|
| CDs | 137.97 | 0.140 | 0.6857 |
| Toys | 131.96 | 0.135 | 0.5107 |

## Reading
- Every model collapses on both datasets: pair cosine is 0.94–1.00.
- On Toys, R³ maps every user to one constant vector (pair cosine 1.0000). The remaining rank is fp16 noise.
- RL makes the collapse worse on CDs (rank 9.01 → 1.55).
- IntuRec is close to one-dimensional (PC1 ≥ 0.99).
- **CDs R³-SFT caveat:** its rank (9.01) is about the same as MotiveRec-T's (8.78). On CDs, argue with pair cosine,
  cosine to mean and PC1, not with rank alone.
- **TODO:** the history-query reference comes from the MotiveRec backbone. Measure R³'s and IntuRec's own history
  query before claiming that the information is lost when the latent is formed.
