# E3b — MotiveRec-T thought geometry (does motive grounding prevent collapse?)

Back to [main plan](../../main.md). Date: 2026-11-13.

## Setup
- **Thoughts:** `diagnostics/thoughts_test.npz` (keys `thought`, `query`, `row_ids`), written by
  `collab-refined-infonce/collab/latent_diagnostics.py` (submit script `collab/submit_latent_diag.sbatch` in that
  worktree).
  - Toys: job 349725, run 345291.
  - CDs: job 364681, run 359280.
- **Measures:** same as [E1a](../01_pilot_latent_collapse/report_geometry.md), via `pilot/common.collapse_stats`.

## Results
| Data | Representation | eff. rank | PR | PC1 var | PCs 90% | pair cos | cos to mean |
|---|---|---|---|---|---|---|---|
| CDs | MotiveRec-T thought | 8.78 | 5.39 | 0.353 | 8 | 0.6865 | 0.8281 |
| CDs | history query | 137.97 | 28.83 | 0.140 | 354 | 0.6857 | 0.8278 |
| Toys | MotiveRec-T thought | 15.53 | 9.56 | 0.200 | 14 | 0.5397 | 0.7348 |
| Toys | history query | 131.96 | 30.19 | 0.135 | 337 | 0.5107 | 0.7144 |

The baselines are in [E1a](../01_pilot_latent_collapse/report_geometry.md): pair cosine 0.94–1.00, rank 1.02–9.01.

**Collaborative structure (Toys, from the diagnostics summary).** Centred cosine of a test thought to training
thoughts:
- same user 0.444;
- same target 0.231;
- sharing a motive 0.100;
- random 0.006.

kNN-10 share of rows sharing a motive: 0.126, against 0.049 for random rows. Mean thought norm: 0.22 (Toys), 0.19 (CDs).

## Reading
- Motive grounding avoids the collapse. The pair cosine is close to the history query's own level, and PC1 drops to
  0.20–0.35.
- On CDs, the rank (8.78) is about the same as R³-SFT's (9.01). Use pair cosine as the primary measure.
- Suggested figure: pair cosine (x) against effective rank (y), with all models on both datasets.

**TODO**
- Geometry of MotiveRec-E ([E4](../04_motiverec_emb/report_setup.md)).
- A β = 0 run, to tell whether the motive input or the alignment loss prevents the collapse.
