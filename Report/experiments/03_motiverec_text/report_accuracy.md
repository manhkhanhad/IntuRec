# E3a — MotiveRec-T accuracy (current runs)

Back to [main plan](../../main.md). Date: 2026-11-13.

## Runs
- **MotiveRec-T:** `run_collabrec.py`, text mode, supports `motive_corpus/stage2` (k = 10), β = 0.1, lr 3e-4.
  - CDs: `<ckpt>/CDs_and_Vinyl/CollabRec/text-motive_k10_beta0.1_rs1_lr0.0003_job359280`
  - Toys: `<ckpt>/Toys_and_Games/CollabRec/text-motive_k10_beta0.1_rs1_lr0.0003_job345291`
- **O0:** `CollabRec_oracle/text-micro2_k0_beta0_rs1_lr0.0003_job351003` (CDs) / `…_job350999` (Toys).
- **IntuRec:** see [E1a](../01_pilot_latent_collapse/report_geometry.md).
- **R³:** `final_result.json` of the original checkpoints.

## Results
Computed from `result/standard/meta.json` (R³ from `final_result.json`) with the same hit rule for all models
(`pilot/common.hit_index`). HR@1 = NDCG@1. Best per column in bold.

**CDs & Vinyl** (n 6157)
| Model | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| O0 (empty thought slot) | **0.0604** | 0.0968 | 0.1178 | **0.0604** | **0.0796** | **0.0864** |
| IntuRec | 0.0580 | **0.0981** | **0.1202** | 0.0580 | 0.0789 | 0.0860 |
| MotiveRec-T | 0.0578 | 0.0918 | 0.1174 | 0.0578 | 0.0755 | 0.0838 |
| LatentR³-SFT | 0.0495 | 0.0819 | 0.0989 | 0.0495 | 0.0662 | 0.0717 |
| LatentR³-RL | 0.0573 | 0.0893 | 0.1095 | 0.0573 | 0.0740 | 0.0805 |

**Toys & Games** (n 6738)
| Model | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| O0 (empty thought slot) | 0.0417 | **0.0867** | **0.1112** | 0.0417 | **0.0650** | **0.0729** |
| IntuRec | **0.0420** | 0.0833 | **0.1112** | **0.0420** | 0.0632 | 0.0721 |
| MotiveRec-T | 0.0410 | 0.0846 | 0.1072 | 0.0410 | 0.0632 | 0.0705 |
| LatentR³-SFT | 0.0377 | 0.0778 | 0.1023 | 0.0377 | 0.0580 | 0.0659 |
| LatentR³-RL | 0.0380 | 0.0784 | 0.1055 | 0.0380 | 0.0586 | 0.0674 |

The old emb run (collaborative rationales, not motives) is reference only: HR@10 / NDCG@10 0.1142 / 0.0811 (CDs),
0.1131 / 0.0715 (Toys).

The R³ re-decodes in the Pilot C wrapper differ slightly from `final_result.json`: CDs RL 0.1100 against 0.1095,
Toys SFT 0.1011 against 0.1023 HR@10. Use `final_result.json` for the main table.

## Reading
- MotiveRec-T beats R³ at every cutoff. It is below IntuRec and O0 at HR@10 and NDCG@{1,5,10} on both
  datasets, and roughly level at HR@1. The one exception is Toys HR@5, where it beats IntuRec (0.0846 vs 0.0833). This is the main open problem of the
  improvement track.
- The old emb runs (335693 / 345290) used `collab/stage2`, i.e. collaborative-rationale supports, so they are **not**
  MotiveRec-E. See [E4](../04_motiverec_emb/report_setup.md).
