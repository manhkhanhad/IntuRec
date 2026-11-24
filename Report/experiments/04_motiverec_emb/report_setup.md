# E4 — MotiveRec-E (embedding integration): setup and results

> **Superseded (2026-11-14).** The MotiveRec-T/E runs in this report (345291 reference aside) were trained with L_align on (β = 0.1). MotiveRec is now NTP-only (β = 0); those runs' checkpoints and results were deleted and are being rerun with β = 0. Treat the β = 0.1 numbers below as invalid for MotiveRec.


Back to [main plan](../../main.md). Date: 2026-11-13.

## Data — `data/<ds>/collab/motive_corpus/stage2_emb/`
Built by `collab/build_motive_emb_support.py` (CPU, no model-code change).

**Support table.** `rationale_emb_train.npy` has two parts:
- the train rationales: the same table as `collab/stage2`, i.e. all-mpnet-base-v2 embeddings of the v5 reasons,
  centred on the train mean and normalised;
- the K prototypes appended: their "motive: reason" texts embedded with the same encoder and centred with the same
  mean.

**Supports.** `support_rows_<split>.npy` = the predicted prototype ids from `motive_corpus/stage2` + n_train.

**Alignment.** `own_rationale_{train,valid}.npy` is copied from `collab/stage2`. The alignment target and the
negatives are therefore the same as in the collaborative emb runs. The K prototype rows also enter the negative bank,
which adds less than 2%.

**Check.** Mean cosine from each prototype to its nearest train rationale: 0.635 (CDs), 0.679 (Toys). Mean
prototype-to-prototype cosine: 0.185 / 0.123.

## Runs (via JobManager, main-repo `submit_collabrec.sbatch`)
`SUPPORT_MODE=emb SUPPORT_DIR=data/<ds>/collab/motive_corpus/stage2_emb RUN_TAG=motive`, with K = 10, β = 0.1, micro 8.
- CDs: job 364727. Folder `<ckpt>/CDs_and_Vinyl/CollabRec/emb-motive_k10_beta0.1_rs1_lr0.0003_job364727`.
- Toys: job 364728, 48 GB GPUs only. Folder `<ckpt>/Toys_and_Games/CollabRec/emb-motive_k10_beta0.1_rs1_lr0.0003_job364728`.

## Results
| Data | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 | eff. rank | pair cos |
|---|---|---|---|---|---|---|---|---|
| CDs | 0.0541 | 0.0901 | 0.1127 | 0.0541 | 0.0726 | 0.0798 | pending | pending |
| Toys | 0.0393 | 0.0827 | 0.1067 | 0.0393 | 0.0617 | 0.0694 | pending | pending |

Comparison targets (all cutoffs): [E3a](../03_motiverec_text/report_accuracy.md).
Status 2026-11-13: in epoch 1 of 3. CDs is at step 50/385, Toys at step 40/422, at about 1.7 h (CDs) and 2.1 h (Toys) per
epoch.
Geometry needs a thought dump: run `collab-refined-infonce/collab/latent_diagnostics.py` on the run, as in E3b.

Progress 2026-11-13 (JobManager): CDs 364727 finished epoch 1, valid loss total 1.7621 (NTP 1.1079, rationale
6.5429). Toys 364728 finished epoch 1: total 1.2621 (NTP 0.6435, rationale 6.1863). CDs epoch 2: total 1.6239 (NTP 0.9799, rationale 6.4396); Toys epoch 2: total 1.1738 (NTP 0.5597, rationale 6.1405). CDs epoch 3: total 1.7300 (NTP 1.0799, rationale 6.5008), up from epoch 2; the "validation" loss of this code base is a train-batch loss (known IntuRec quirk), so the test decode decides. Toys in epoch 3.

CDs result (2026-11-13, recomputed from `result/standard/meta.json` with `pilot/common.hit_index`, n 6157) vs E3a:
| Model | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| O0 | 0.0604 | 0.0968 | 0.1178 | 0.0604 | 0.0796 | 0.0864 |
| IntuRec | 0.0580 | 0.0981 | 0.1202 | 0.0580 | 0.0789 | 0.0860 |
| MotiveRec-T | 0.0578 | 0.0918 | 0.1174 | 0.0578 | 0.0755 | 0.0838 |
| **MotiveRec-E** | 0.0541 | 0.0901 | 0.1127 | 0.0541 | 0.0726 | 0.0798 |

MotiveRec-E (v5 motives) is below MotiveRec-T at every cutoff on CDs (HR@10 −0.0047), and below IntuRec / O0.

Toys result (job 364728, recomputed the same way, n 6738) vs E3a:
| Model | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| O0 | 0.0417 | 0.0867 | 0.1112 | 0.0417 | 0.0650 | 0.0729 |
| IntuRec | 0.0420 | 0.0833 | 0.1112 | 0.0420 | 0.0632 | 0.0721 |
| MotiveRec-T | 0.0410 | 0.0846 | 0.1072 | 0.0410 | 0.0632 | 0.0705 |
| **MotiveRec-E** | 0.0393 | 0.0827 | 0.1067 | 0.0393 | 0.0617 | 0.0694 |

On Toys, MotiveRec-E ≈ MotiveRec-T (HR@10 −0.0005), both below IntuRec / O0. With v5 motives the integration
variant is not the bottleneck; E stays a planned ablation (E vs T), and v6 motives are tested with T first.
