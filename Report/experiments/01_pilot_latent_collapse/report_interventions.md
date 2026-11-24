# E1b — Thought interventions (development only, NOT for the MotiveRec paper)

Back to [main plan](../../main.md). Date: 2026-11-13.
Author decision: interventions are kept out of the MotiveRec paper and will be addressed in a separate paper. They
are recorded here so that later sessions know them.

## Setup
Each test row is decoded with its thought replaced, and the change in HR@10 against the raw model is reported with
a paired-bootstrap 95% CI.

**R³** (`pilot/pilot_c_r3.py` + `pilot/submit_pilot_c.sbatch`; jobs 364570 / 364571 CDs, 364572 / 364573 Toys):
- `mask`: the thought is set to zero.
- `mean`: the train-mean thought.
- `shuffle_user` / `shuffle_target`: the thought of a training row with a different user / a different target.

**IntuRec** (existing decodes 355229 / 355228):
- `mask`, `mean` (test-mean thought), `shuffle`;
- `oracle_intuitions`: the target title put into the intuition list;
- `remove_target`: the target taken out of the list.

## Results — ΔHR@10 vs raw [95% CI]
**CDs**
| Model | raw | mask | mean | shuffle |
|---|---|---|---|---|
| LatentR³-SFT | 0.0989 | −0.0258 [−0.0313, −0.0210] | +0.0005 [−0.0005, +0.0015] | +0.0008 [−0.0003, +0.0019] |
| LatentR³-RL | 0.1100 | −0.0369 [−0.0429, −0.0315] | +0.0005 [−0.0006, +0.0018] | +0.0002 [−0.0013, +0.0016] |
| IntuRec | 0.1191 | −0.0050 [−0.0076, −0.0026] | −0.0029 [−0.0054, −0.0005] | −0.0016 [−0.0041, +0.0006] |

IntuRec on CDs also: target put in +0.0000 [−0.0016, +0.0016]; target removed −0.0016 [−0.0037, +0.0005].

**Toys** (HR@10 / NDCG@10)
| Model | raw | mask | mean | shuffle |
|---|---|---|---|---|
| LatentR³-SFT | 0.1011 / 0.0654 | 0.0938 / 0.0525 | 0.1014 / 0.0655 | 0.1017 / 0.0655 (user) |
| LatentR³-RL | 0.1049 / 0.0673 | 0.0938 / 0.0525 | 0.1048 / 0.0673 | 0.1055 / 0.0674 (user) |

IntuRec on Toys (raw 0.1119), ΔHR@10:
- mask −0.0013 [−0.0031, +0.0004]
- mean −0.0010
- shuffle −0.0016 [−0.0031, −0.0001]
- target put in −0.0013
- target removed −0.0010

Toys R³, ΔHR@10 [95% CI] (`pilot/pilot_bc.py collapse Toys_and_Games`, all jobs done):
| Model | raw | mask | mean | shuffle (user = target) |
|---|---|---|---|---|
| LatentR³-SFT | 0.1011 | −0.0073 [−0.0113, −0.0034] | +0.0003 [−0.0009, +0.0015] | +0.0006 [−0.0004, +0.0018] |
| LatentR³-RL | 0.1049 | −0.0111 [−0.0153, −0.0071] | −0.0001 [−0.0013, +0.0010] | +0.0006 [−0.0003, +0.0016] |

## Notes
- **SFT and RL `mask` results are identical to every digit, as expected.** A tensor comparison shows that only 6 of
  344 tensors differ between the SFT and RL checkpoints, all in the pooling module
  `attention.{query,key,value}.{weight,bias}`. R³'s RL stage trains only that module, so zeroing the thought gives
  the same model.
- **The whole R³ RL gain passes through a near-constant vector,** because the mean thought matches the per-user one.
- **`shuffle_target` equals `shuffle_user` on 100% of rows,** by construction in R³'s `experiment.py` (lines
  641–650):
  - both modes start from the same hashed donor index, (sample_id·104729 + seed·7919) mod n_train;
  - they step forward only on a user or target collision, which is rare.

  Report one shuffle row.
- **IntuRec's `mean` is the test mean; R³'s is the train mean.** Re-decode IntuRec with the train mean for a matched
  table if this is ever used.
- **MotiveRec-T on Toys** (`collab-refined-infonce/collab/latent_diagnostics.py`, job 349725; raw 0.1101), ΔHR@10:
  - mask +0.0004 [−0.0012, +0.0019]
  - mean −0.0007
  - shuffle −0.0006
  - swap motives −0.0004
  - random motives −0.0007
  - (dev oracle) true motives +0.0001

  All CIs include 0, so the decoder does not use the thought.
- **MotiveRec-T on CDs** (job 364681, run job359280; raw HR@10 0.1069), ΔHR@10 [95% CI]:
  - mask −0.0021 [−0.0042, +0.0000]; ΔNDCG@10 −0.0013 [−0.0023, −0.0002]
  - mean −0.0005 [−0.0021, +0.0010]
  - shuffle −0.0010 [−0.0029, +0.0010]
  - swap motives −0.0006 [−0.0023, +0.0010]
  - random motives −0.0003 [−0.0019, +0.0011]

  Only mask reaches the CI edge; motive content (swap / random) has no measurable effect, as on Toys. Note: the
  diagnostics decode gives raw HR@10 0.1069, below the run's own test decode (0.1174, E3a); same model, so the
  decode settings of `latent_diagnostics.py` differ. Compare interventions only within this table.
  Geometry of the refined thought (CDs): effective rank 38.7; centred cosine same user 0.455, same target 0.228,
  same motive 0.122, random ≈ 0.

## Toys R³, all cutoffs (JobManager, 2026-11-13)
Format: HR@1 / HR@5 / HR@10 | NDCG@1 / NDCG@5 / NDCG@10

| model | raw | mask | mean | shuffle_user |
|---|---|---|---|---|
| LatentR³-SFT | .0374 / .0772 / .1011 \| .0374 / .0576 / .0654 | .0211 / .0640 / .0938 \| .0211 / .0429 / .0525 | .0373 / .0776 / .1014 \| .0373 / .0578 / .0655 | .0373 / .0773 / .1017 \| .0373 / .0576 / .0655 |
| LatentR³-RL | .0383 / .0776 / .1049 \| .0383 / .0584 / .0673 | .0211 / .0640 / .0938 \| .0211 / .0429 / .0525 | .0386 / .0776 / .1048 \| .0386 / .0585 / .0673 | .0378 / .0782 / .1055 \| .0378 / .0585 / .0674 |

`shuffle_user` (jobs 364572 / 364573) is within ±0.0006 of raw at every cutoff, the same as on CDs. RL
`shuffle_target` (364573, completed): .0378 / .0782 / .1055 | .0378 / .0585 / .0674, identical to `shuffle_user` as
expected (see Notes). SFT `shuffle_target` (364572): .0373 / .0773 / .1017 | .0373 / .0576 / .0655, also identical.
