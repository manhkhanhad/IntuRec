# E5b — Curriculum training (clean → noisy motives): existing results

Back to [main plan](../../main.md). Date: 2026-11-13. Source: main-repo `collab/RESEARCH_LOG.md`, entries for
2026-11-07 to 2026-11-09. Code: `collab-refined-infonce` worktree (`--noisy_precision_start`, gated readout).

## Method
- **Train inputs:** the k = 2 motive slots get the row's true motives with probability p, otherwise the retriever's
  top non-true prediction.
- **Schedule:** p decreases per epoch, 1.0 → 0.625 → 0.25. The model first learns to *use* motives on clean input
  (as with oracle input), then learns to cope with noise.
- **Readout:** gated. Each slot gets a sigmoid gate trained with selection BCE, label = "slot holds a true motive",
  weight 1.0.
- **Other settings:** β = 0, no InfoNCE, micro-batch 2, text mode.
- **Test inputs:** the retriever's motives only (deployable). True motives are used only as training inputs.
- **Best checkpoint:** epoch 2, trained at p = 0.625.

## Runs (Toys only)
- 356575: `CollabRec_oracle/text-micro2-noisy1.0to0.25-gated1.0_k2_beta0_rs1_lr0.0003_job356575`
- 357942: rerun of the same setting.

## Results, Toys (test, retriever motives)
| model | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| **curriculum 356575, cls top-10 (retest 360215)** | **0.0421** | **0.0852** | **0.1126** | **0.0421** | **0.0643** | **0.0732** |
| curriculum 357942, retriever top-2 | — | — | 0.1121 | — | — | 0.0740 |
| IntuRec (348597) | 0.0420 | 0.0833 | 0.1112 | 0.0420 | 0.0632 | 0.0721 |
| MotiveRec-T (345291) | 0.0410 | 0.0846 | 0.1072 | 0.0410 | 0.0632 | 0.0705 |
| O0 (350999) | 0.0417 | 0.0867 | 0.1112 | 0.0417 | 0.0650 | 0.0729 |

## Reading
- **The curriculum is the best deployable motive model on Toys.** It is the only one at or above IntuRec on all six
  metrics: HR@10 +0.0014, NDCG@10 +0.0011, HR@5 +0.0019. Two runs agree (0.1119–0.1126 HR@10).
- **The margin is within seed spread (~0.003),** and the earlier log rated it "= O0". It needs seeds and CDs to
  become a claim.
- **Not run yet:**
  - CDs;
  - k = 10;
  - combining it with the β = 0.1 alignment (the MotiveRec recipe).
