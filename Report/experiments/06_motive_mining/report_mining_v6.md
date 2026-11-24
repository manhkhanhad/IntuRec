# E8b — Motive mining v6: need → item kind, from GEARec reasoning (Toys)

> **Superseded (2026-11-14).** The MotiveRec-T/E runs in this report (345291 reference aside) were trained with L_align on (β = 0.1). MotiveRec is now NTP-only (β = 0); those runs' checkpoints and results were deleted and are being rerun with β = 0. Treat the β = 0.1 numbers below as invalid for MotiveRec.


Back to [main plan](../../main.md). Date: 2026-11-13. Job 364815 (2 h 57 min, no preemption).
Code: `collab/build_motive_corpus_v6.py`, `collab/submit_motive_corpus_v6.sbatch`. Corpus:
`data/Toys_and_Games/collab/motive_corpus_v6/{P1000,P2000}/`. Checks: `pilot/mining_diagnostics.py`,
`pilot/reasoning_reachability.py 1500 <corpus>`.

## Build
- Source: GEARec v2 best-of-8 reasoning. Extraction by Qwen3-8B, 0 errors, 2 motives per row (106,307 train motives,
  102,915 distinct statements).
- Rows without reasoning get no motive: train 1.37%, valid 6.78%, test 4.14% (almost all with a single history item;
  accepted by the author).
- Louvain → P1000: K = 998, rows per prototype 9 / 99 / 270 (min / median / max), largest share 0.26%.
  P2000: K = 1,907, 8 / 52 / 222, largest share 0.21%. (v5: largest share 7.5%; 41 prototypes hold half the labels.)
- Examples: "collecting detailed superhero representations → Marvel-themed action figures"; "building a train-themed
  play system for a young child → wooden train track and accessory sets"; "sensory and developmental play during bath
  time → bath time sensory toys".

## Results vs v5 (Toys test)
| | v5 (K 881) | v6 P1000 (K 998) | v6 P2000 (K 1907) |
|---|---|---|---|
| prototypes holding 50% / 80% of labels | 41 / 269 | 325 / 634 | 636 / 1226 |
| popularity R@1 / R@10 | 0.079 / 0.257 | 0.003 / 0.026 | 0.002 / 0.014 |
| user's past motives R@1 / R@10 | 0.078 / 0.256 | 0.022 / 0.082 | 0.017 / 0.055 |
| history-item profile R@1 / R@10 | 0.090 / 0.265 | 0.059 / 0.215 | 0.042 / 0.150 |
| target-item profile (oracle) R@1 / R@10 | 0.146 / 0.401 | 0.090 / 0.240 | 0.067 / 0.178 |
| **history R@10 ÷ popularity R@10** | **1.03** | **8.2** | **11.1** |
| **history R@10 ÷ target-item oracle R@10** | 0.66 | **0.90** | 0.84 |
| next item from true motives (oracle), HR@10 / NDCG@10 | 0.1383 / 0.0765 | **0.1808 / 0.1024** | 0.1747 / 0.1036 |
| prototype text → target, top-10 / top-20 / top-100 (mxbai) | 0.031 / 0.059 / 0.140 | 0.050 / 0.094 / 0.275 | 0.055 / 0.096 / 0.282 |
| prototype text, median target rank | 1022 | 373 | 368 |

For reference, on the same rows: v5 reason sentence top-20 0.287; GEARec best reasoning 0.245.

## Reading
1. **v6 motives are predictable from the history; v5 motives were not.** A simple history-item profile finds the true
   motive in the top 10 8× more often than popularity (v5: 1.03×), and almost as often as knowing the target item
   (0.90 of the oracle; v5: 0.66). Absolute recall is lower than v5 only because v6 has no dominant generic motives
   (the popularity baseline falls from 0.257 to 0.026).
2. **v6 motives point to the item.** The prototype text reaches the target twice as often at top-100 (0.275 vs 0.140)
   and the median rank falls from 1022 to 373. The true motives alone rank the next item at HR@10 0.181 (v5 0.138).
3. **Some information is still lost in clustering.** Prototype top-20 0.094 is below the per-row GEARec reasoning
   (0.245). Finer clustering (P2000) barely helps reachability and lowers predictability.
4. **Choice: P1000** (best predictability, same usefulness as P2000).
5. Note: redundancy counts (centroid pairs > 0.8: 1,518 for P1000) are not comparable with v5 (different embedding
   centring and K).

Next: aligner vs plain retriever on v6 P1000 (fixed item scorer), then supports for MotiveRec-T.

## Retriever vs aligner on v6 P1000 (Toys, CPU)
Retrieval of the v6 labels (test):
| model | R@1 | R@5 | R@10 | R@20 | MRR |
|---|---|---|---|---|---|
| popularity | 0.0033 | 0.0165 | 0.0263 | 0.0445 | 0.0267 |
| history-item profile | 0.0588 | 0.1558 | 0.2152 | 0.2908 | 0.1968 |
| user's past motives | 0.0215 | 0.0626 | 0.0818 | 0.1030 | 0.0827 |
| cls retriever (MLP) | 0.0574 | 0.1647 | 0.2368 | 0.3338 | 0.2058 |
| aligner A (ce only) | 0.0595 | 0.1689 | 0.2399 | 0.3342 | 0.2097 |
| aligner D (ce + hm + mi + match) | 0.0563 | 0.1683 | 0.2401 | 0.3365 | 0.2058 |

Next item from the retrieved motives, fixed scorer Σ_k π_k p(item | motive k) with v6 links, HR@10 / NDCG@10:
| model | valid | test |
|---|---|---|
| **v5 cls retriever (v5 motives, v5 links; E8d)** | 0.0530 / 0.0320 | 0.0439 / 0.0244 |
| **v6 cls retriever** | **0.0761 / 0.0469** | **0.0695 / 0.0407** |
| v6 aligner A | 0.0720 / 0.0438 | 0.0626 / 0.0357 |
| v6 aligner D | 0.0761 / 0.0459 | 0.0623 / 0.0364 |
| v6 relabel γ = 1, round 0 / 1 / 2 | 0.0677 / 0.0721 / 0.0690 | 0.0603 / 0.0616 / 0.0622 |

Relabel train-label stats: 998 motives in use, largest share 0.6%; 47% of new top-1 labels equal an original v6 label.

**Reading.** v6 retrieved motives point to the next item far better than v5 (test HR@10 0.0695 vs 0.0439, +58%), with
the same plain retriever. The aligner and the relabelling do not beat the plain retriever on v6 either (test 0.0623 vs
0.0695), so by the rule set before the run they are **dropped**. MotiveRec keeps the cls retriever; the gain comes
from the v6 mining.

## MotiveRec-T with v6 motives (E8f, job 365120, done 2026-11-14)
- Supports: `motive_corpus_v6/P1000/stage2/` = top-10 prototypes from the cls retriever (5-fold cross-fitted for
  train rows; train OOF MRR 0.240), texts "need -> item kind: reason". True motive in the top 10: train 0.449,
  valid 0.410, test 0.396.
- Run: `submit_collabrec.sbatch` with `SUPPORT_DIR=…/motive_corpus_v6/P1000/stage2 RUN_TAG=motivev6`, otherwise the
  same as the v5 MotiveRec-T run 345291 (text, K 10, β 0.1, micro 8), so v5 → v6 is the only change.
- Results (Toys test, 6,738 rows, epoch-3 model; valid NTP by epoch 0.6356 / 0.5551 / 0.5805):

| Model | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| MotiveRec-T v6 (365120) | 0.0401 | 0.0818 | 0.1113 | 0.0401 | 0.0613 | 0.0709 |
| MotiveRec-T v5 (345291) | 0.0410 | 0.0846 | 0.1072 | 0.0410 | 0.0632 | 0.0705 |
| MotiveRec-E v5 (364728) | 0.0393 | 0.0827 | 0.1067 | 0.0393 | 0.0617 | 0.0694 |
| IntuRec | 0.0420 | 0.0833 | 0.1112 | 0.0420 | 0.0632 | 0.0721 |
| CollabRec k=3 (364239) | 0.0441 | 0.0867 | 0.1123 | 0.0441 | 0.0662 | 0.0744 |

- Reading:
  - v6 raises HR@10 by 0.0041 over v5, which ties IntuRec. It lowers @1 and @5 (HR@5 −0.0028), so NDCG@10 is flat.
  - All gaps are about one binomial standard error (≈ 0.004 at HR@10 on 6,738 rows), and each model has a single seed.
  - So the much better motive labels (E8b) have not yet turned into a clear recommendation gain. Two likely causes:
    - The supports reach the model through the retriever, and the true motive is in its top 10 for only 0.396 of test rows.
    - The test used the epoch-3 model, although validation NTP was best at epoch 2.
  - MotiveRec still does not beat IntuRec at @1/@5. CollabRec k=3 remains the best Toys run.

## MotiveRec-T v6 with 3 supports (E8i, jobs 365345 / 365351, done 2026-11-14)
- Same as 365120 except SUPPORT_K = 3.
  - **cls:** the first 3 columns of `P1000/stage2`.
  - **aligner:** aligner v2 user+item (w_inter 0), 5-fold cross-fitted for train rows, in `P1000/stage2_aligner`.
- True motive in the top 3 (train / valid / test):
  - cls: 0.264 / 0.225 / 0.225
  - aligner: 0.294 / 0.245 / 0.237
  - The aligner's train-test gap is larger because item-motive texts include held-out rows (mild leak).

| Model (Toys test) | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| MotiveRec-T v6, cls, K=3 (365345) | **0.0424** | **0.0847** | 0.1097 | **0.0424** | **0.0642** | **0.0722** |
| MotiveRec-T v6, aligner, K=3 (365351) | 0.0405 | 0.0824 | 0.1070 | 0.0405 | 0.0621 | 0.0700 |
| MotiveRec-T v6, cls, K=10 (365120) | 0.0401 | 0.0818 | **0.1113** | 0.0401 | 0.0613 | 0.0709 |
| IntuRec | 0.0420 | 0.0833 | 0.1112 | 0.0420 | 0.0632 | 0.0721 |

- Reading:
  - K=3 with the cls retriever is the first MotiveRec-T run at or above IntuRec on HR@1/5 and NDCG@1/5/10. It is 0.0015 below IntuRec on HR@10.
  - The aligner's supports hit the true motive more often, yet they give lower test scores on every metric (HR@10 −0.0027 vs cls K=3). Better label retrieval again does not turn into better recommendation; the train-side leak may contribute.
  - All gaps are about one standard error (single seed). Validation was best at epoch 2 in all runs, but tests use the epoch-3 model.
