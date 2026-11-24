# E5c — Joint training of motive selector and reader (Atlas / RA-DIT style): existing result

Back to [main plan](../../main.md). Date: 2026-11-13. Source: `collab-grpo-select` worktree,
`collab/RESEARCH_LOG_grpo.md` Phase 4 (2026-11-12). Code: `collab/joint_reader_selector.py` in that worktree.

## Method
- **Loss:** −log Σ_{m∈S} π_S(m) · p(target | history, [m, m]), a marginal likelihood over a candidate set
  S = {null} ∪ selector top-3.
- **Reader:** initialised from the O2 model 351001 (oracle-trained, β-InfoNCE), lr 1e-5.
- **Selector:** phase-1 features plus a null logit. It starts at the retriever distribution and picks the top-1
  motive for 0.83 of rows.
- **Training:** one pass, 32 rows per step. Job 364179 (`joint_k4`); decode of its own reader + picks: job 364329.
- Toys only.

## Results, Toys (test)
| model | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| joint reader + selector (364329) | 0.0401 | 0.0800 | 0.1042 | 0.0401 | 0.0605 | 0.0683 |
| retriever top-1 + reader 351001 | 0.0405 | 0.0766 | 0.0994 | 0.0405 | — | 0.0662 |
| IntuRec (348597) | 0.0420 | 0.0833 | 0.1112 | 0.0420 | 0.0632 | 0.0721 |
| curriculum (356575, [E5b](report_curriculum.md)) | 0.0421 | 0.0852 | 0.1126 | 0.0421 | 0.0643 | 0.0732 |

## Reading
- **Best selection-based decode,** +0.0048 HR@10 over retriever top-1 with the same reader.
- **Still below IntuRec** (−0.0070 HR@10) and O0.
- **The selector abstains:** it picks "no motive" for 74% of test rows (null 0.742, top-1 0.234, 26 distinct motives).
- **The reader starts weak:** 351001 drops below O0 when it gets no motive or a wrong one, so part of the joint
  gain is recovery from a weak start.
- **Untried idea:** joint training starting from a reader that is robust to noise, i.e. the curriculum checkpoint.
