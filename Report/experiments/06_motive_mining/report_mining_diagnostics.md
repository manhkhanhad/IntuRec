# E8a — Motive mining: diagnostics of the current vocabulary

Back to [main plan](../../main.md). Date: 2026-11-13. CPU only.
Code: `pilot/mining_diagnostics.py`, `pilot/mining_merge_sweep.py`. Outputs:
`<ckpt>/<ds>/MotiveRec_pilot/mining_diag_motive_corpus.json`, `mining_merge_sweep.json`.

## Question
Is the mined vocabulary (§2.1) a good target for retrieval and a useful input for the recommender? A good motive
vocabulary should be:
- **predictable from the history**, so the retriever can find it without the target;
- **informative about the next item**, so it helps the recommender.

## Measures (test rows; all statistics built from train labels only)
- **Redundancy:** number of prototype pairs whose centroid cosine is above 0.7 / 0.8 / 0.9.
- **Concentration:** number of prototypes that cover 50% / 80% of the label occurrences.
- **Motive prediction, Recall@1 / Recall@10 of the true motives:**
  - popularity prior;
  - the user's own past motives;
  - the motive profile of the history items (target-free, close to what the retriever sees);
  - the target item's motive profile (oracle side information).
- **Usefulness:** HR@10 / NDCG@10 of ranking the next item by p(item | true motives), against item popularity.

## Results — current vocabulary
| | CDs | Toys |
|---|---|---|
| K | 859 | 881 |
| pairs with cos > 0.7 / 0.8 / 0.9 | 1433 / 371 / 3 | 1319 / 303 / 3 |
| prototypes covering 50% / 80% | 29 / 233 | 41 / 269 |
| popularity R@1 / R@10 | 0.061 / 0.384 | 0.079 / 0.257 |
| user's past motives R@1 / R@10 | 0.068 / 0.372 | 0.078 / 0.256 |
| history-item profile R@1 / R@10 | 0.083 / 0.362 | 0.090 / 0.265 |
| target-item profile (oracle) R@1 / R@10 | 0.114 / 0.412 | 0.146 / 0.401 |
| next item from popularity, HR@10 / NDCG@10 | 0.0192 / 0.0101 | 0.0137 / 0.0083 |
| next item from true motives (oracle), HR@10 / NDCG@10 | 0.0757 / 0.0402 | 0.1383 / 0.0765 |

## Coarsening sweep (Toys; CDs shows the same pattern)
Average-linkage merging of the prototypes on semantic (Qwen3 centroid), behavioural (target-item distribution) or
mixed similarity:

| merge | K | pop R@10 | hist R@10 | oracle next-item HR@10 |
|---|---|---|---|---|
| none | 881 | 0.257 | 0.265 | 0.1383 |
| semantic | 400 | 0.463 | 0.474 | 0.1121 |
| semantic | 100 | 0.693 | 0.708 | 0.0754 |
| behavioural | 400 | 0.517 | 0.529 | 0.1125 |
| behavioural | 100 | 0.776 | 0.755 | 0.0793 |

Removing the most frequent (generic) prototypes raises the lift over popularity (Toys, drop 30: history R@10 0.123
against popularity 0.085), but it removes labels from 23% of test rows and lowers the oracle next-item HR@10 to
0.1202.

## Continuous check (all-mpnet rationale embeddings, centred)
Mean cosine between a test row's rationale and the mean rationale of a group of train rows:

| group | CDs | Toys |
|---|---|---|
| same user's past rows | 0.148 | 0.125 |
| random user | 0.001 | 0.002 |
| other buyers of the history items (target-free) | 0.182 | 0.208 |
| other buyers of the target item (oracle) | 0.264 | 0.432 |

## Reading
1. **The discrete labels carry almost no target-free signal.** From the history, the true motive is predicted no
   better than by popularity (R@10 0.265 vs 0.257 on Toys, 0.362 vs 0.384 on CDs). The user's own past motives are
   no better either. This caps the retriever: its MRR gain over the prior (§2.2) is small for this reason.
2. **The signal exists in the continuous rationale space.** The history items' other buyers give cosine 0.18–0.21,
   against 0.00 for a random user. Hard 0–2 labels over ~870 prototypes discard it.
3. **Synonym merging is not the fix.** Merging makes labels easier to hit (R@10 rises), but the history lift over
   popularity stays at about 0.01, and the oracle usefulness falls steadily (0.138 → 0.075 at K = 100).
4. **The vocabulary is dominated by generic motives.** 29–41 prototypes hold half of all labels ("complete
   collection", "gift for special occasion", "nostalgic connection", "value for money"). They are the easiest to
   predict and the least informative.
5. **The rationale is mostly decided by the item, then by the user.** Same target item 0.26–0.43, same user
   0.12–0.15. Part of each motive describes *what* was bought rather than *why*.
6. **Oracle motives are informative.** On Toys, true motives alone rank the next item at HR@10 0.138, 10× popularity.
   The mining problem is predictability and label noise, not the lack of information.

## Implications for the redesign (E8b, proposed)
- **Soft labels:** replace hard argmax labels with a distribution over prototypes (similarity of the row's rationale
  to every prototype), so the retriever learns graded targets and keeps the continuous signal.
- **Utility-scored vocabulary:** score each prototype by its history lift over popularity and by its item
  informativeness, and down-weight or drop generic, unpredictable ones from the support slots.
- **Persistent vs. situational motives:** separate motives that recur for a user from one-off triggers (gift,
  occasion). Only the first kind is predictable from the history.

## Addendum — does the motive text point to the target item? (Toys, 1,500 random test rows)
Code: `pilot/reasoning_reachability.py`. Same check as GEARec's V3: embed the text alone with mxbai-embed-large and
rank the target among all items (history items masked). Higher is better; random ≈ 0.

| text | top-10 | top-20 | top-100 | median rank |
|---|---|---|---|---|
| our v5 reason (Qwen3-8B, target + review seen) | 0.232 | 0.287 | 0.471 | 132 |
| our v5 motive phrases | 0.081 | 0.127 | 0.264 | 528 |
| **our mined prototypes of the row (what MotiveRec uses)** | **0.031** | **0.059** | **0.140** | **1022** |
| GEARec v2 reasoning, first candidate | 0.086 | 0.129 | 0.266 | 601 |
| GEARec v2 reasoning, best of 8 by this rank | 0.185 | 0.245 | 0.421 | 178 |

GEARec v2 (`/common/home/users/m/mk.ngohuu.2024/research/GEARec_R3_toys/v2_target_reasoning`): gemma-3-4b sees the
history and the target, picks 1–3 support history items, writes a 3–5 sentence reasoning phrased as a prediction from
the history, without naming the target. Best of 8 by mxbai rank; final set = rank ≤ 100 + LLM judge ≥ 4
(train 21,984 / 53,898 rows; test 2,537 / 6,738). Rows are identical to our Toys splits (checked).

**Reading.** The information about the target is lost in mining, not in generation: reason → phrases → prototype
drops top-20 reachability 0.287 → 0.127 → 0.059. The generic prototype texts ("gift for a special occasion")
describe a reason to buy *anything*. This supports the author's view (2026-11-13): a good motive must explain the
purchase in a way that is semantically related to the item it leads to, and should be checked for that, as GEARec
does with its reachability filter.

## Next: E8b — motive mining v6 (job 364815, submitted 2026-11-13)
Code: `collab/build_motive_corpus_v6.py` + `collab/submit_motive_corpus_v6.sbatch` (Toys). Results go to
`report_mining_v6.md`.
1. Source: GEARec v2 best-of-8 reasoning (train 53,157 rows with a well-formed candidate).
2. Qwen3-8B extracts 1–2 motives per row as **need → item kind** (no proper names; vague motives only when tied to
   an item kind).
3. Qwen3-Embedding-8B of the statements; kNN-Louvain on train statements at P = 1000 and 2000; fold < 5 rows.
4. Qwen3-8B names each cluster from its most central statements, keeping both need and item kind.
5. Accept if, against v5: reachability of the prototype text (`pilot/reasoning_reachability.py 1500 <corpus>`) beats
   top-20 0.059, and history prediction (`pilot/mining_diagnostics.py`) beats popularity by a clear margin.
