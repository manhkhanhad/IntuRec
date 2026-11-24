# E8g — Aligner v2: user motive × item motive (IRLLRec-style, averaged texts), v6 P1000, Toys

Back to [main plan](../../main.md). Date: 2026-11-13. CPU, 20 epochs, early stopping on valid fixed-scorer NDCG@10.
Code: `collab/train_motive_aligner2.py`; outputs `motive_corpus_v6/P1000/retriever/aligner2_<tag>/`.

## Design (cheap check before generating per-user / per-item motive texts)
- User-motive text (history only): mean v6 statement of the user's earlier train rows + of train buyers of the history
  items (99.8% of rows have one). Item-motive text: mean v6 statement of the item's train buyers (leave-one-out).
- Towers: history q, item e (SASRec), shared text tower; π_u, π_i over the 998 prototypes.
- Losses: `ce` (π_u vs v6 labels); `user` (InfoNCE q↔user text + KL); `item` (InfoNCE e↔item text + KL);
  `inter` (next-item CE with log Σ_k π_u π_i).
- Read-outs handed to MotiveRec: `user` = top-k of π_u; `joint` = π_u weighted by the motives of the 50 most likely
  next items.

## Results (test; valid in results.json)
| variant | read-out | R@10 | MRR | fixed HR@10 | fixed NDCG@10 | motive-space item HR@10 |
|---|---|---|---|---|---|---|
| **plain cls retriever (E8b)** | — | 0.2368 | 0.2058 | **0.0695** | **0.0407** | — |
| ce only | user | 0.2190 | 0.1890 | 0.0675 | 0.0387 | 0.0004 |
| ce + user + item | user | **0.2508** | **0.2169** | 0.0659 | 0.0391 | 0.0521 |
| ce + inter | user | 0.2377 | 0.2026 | 0.0669 | 0.0382 | 0.0591 |
| all | user | 0.2411 | 0.2132 | 0.0672 | 0.0394 | 0.0641 |
| all | joint | 0.2377 | 0.2077 | 0.0631 | 0.0368 | — |

## Reading
1. **No variant beats the plain retriever on the fixed item scorer** (best 0.0675 vs 0.0695 test; valid best 0.0764
   vs 0.0761 — a tie). By the rule set before the run, option (b) (LLM-generated per-user / per-item motive texts)
   is **not** pursued for now.
2. **User/item alignment helps motive retrieval a little:** within this script, ce only → ce + user + item raises
   R@10 0.2190 → 0.2508 and MRR 0.1890 → 0.2169, and it is the best retrieval of the v6 labels so far (plain
   retriever 0.2368 / 0.2058). The gain does not carry over to item usefulness.
3. **The joint read-out is worse than the user read-out** (0.0631 vs 0.0672): weighting by likely next items does not
   pick more useful motives.
4. Recommending purely through motive space reaches HR@10 0.064 (test), about the same as the fixed scorer: the
   prototypes carry real item information, but the plain retriever already extracts it.
5. Conclusion: the remaining lever is the motive labels themselves (e.g. keep only the forward-looking motive of each
   row), not the retriever objective.

> **Scope note (2026-11-13):** this aligner is *inspired by* IRLLRec (pairwise InfoNCE alignment + intent matching), not a reproduction. Missing vs IRLLRec: LightGCN backbone + graph denoising, recommendation as the main task, learnable interaction-space intent prototypes, translation alignment (Gaussian perturbation), EMA-teacher bidirectional matching, LLM-generated per-user/per-item intent texts, their hyper-parameters. The negative results apply to this family of objectives for motive retrieval, not to IRLLRec itself.
