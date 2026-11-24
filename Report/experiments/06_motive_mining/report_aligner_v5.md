# E8c — Motive–interaction aligner on v5 motives (Toys)

Back to [main plan](../../main.md). Date: 2026-11-13. CPU. Code: `collab/train_motive_aligner.py`. Outputs:
`data/Toys_and_Games/collab/motive_corpus/retriever/aligner_<tag>/`.

## Idea
Inspired by IRLLRec (Wang et al., SIGIR '25, "Intent Representation Learning with LLM for Recommendation"), which
aligns LLM text intents with interaction intents. Here the alignment is per interaction (history → motive → next item):
- history tower q (current retriever features: SASRec state, gte history text, item motive profile, user's past motives);
- motive text tower t (the row's v5 motive phrases, Qwen3 embeddings); the same tower maps the 881 prototype centroids
  to p_k;
- item tower e (target item's SASRec embedding).

Losses: `ce` multi-positive CE of softmax_k(q·p_k) on hard labels; `hm` InfoNCE q↔t; `mi` InfoNCE t↔e; `match`
KL(softmax(t·p)‖softmax(q·p)). Inference uses q only. Weights 0.5 for each auxiliary loss, 30 epochs, early stopping
on valid MRR.

Extra metric: **item HR@10 / NDCG@10** = rank of the next item by (softmax-weighted mix of the top-10 retrieved p_k)·e_i,
history masked: does the retrieved motive point to the next item in interaction space?

## Results (Toys test; valid in the json files)
| variant | R@1 | R@5 | R@10 | R@20 | MRR | item HR@10 | item NDCG@10 |
|---|---|---|---|---|---|---|---|
| current cls MLP retriever | 0.0975 | 0.2358 | 0.3181 | 0.4132 | 0.2895 | — | — |
| A ce only | 0.1025 | 0.2354 | 0.3166 | 0.4090 | 0.2961 | 0.0024 | 0.0011 |
| B ce + hm | 0.0995 | 0.2359 | 0.3164 | 0.4056 | 0.2934 | 0.0027 | 0.0015 |
| C ce + hm + mi | 0.0989 | 0.2335 | 0.3155 | 0.4034 | 0.2908 | 0.0332 | 0.0220 |
| D ce + hm + mi + match | 0.0975 | 0.2369 | 0.3173 | 0.4058 | 0.2905 | 0.0407 | 0.0323 |
| E hm + mi + match (no ce) | 0.0974 | 0.2315 | 0.3030 | 0.3950 | 0.2901 | 0.0218 | 0.0138 |

Reference (E8a, a different scorer): next item from item popularity HR@10 0.0137.
Valid item HR@10 is lower than test (D: 0.0171), so read the item columns as relative between variants.

## Reading
1. **Alignment does not make v5 labels easier to retrieve.** Every variant stays at MRR 0.290–0.296 and R@10 ≈ 0.316,
   the same as the current retriever. The ceiling is the labels: v5 hard labels are barely predictable from the
   history (E8a), and no retriever objective fixes that.
2. **Correction (2026-11-13): the item columns of A and B are not meaningful.** The item tower E is trained only by
   the `mi` loss, so in A and B it keeps its random initial weights and 0.002 is a random-level score. The C/D values
   (0.033–0.041) show that retrieved motives plus a trained item tower can rank the next item, but they are unstable
   (D valid 0.0171) and have no counterpart for the cls retriever. The fair comparison is the fixed scorer in
   [E8d](report_relabel_v5.md): cls retriever 0.0439, aligner D 0.0350 test HR@10, i.e. the aligner does **not**
   make motives more item-informative than the current retriever on v5.
3. Hence the aligner is worth keeping, but it needs better motives as targets: v6 (need → item kind, job 364815) is
   the next input. With v6, both the label retrieval and the item columns can move.

> **Scope note (2026-11-13):** this aligner is *inspired by* IRLLRec (pairwise InfoNCE alignment + intent matching), not a reproduction. Missing vs IRLLRec: LightGCN backbone + graph denoising, recommendation as the main task, learnable interaction-space intent prototypes, translation alignment (Gaussian perturbation), EMA-teacher bidirectional matching, LLM-generated per-user/per-item intent texts, their hyper-parameters. The negative results apply to this family of objectives for motive retrieval, not to IRLLRec itself.
