# E8d — Interaction-fitted motive relabelling on v5 (Toys)

Back to [main plan](../../main.md). Date: 2026-11-13. CPU. Code: `collab/train_motive_aligner.py --relabel-rounds 2
--select item --epochs 20 --prior-power γ`; outputs `motive_corpus/retriever/aligner_relabel_g{0,05,1}/`.

## Idea (author's preference: "the motive that fits the interaction")
Each train interaction's motive is re-chosen as the motive that both fits the purchased item and is supported by the
history, instead of the motive the review happened to name:
- p(m | item): motive mix of all train buyers of the item (original v5 labels, smoothed, fixed);
- p(m | history): aligner scores from the history only, out-of-fold (5 folds by user), so a row cannot copy its own
  label;
- new label = top-2 of p(m | item) · p(m | history) / prior^γ (soft); the row's motive text becomes the weighted mix of
  those prototype centroids; the aligner (all four losses) is retrained; 2 rounds.

## Results (Toys)
Train label stats after relabelling: distinct motives used as top-1 (of 881), largest share, share of rows whose new
top-1 is one of their v5 labels.

| γ | round | distinct | largest share | top-1 in v5 labels |
|---|---|---|---|---|
| 0 | 1 / 2 | 509 / 170 | 0.223 / 0.280 | 0.33 / 0.33 |
| 0.5 | 1 / 2 | 744 / 460 | 0.171 / 0.202 | 0.33 / 0.36 |
| 1 | 1 / 2 | 869 / 849 | 0.103 / 0.103 | 0.32 / 0.37 |

**Does the retrieved motive point to the next item?** Two scorers, history-only retrieval, next item ranked over all
items (history masked), HR@10 / NDCG@10:
- *fixed*: score_i = Σ_k π_k p(item i | motive k), with p(item | motive) from the v5 train labels and π the softmax
  over the top-10 retrieved motives (no learned parts; same scorer for every model);
- *learned*: the aligner's own item tower (unstable: valid and test often move in opposite directions).

| model | valid fixed | test fixed | valid learned | test learned |
|---|---|---|---|---|
| current cls retriever | 0.0530 / 0.0320 | **0.0439 / 0.0244** | — | — |
| aligner, round 0 (no relabel) | 0.0416 / 0.0246 | 0.0350 / 0.0179 | 0.0171 / 0.0121 | 0.0407 / 0.0323 |
| relabel γ = 0, round 2 | 0.0337 / 0.0202 | 0.0245 / 0.0131 | 0.0184 / 0.0105 | 0.0122 / 0.0060 |
| relabel γ = 0.5, round 1 | 0.0460 / 0.0281 | 0.0338 / 0.0183 | 0.0281 / 0.0177 | 0.0174 / 0.0091 |
| relabel γ = 1, round 1 | 0.0508 / 0.0317 | 0.0381 / 0.0205 | 0.0350 / 0.0220 | 0.0264 / 0.0147 |
| relabel γ = 1, round 2 | 0.0524 / 0.0328 | 0.0413 / 0.0232 | 0.0362 / 0.0223 | 0.0278 / 0.0160 |

Retrieval of the original v5 labels drops with relabelling (test MRR 0.2905 → 0.2229 at γ = 1), as expected: the
labels changed.

## Reading
1. **Without a prior correction the relabelling collapses** (γ = 0: 170 motives left, one takes 28% of rows), and the
   motives become less useful. Popular motives win both p(m | item) and p(m | history).
2. **γ = 1 keeps the vocabulary** (849 motives, largest share 10%) and is the only setting that improves with rounds:
   on the fixed scorer, aligner 0.0350 → 0.0413 test HR@10 (valid 0.0416 → 0.0524).
3. **It does not beat the current cls retriever** on the fixed scorer (test 0.0413 vs 0.0439; valid 0.0524 vs 0.0530).
   The aligner itself starts lower than the cls retriever on this scorer (0.0350), and relabelling recovers that gap
   but no more.
4. Caveat: the fixed scorer uses item–motive links from the v5 labels, so it can under-rate relabelled motives whose
   meaning shifted; the learned item tower is too noisy to decide. The final judge is MotiveRec accuracy.
5. **Conclusion for v5:** fitting each interaction's motive is feasible without collapse (γ = 1), but with the v5
   vocabulary (generic prototypes) it reaches, not exceeds, the current retriever. Apply it to v6 (need → item kind),
   where the vocabulary itself is item-linked.

> **Scope note (2026-11-13):** this aligner is *inspired by* IRLLRec (pairwise InfoNCE alignment + intent matching), not a reproduction. Missing vs IRLLRec: LightGCN backbone + graph denoising, recommendation as the main task, learnable interaction-space intent prototypes, translation alignment (Gaussian perturbation), EMA-teacher bidirectional matching, LLM-generated per-user/per-item intent texts, their hyper-parameters. The negative results apply to this family of objectives for motive retrieval, not to IRLLRec itself.
