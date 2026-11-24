# E8h — IRLLRec-faithful motive model on v6 motives (Toys)

**Question.** Given an interaction (history only), can an IRLLRec model predict the user motive (the *need* part of a v6
motive) and the item motive (the *item kind* part)? Can it predict the item motive from the interaction plus the user motive?
Is either better than the plain retriever?

**Answer.** No. The best read-out (need + kind) reaches test R@10 0.1781 and fixed-scorer HR@10 0.0560. The plain cls
retriever reaches 0.2368 and 0.0695. The conditional read-out (need → item kind) is worse than the plain sum. The motive
alignment also *hurts* the recommender itself: test HR@10 drops from 0.0974 (plain LightGCN) to 0.0775 with IRLLRec losses.
IRLLRec is dropped as a motive retriever.

## Setup
- **Code.** `collab/irllrec_motive.py`. It ports `encoder/models/general_cf/lightgcn_int.py` from the official IRLLRec
  repository (amazon config); the loss helpers are copied verbatim.
  - LightGCN, d = 32, 3 layers, layers summed. BPR loss, reg 1e-7.
  - RLMRec profile alignment: InfoNCE, weight 1e-2, temperature 0.2.
  - K = 128 user and item intents with the adaptive mask.
  - kd_int (InfoNCE vs intent text): weight 2e-2, temperature 0.2. kd_int_2 / itm: 1e-7 (kd_int_2 on 4,096 random nodes).
  - Adam, lr 1e-3, batch 4096. Evaluate every 3 epochs, patience 5, early stopping on valid HR@10.
- **Adaptation.** The user node is an *interaction row*, because each test user has about 2.6 rows with different
  histories.
  - Graph: row → history items, all splits (past interactions only). BPR positives: train targets.
  - Profiles: item = mxbai item text; row = mean of its history items' profiles.
  - User intent text: mean mxbai of the row's v6 *needs* (train rows only).
  - Item intent text: mean mxbai of the *item kinds* over the item's train buyers.
- **Read-outs** on the 1,000 v6 prototypes (P1000):
  - **need:** cos(row intent embedding, int_mlp(prototype need)).
  - **kind:** softmax-weighted over the top-50 LightGCN next items of cos(item intent embedding, int_mlp(prototype kind)).
  - **need+kind:** z-scored sum.
  - **need+cond_kind:** the kind score re-weighted by the prototypes whose need matches the predicted need.
- **Fixed scorer.** As in E8c/E8g: Σ softmax(top-10 motive scores) · p(item | motive).
- **Ablation `plain`.** All text weights are 0, i.e. LightGCN only. Its text read-outs use an untrained int_mlp, so they
  are a chance-level reference.
- **Run.** Job 365134 on GPU, 4 s/epoch; `irllrec` stopped near epoch 105.

## Results (test; valid in brackets)

| Read-out | R@1 | R@10 | MRR | fixed HR@10 | fixed NDCG@10 |
|---|---|---|---|---|---|
| need (user motive) | 0.0297 | 0.1483 (0.1483) | 0.1265 | 0.0534 (0.0672) | 0.0297 |
| kind (item motive) | 0.0280 | 0.1490 (0.1574) | 0.1233 | 0.0491 (0.0569) | 0.0269 |
| need+kind | 0.0358 | 0.1781 (0.1816) | 0.1467 | 0.0560 (0.0659) | 0.0302 |
| need+cond_kind | 0.0316 | 0.1515 (0.1476) | 0.1307 | 0.0484 (0.0662) | 0.0267 |
| plain LightGCN, untrained text head (chance) | 0.0001 | 0.0095 | 0.0112 | 0.0061 | 0.0051 |
| **cls retriever (E8b)** | — | **0.2368** | **0.2058** | **0.0695** | — |

In `plain`, the need, kind and need+kind rows are identical. With an untrained head the scores are degenerate, so read
that run's text rows only as chance.

LightGCN recommendation (HR@1 / HR@5 / HR@10 | NDCG@1 / NDCG@5 / NDCG@10):

| Run | valid | test |
|---|---|---|
| plain LightGCN | 0.0346 / 0.0797 / 0.1130 \| 0.0346 / 0.0578 / 0.0685 | 0.0227 / 0.0665 / 0.0974 \| 0.0227 / 0.0451 / 0.0551 |
| IRLLRec (motive aligned) | 0.0260 / 0.0580 / 0.0876 \| 0.0260 / 0.0423 / 0.0517 | 0.0171 / 0.0476 / 0.0775 \| 0.0171 / 0.0324 / 0.0420 |

## Reading
1. **The alignment does learn the motives:** R@10 is 0.15–0.18, against 0.01 for chance. But it learns them less well than
   the cls retriever, which classifies the history directly over the SASRec item embeddings. In IRLLRec the motive must
   pass through a 32-d collaborative embedding, and that bottleneck is the likely cause.
2. **Conditioning on the user motive does not help.** need+cond_kind (test fixed HR@10 0.0484) is below need+kind
   (0.0560). The need and kind halves of a v6 motive are each nearly as predictive alone (R@10 0.148 / 0.149) and add
   up when summed.
3. **Motive alignment costs recommendation accuracy here:** test HR@10 is 0.0974 → 0.0775. This is the opposite of the
   IRLLRec paper. Our setting differs from the paper's in three ways that may explain it:
   - row-level user nodes, each with ≤ 10 edges;
   - intent text exists for only 79% of rows (train only);
   - weights tuned for user-level graphs.
   The weights were not tuned (faithful port).
4. **Decision.** Keep the cls retriever as the v6 support source for MotiveRec-T. IRLLRec-style alignment, like the
   aligners of E8c/E8g, does not beat it.

Outputs: `data/Toys_and_Games/collab/motive_corpus_v6/P1000/retriever/irllrec_{irllrec,plain}/`.
Log: `run_logs/IRLLRecMotive.365134.out`.
