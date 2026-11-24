# CollabRec GRPO — research log

Replaces `RESEARCH_LOG_refined_infonce.md`, lost with the `collab-refined-infonce` worktree (removed
~2026-11-11 22:36, not by this session; its committed GRPO code survives on branch
`worktree-collab-refined-infonce`, commits ae3db4b / 28edb06 / 17fdf34 / da13562). Values are fractions.

## 1. R3 stage-2 noise GRPO on CollabRec (2026-11-08 → 11-10): all 4 runs collapsed
Code: `run_collabrec_grpo.py` (branch above). R3-exact runtime (trl 0.16.1 semantics: bf16 log-probs,
pad-to-8, reference on the policy's embeddings, fixed-size norm-scaled advantages, 1 update per rollout),
8 samples / prompt, noise on the `<|Thought|>` embedding, reward −PPL of the target title.

| Run | Data / init | Setup | Outcome |
|---|---|---|---|
| 357958 | Toys / 345291 | LLM trained, lr 5e-5, noise σ 1.5 | diverged by step 200 (valid NTP 0.5224 → 1.6750) |
| 358206 | Toys / 345291 | LLM frozen, σ 1.5 (≈260× thought norm) | collapsed ~step 420 (clean PPL → 4.5e8) |
| 359243 | Toys / 345291 | LLM frozen, noise 0.65× thought (R3's ratio, latent norm ~91 vs 59) | collapsed ~step 300 |
| 359282 | CDs / 359280 | LLM frozen, noise 0.65× thought | collapsed step 440 (grad norm ×120 in 10 steps) |

Why: (1) the SFT models ignore their thought (mask/shuffle leave HR@10 unchanged), so the reward signal is
tiny (in-group std ~0.005 Toys / ~0.013 CDs PPL) and below bf16 resolution; (2) R3's update has no trust
region (KL ≡ 0, clip inactive, norm-scaled advantages → fixed-size steps); (3) lr 5e-5 is huge relative
to CollabRec's generator (thought norm 0.18–0.22, InjectNet weight rms ~0.02); (4) a clean sample below
the baseline gets a negative advantage and is pushed further down — a runaway. R3 itself collapses this
way (CDs v4; 1 of 3 Toys seeds at 5e-5; Musical at lr ≥ 1e-4). CDs SFT retrain 359280: test HR@10 0.1174 /
NDCG@10 0.0838 (archived 335237: 0.1106 / 0.0802). Failed-run checkpoints deleted (user, 2026-11-12).

## 2. Motive-selection GRPO (2026-11-12, user's idea): GRPO on *which motives to retrieve*
CollabRec retrieves its latent (motives) instead of generating it, so GRPO's action becomes motive choice.
Reader = frozen O2 + InfoNCE model (351001); reward = utility u(m) = log p(target | history, [m, m]) −
log p(target | history, null), stored by the cross-fitted utility judges (train) and the full judge (test).

### Phase 1 — offline selector (`collab/grpo_motive_select.py`, CPU, seconds per run)
Selector over the retriever's top-10: retriever log-prob + rank + popularity / history agreement +
bilinear(history PCA, motive-text PCA); initialised to the retriever. GRPO: 8 samples / row, z-scored
reward, KL to the retriever's top-10 distribution. Held-out = 10% of train users. Toys test (6,738 rows),
3 seeds:

| Method | mean u | share of oracle gain | u > 0 | label hit@1 | picks rank 0 |
|---|---|---|---|---|---|
| retriever top-1 | −0.695 | −0.395 | 0.404 | 0.194 | 1.00 |
| random of top-10 | −0.743 | −0.421 | 0.384 | 0.060 | 0.10 |
| supervised (CE to softmax(u/0.5)) | −0.801 ± 0.014 | −0.456 | 0.384 | 0.134 | 0.37 |
| **GRPO, KL 0** | **−0.305 ± 0.013** | −0.174 | 0.442 | 0.060 | 0.10 |
| GRPO, KL 0.05 | −0.324 ± 0.014 | −0.185 | 0.439 | 0.078 | 0.15 |
| GRPO, KL 0.2 | −0.376 ± 0.004 | −0.214 | 0.439 | 0.139 | 0.37 |
| oracle best of top-10 | +1.729 | +0.988 | 0.900 | 0.156 | 0.11 |

- Held-out train users: retriever top-1 −0.349 → GRPO −0.024 (≈ null level).
- GRPO moves away from "complete collection" (retriever top-1 for 0.28 of rows, harmful when wrong) to
  broad motives (Creative Play Engagement 0.12, Encouraging Imaginative Play 0.09, …); not a collapse onto
  one motive (129 distinct). It rarely picks a labelled-true motive (0.06) yet helps the reader more.
- Supervised imitation of each row's best motive is worse than the retriever (the best motive is not
  predictable; GRPO learns the choice with the best expected payoff).
- Limitation: u is relative to the reader's null, which is far below O0 (HR@10 0.0763 vs 0.1112), so
  "u < 0" can still beat null in HR. No abstain / O0-fallback action yet (no O0 rewards on train rows).

### Phase 2 — decode with the reader (`collab/motive_reader.py`, `collab/submit_motive_reader.sbatch`)
Uses the main checkout's (older) CollabRec code; verified on CPU against stored utilities (test row 0:
rank-5 motive −7.806 vs −7.806 stored; top-1 −9.989 vs −9.952; null −11.131 vs −10.945). Toys test decode
with one motive per row as [m, m]: GRPO picks (KL 0, seed 0; test u −0.288), retriever top-1, oracle
best-of-10. References: O0 0.1112, reader with own true motives 0.1892 (run decode) / 0.1775 (diag decode),
reader + retriever top-2 0.1015, reader + null 0.0763. Jobs (JobManager, from commit aff69fd):
363689 grpo_b0_s0, 363690 retriever_top1, 363691 oracle_best10; outputs
`checkpoints/Toys_and_Games/CollabRec_GRPO_select/decode_<NAME>/`.

| Reader 351001 input ([m, m]) | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| oracle best of top-10 (363691) | 0.0530 | 0.1110 | 0.1560 | 0.0530 | 0.0824 | 0.0969 |
| O0, no motives (350999) | 0.0417 | 0.0867 | 0.1112 | 0.0417 | 0.0650 | 0.0729 |
| retriever top-1 (363690) | 0.0405 | 0.0766 | 0.0994 | 0.0405 | 0.0589 | 0.0662 |
| GRPO picks, KL 0 seed 0 (363689) | 0.0347 | 0.0706 | 0.0944 | 0.0347 | 0.0533 | 0.0610 |

- Selection headroom is real: the best of the retriever's top-10 (chosen with the target, so a ceiling)
  beats O0 by +0.045 HR@10 — the first deployable candidate pool that clearly separates from O0.
- GRPO raised its training reward (mean u −0.69 → −0.29) but LOWERED ranking vs retriever top-1
  (−0.0050 HR@10 / −0.0052 NDCG@10): expected log-likelihood utility favours broad, "safe" motives that lift
  the target's likelihood a little everywhere without separating it from competitors. The reward must be
  ranking-aware (target vs competitor items under the same motive), and an abstain → O0 action is needed,
  since every non-oracle pick is below O0.

### Phase 3 — ranking-aware reward (2026-11-12, user's choice)
`collab/hard_negatives.py` (SASRec top-20 items per row, not the target or history) + `collab/rank_utility.py`
(fold judges score target + 20 negatives under null and the top-10 motives; cached prefix per motive;
CPU-checked vs stored utilities, mean |diff| 0.02–0.07 nats). Jobs 363812 / 363813 / 363814 (train0 /
train1 / test). Reward rank_lse = target's length-normalised log-softmax among the 21 items.

Test signal (6,738 rows; target rank among 21, 0 = first): null 10.01 / retriever top-1 10.08 / best of
top-10 6.54; ranked 1st 0.079 / 0.084 / 0.191 (summed log-lik: 0.092 / 0.092 / 0.240); rank moves 2–3
positions across a row's motives. The rank-best motive matches the likelihood-best motive for only 0.636
of rows.

Selector sweep (all train rows, 3 seeds, length-normalised reward), test:

| Method | target 1st of 21 | target in top 5 | mean rank |
|---|---|---|---|
| retriever top-1 | 0.0841 | 0.2757 | 10.08 |
| random of top-10 | 0.0795 | 0.2771 | 10.04 |
| supervised (CE to softmax(r/0.5)) | 0.0857 ± 0.0010 | 0.2785 ± 0.0016 | 9.94 |
| GRPO, KL 0 | 0.0888 ± 0.0008 | 0.2841 ± 0.0017 | 9.97 |
| GRPO, KL 0.05 | 0.0889 ± 0.0010 | 0.2847 ± 0.0012 | 9.96 |
| hindsight best of top-10 | 0.1576 | 0.4651 | 6.79 |

- Unlike the likelihood reward, GRPO on the ranking reward moves the target the right way, consistently
  across seeds (+0.0048 ranked 1st, +0.009 in top 5 vs retriever top-1) — but recovers only ~7% of the
  hindsight headroom (0.0841 → 0.1576). Summed-log-lik reward (fold 1 only, 1 seed): no gain.
- Same wall as retrieval: which motive will make the target stand out is mostly not predictable from the
  history; a history-only selector cannot reach the oracle-selection gain.

Collapse check (seed 0, 15 epochs, per-epoch diagnostics): no blow-up (grad norm 0.3 → 0.67, max 0.81), but
ENTROPY COLLAPSE — policy entropy 1.35 → 0.15 (KL 0) / 1.60 → 0.37 (KL 0.05), max prob → 0.94 / 0.86, KL to
the retriever → 2.16 / 1.84, while held-out reward stays flat: the 15-epoch schedule (my choice, not R3's)
reuses each row's fixed reward 15× and memorises per-row noise.

R3-style single pass (1 epoch, 3 seeds), test:

| Setting | target 1st | top 5 | mean rank | end entropy | max prob | KL to retriever | max grad norm |
|---|---|---|---|---|---|---|---|
| batch 512, lr 3e-3, KL 0.05 | 0.0894 ± 0.0016 | 0.2836 | 9.95 | 1.62 | 0.44 | 0.39 | 0.48 |
| batch 512, lr 3e-3, KL 0 | 0.0870 ± 0.0003 | 0.2852 | 9.97 | 1.32 | 0.55 | 0.65 | 0.58 |
| batch 32 (R3), lr 3e-4, KL 0.05 | 0.0870 ± 0.0002 | 0.2824 | 9.99 | 1.80 | 0.38 | 0.20 | 2.46 |
| batch 32 (R3), lr 3e-4, KL 0 | 0.0869 ± 0.0004 | 0.2827 | 9.99 | 1.60 | 0.45 | 0.38 | 4.33 |

- One pass gives the same small gain as 15 epochs (best 0.0894 vs 0.0889; retriever 0.0841) WITHOUT entropy
  collapse (end entropy 1.3–1.8 vs 0.15–0.37). Small-batch runs have larger single-batch gradient norms
  (2.5–4.3, noisier per-batch estimates) but no instability.
- The gain stays ~7% of the hindsight headroom: the limit is predictability, not the training schedule.
- Related work (2026-11-12 search): REPLUG / Atlas / RA-DIT train retriever and reader together (LM-supervised
  retrieval ≈ our supervised baseline, LOOP ≈ our contrastive reward); Rec-R1 / Search-R1 reward the downstream
  metric; Rank-R1 / REARANK use an LLM setwise/listwise selector trained with GRPO; DAPO addresses entropy
  collapse (Clip-Higher, dynamic sampling); Self-RAG learns when not to retrieve (abstain).

### Phase 4 — literature-based methods + decode of the current direction (2026-11-12, user's request)
- Decode of the single-pass ranking-reward GRPO picks (KL 0.05, seed 0 by a fixed rule; test 0.0873 ranked
  1st of 21): job 364174 (`decode_grpo_rank_1pass_b0.05_s0`).
- LLM setwise selector, Rank-R1 style (`collab/llm_selector_grpo.py`): Qwen2.5-1.5B-Instruct + LoRA (r 16)
  picks one of the retriever's top-10 motives (A–J) or "None" (K) from the user's last 10 purchases; action =
  next-token distribution over the 11 letters; GRPO, 8 samples / row, ranking-aware reward (None = the
  reader's no-motive score), KL 0.05 to the initial policy (adapter off), zero-variance groups skipped (DAPO),
  one pass, lr 2e-5 warmup + cosine. Job 364178 (`llm_qwen1.5b`, LoRA) cancelled after ~2 min at the
  user's request ("use train full"): replaced by FULL fine-tuning (`--full`: all ~1.5B weights, frozen
  reference copy for the KL, gradient checkpointing, lr 1e-6 since 5e-5 destroyed the full LLM in 357958),
  job 364182 (`llm_qwen1.5b_full`).
- Joint reader + selector, Atlas / RA-DIT style (`collab/joint_reader_selector.py`): loss
  −log Σ_{m∈S} π_S(m) p(target | history, [m, m]), S = null + selector top-3; reader init 351001 (lr 1e-5),
  selector = phase-1 features + null logit (init: retriever distribution, null logit = median top-1 log-prob − 1;
  starts with argmax top-1 for 0.83 of rows, null 0.17); one pass, 32 rows / step; resumable. Job 364179
  (`joint_k4`); its test picks will be decoded with its own reader.

### 364182 result — LLM setwise selector, full fine-tuning (2026-11-12, 48 min on lunar)
Test (6738 rows, target among 21): llm_grpo top1 0.0846 / top5 0.2786 / mean rank 9.987, picked_A 0.101, none 0.0016;
zero-shot 0.0846 / 0.2795 / 9.970; retriever top-1 0.0841 / 0.2757 / 10.084; oracle 0.1582 / 0.4673 / 6.767.
Training: reward -3.14 -> ~-3.10 (noisy, flat), KL to init rose to 0.05, entropy 1.74 -> 1.65 (max ln 11 = 2.40),
held-out top1 0.119 -> 0.120 across steps 500-3000. GRPO moved the policy (KL > 0, A-rate 0.05 -> 0.10) but not
toward better picks: no gain over zero-shot or the retriever. Verdict: negative.

### 364174 result — decode of single-pass ranking-reward GRPO picks (KL 0.05, seed 0), reader 351001 [m, m]
HR@1/5/10 0.0381 / 0.0748 / 0.0996, NDCG@1/5/10 0.0381 / 0.0569 / 0.0649.
vs retriever top-1 0.0405 / 0.0766 / 0.0994 (NDCG@10 0.0662); likelihood-reward GRPO 0.0944 / 0.0610; oracle 0.1560 / 0.0969;
O0 (no motive) HR@10 0.1112 / NDCG@10 0.0729. Verdict: ties retriever on HR@10, slightly worse elsewhere; below O0.

### 364232 result — decode of LLM setwise selector (364182, full FT) picks, reader 351001 [m, m]
HR@1/5/10 0.0344 / 0.0683 / 0.0907, NDCG@1/5/10 0.0344 / 0.0521 / 0.0594 — worst of all decodes
(retriever top-1 HR@10 0.0994, single-pass GRPO 0.0996, likelihood GRPO 0.0944, O0 0.1112). Tie on the 21-way
rank reward did not transfer: the LLM mostly picks lower-ranked motives, which hurt the full-catalogue decode.

### 364179 result — joint reader + selector (k4, 2:54 on lunar)
Test picks: null 0.742, retriever top-1 0.234, 26 distinct motives. The marginal-likelihood objective drove the
selector to "no motive" for 3/4 of rows: consistent with O0 (no motive) beating every motive-based decode.
Decode with its own reader (joint_k4/reader) requested from JobManager.

### 364329 result — decode of joint_k4 picks with its own trained reader
HR@1/5/10 0.0401 / 0.0800 / 0.1042, NDCG@1/5/10 0.0401 / 0.0605 / 0.0683.
Best motive-based decode (vs retriever top-1 + reader 351001: 0.0994 / 0.0662), but still below O0 (no motive)
HR@10 0.1112 / NDCG@10 0.0729. Note: the starting reader 351001 is itself below O0, so the joint gain partly
recovers from a weaker start rather than adding value from motives.

## Phase 4 summary
No motive-selection method beats O0. Ranking: O0 .1112 > joint .1042 > rank-GRPO 1pass .0996 ≈ retriever top-1 .0994
> likelihood GRPO .0944 > LLM selector .0907 (HR@10). The headroom (oracle .1560) needs target knowledge the
selector does not have; when free to choose (joint), training abstains on 74% of rows.

## Phase 5 — motive-only (no InfoNCE), reader 364239 (2026-11-13)
User decision: set InfoNCE aside; build GRPO on the motive-only model. Reader 364239 = predicted (retriever top-3, cls
scores) motives, k=3, inject readout, COLLAB_WEIGHT=0, test HR@10 0.1123 / NDCG@10 0.0743 (O0 350999: 0.1111 / 0.0729;
true-motive no-InfoNCE upper bound 351000: 0.1357 / 0.0864). Retriever top-3 contains a true motive in 0.3100 of test rows.
Step 1 check (does 364239 use motive content?): decodes with supports top3 (sanity, must give 0.1123), true (true
motives first, predicted fill) and null x3. build_k3_supports.py reproduces the dataset's float32 tie order exactly
(slot ids identical to motive_source="predicted"). motive_reader.py: COLLAB_CODE env selects the CollabRec code; k slots
taken from the picks' width.

### Phase 5b — joint retriever + recommender, 3 motive slots (collab/joint_k3.py), user request 2026-11-13
Reader init 364239 (k=3, inject, no InfoNCE; COLLAB_CODE = recovered code). Selector = SelectorNull over retriever
top-10 (cls, float32 tie order) + null; motive SET of 3 drawn Plackett-Luce (set prob = sum over 6 orderings; unit
test: probabilities of all 165 sets sum to 1, Gumbel-top-3 frequency matches). Per row S = greedy top-3 + 3 Gumbel
samples (duplicates masked); loss -log sum_S pi_S(s) p(y | h, s); slots fed in candidate-rank order, null last.
reader lr 1e-5, selector lr 1e-3, 32 rows/step, 1 pass, warmup 50 + cosine. null logit init = 5th pct of rank-3
retriever log-prob - 1 (so training starts at the reader's own input distribution). CPU dry run (2 rows) passed.
Run: MODE=joint_k3 NAME=nsets4 -> CollabRec_GRPO_select/joint_k3_nsets4/. Then decode picks_test.npy [n, 3] with
--reader joint_k3_nsets4/reader (COLLAB_CODE set).

### 365125 — sanity decode of reader 364239 with its own top-3: HR@1/5/10 0.0413 / 0.0818 / 0.1077, NDCG 0.0413 / 0.0623 / 0.0706
Does not reproduce the run's test line (0.1123) because that line is the in-memory EPOCH-3 model, while the saved
checkpoint is EPOCH 2 (best valid NTP 0.5866 < epoch-3 0.6143; "saved best checkpoint" in train.log). Same pattern as
345291. So the true baseline of the saved reader (and the joint_k3 init) is HR@10 0.1077 / NDCG@10 0.0706, below O0
(0.1111 / 0.0729). Decode batch 8 vs the run's 2 may add a small difference on top.

### 365126 / 365127 — reader 364239 with true motives / no motive
true: HR@1/5/10 0.0413 / 0.0824 / 0.1088, NDCG 0.0413 / 0.0626 / 0.0711; null: 0.0417 / 0.0824 / 0.1089, NDCG 0.0417 / 0.0628 / 0.0713.
top3 = true = null within noise: the predicted-motive k=3 reader IGNORES motive content. Selection (GRPO or otherwise)
cannot help this reader; the reader must first be trained to use motives (e.g. noisy-oracle training).

### 365128 — joint_k3 nsets4 (5:14 on hady/alarm)
loss 5.97 (step 20) -> 5.83 (step 1680); post_greedy 0.56 -> 0.61; greedy = retriever top-3 0.995 -> 0.148; null never
in the greedy set; selector entropy 2.12 -> 1.87. Test picks: = retriever top-3 0.161, distinct motives 223.
Picks contain >= 1 true motive on 0.2792 of test rows (retriever top-3: 0.3100): the selector re-ranked AWAY from true
motives, consistent with a reader that ignores motive content (the marginal likelihood cannot reward correct motives).
Decode with its own reader requested (decode_joint_k3_nsets4).

### 365309 — decode of joint_k3 picks with its own reader
HR@1/5/10 0.0405 / 0.0856 / 0.1116, NDCG@1/5/10 0.0405 / 0.0638 / 0.0721.
vs saved 364239 + retriever top-3 0.1077 / 0.0706 (+0.0039 HR@10); ≈ O0 0.1111 / 0.0729 (HR@10 +0.0005, NDCG@10 -0.0008).
Gain attributable to one more pass of reader training, not motives (reader ignores motives; picks have fewer true
motives, 0.2792 vs 0.3100). Phase 5 verdict: motive-only readers trained on retrieved motives ignore them; selection /
joint training cannot add value until motive retrieval improves. No open jobs.

### Phase 5c — joint training of the REAL cls retriever + 3-slot recommender (user request 2026-11-14)
Motivation (user): joint_k3 only re-ranked the frozen retriever's top-10 (true motive reachable on 0.4908 of test
rows; top-50 0.7268), so gradients never reached the retriever. Now no selector: the cls MLP itself (881 logits)
is trained by the recommender's marginal likelihood.
- collab/retriever_features.py caches the retriever's standardised inputs (SASRec 128 + gte 1024 + item profile 881
  + user past 881 = 2914) -> CollabRec_GRPO_select/retriever_feats/x_{train,valid,test}.npy; mlp.pt on them reproduces
  stored scores (max |diff| 0.0078 = float16 storage; top-3 agree 0.996 valid/test).
- Caveat: mlp.pt was fit on ALL train rows: train top-3 contains a true motive 0.4610 (out-of-fold 0.3407), test 0.3099.
- collab/joint_retriever_k3.py: Plackett-Luce sets of 3 over all 881 motives; S = greedy + 3 Gumbel samples; slots in
  descending score; reader lr 1e-5, retriever lr 1e-4, optional --sup_weight CE (default 0 = recommender signal
  only); test retrieval (true in top-1/3/10) logged every 200 steps. CPU dry run passed (init test 0.1944/0.3116/0.4937
  on rows with >= 1 motive). Run: MODE=joint_retriever NAME=nsets4.

### 365364 step 200 (no margin loss): retriever degrades
test true-in-top-1/3/10 0.1944/0.3116/0.4937 -> 0.1843/0.2935/0.4826; post_greedy 0.92, entropy 4.91 -> 4.81, max_p
0.129 -> 0.139. With 881 motives the sampled sets have tiny prior vs the greedy set and the reader's likelihoods barely
differ, so the marginal-likelihood gradient mostly sharpens the retriever on its own top-3 (self-reinforcement).

### Phase 5d — + contrastive support (margin) loss, user suggestion 2026-11-14
User: add IntuRec's contrastive intuition loss (= CollabRecF2's support loss, gamma; same as the other session's margin
loss) to force the recommender to use its motives. joint_retriever_k3.py --gamma/--sup_anchor: softplus(neg - pos),
pos = thought(chosen motives) . anchor, neg = thought(3 uniform random motives) . anchor; anchor = target title (or the
true motives). CPU dry run: margin starts at 0.6932 = log 2 = softplus(0) -> thought from retrieved motives is
indistinguishable from thought from random motives (direct evidence the reader ignores motive content).
Run: NAME=nsets4_gamma1_title, EXTRA="--gamma 1.0 --sup_anchor title" (retriever: reader signal only, sup_weight 0).

2026-11-14: deleted model weights of failed runs at the user's request (scratch near quota): llm_qwen1.5b_full/model,
joint_k4/reader, joint_k3_nsets4/reader (~8.9 GB). Kept train.log, picks_test.npy, selector.pt, results.json.

### 365364 step 400: test true-in-top-1/3/10 0.1827 / 0.2825 / 0.4816 (still degrading)

### Phase 5e — joint retriever + recommender + motive retrieval loss (user request 2026-11-14)
User: back to the version without the margin loss, add a motive retrieval loss. = --sup_weight 1.0: the retriever's own
training objective (multi-positive softmax CE over the 881 prototypes on the row's true motives, as mlp.pt was trained)
added to the marginal likelihood, to anchor the retriever against self-reinforcing drift. Run NAME=nsets4_sup1.
365382 (margin loss gamma 1, title anchor): preempted at step ~60 before the first resume save; margin stayed exactly
0.6932 (= log 2) through step 60 (past warmup) -> pos ≈ neg ≈ 0 (thought norm ~0.2, dot with anchor ~0): no usable
signal at gamma 1. Cancelled (user dropped the margin-loss version).

### 365364 test retrieval (top-1/3/10): step 600 0.1716/0.2828/0.4750, step 800 0.1762/0.2746/0.4734 (still degrading)
### 365391 (+ retrieval loss, sup_weight 1) step 200: 0.1949/0.3158/0.4983 (initial 0.1944/0.3116/0.4937)
### Control: retriever ALONE on the retrieval loss, same schedule (collab/retriever_sup_control.py, CPU)
step 200 0.1947/0.3164/0.4972; 400 0.1974/0.3174/0.4980; 800 0.1973/0.3192/0.4957; final (1685) 0.1985/0.3177/0.4981.
At step 200 the joint run (0.1949/0.3158/0.4983) = the control (0.1947/0.3164/0.4972): the joint run's retrieval gain
comes entirely from the retrieval loss; the recommender's signal adds ~nothing (it ignores motives). Ceiling of this
fine-tune: top-3 ~0.318 (+0.006).

### 365364 final (joint cls retriever + recommender, no extra loss; 3 h 48 min on lexicon)
Test retrieval top-1/3/10: 0.1944/0.3116/0.4937 -> 0.1745/0.2780/0.4686 (plateau from step ~800); distinct motives in
test picks 188 (original retriever top-3: 228). Retriever degraded by self-reinforcement. Decode requested
(decode_joint_retriever_nsets4).

### Phase 5f — straight-through Gumbel-softmax top-3, end to end (user's design, 2026-11-14)
User: context -> retriever -> Gumbel-softmax -> top-3 -> recommender -> loss, gradient back to the retriever. The 4-set
marginal-likelihood runs (365364 / 365391) did NOT match this: their Gumbel-max top-3 is hard (no_grad) and the retriever
gradient goes only through pi(set). New collab/joint_retriever_gumbel.py: z = scores + noise_scale * Gumbel; 3 slots in
turn, w_t = softmax(z / tau) over untaken motives, hard = argmax; forward hook on reader.model.embed_tokens for the
[N, 3, 64] motive-slot tokens returns sum_m (onehot + w - w.detach())[m] * emb(m). CPU test: max |logits(ST) - logits(hard)|
= 0.0 (forward exact), retriever grad norm 0.016 on 2 rows. Loss = -log p(target | 3 chosen motives) (+ optional sup CE).
1 reader pass per row; tau 1.0 -> 0.3 geometric; logs retriever grad norm. Dry run: sampled sets never equal the
retriever's top-3 at noise 1 (broad softmax, max_p ~0.13), pick_true 0.25 on 40 rows -> --noise_scale option added.
Run: MODE=joint_gumbel NAME=st_tau1to0.3 (noise 1.0, sup 0).

### 365489 (Gumbel, no anchor) test retrieval top-1/3/10: step 200 0.1831/0.3040/0.4880, 400 0.1480/0.2822/0.4714,
600 0.1504/0.2764/0.4628 -> degrades faster than 365364 (dense gradient on all 881 scores, sharper as tau falls).
365490 (Gumbel + sup 1) started.

### Phase 5g — fixes for the two drift causes (user request; NEW jobs, Gumbel setup, no retrieval loss)
Diagnosis: the recommender's gradient carries no information about which motive is right, yet (1) AdamW makes every
step ~lr regardless of gradient size, so a tiny uninformative gradient still drives full-size updates, and (2) mlp.pt is
already fit on these train rows, so any label-free move loses fit.
(a) --retr_optim sgd --lr_retriever 0.04: SGD step proportional to the gradient; calibrated: the retrieval-loss
    gradient is ~4.7 per 32 rows (clipped to 1 -> step 0.04), AdamW lr 1e-4 step ~0.197; the joint gradient ~0.003 ->
    step ~1e-4. Reader and retriever clipped separately in this path.
(b) collab/train_retriever_fold.py: same cls recipe fit on half A of train users (GroupKFold 2); joint training on
    half B (--retriever_ckpt retriever_foldA/mlp.pt --train_rows rows_B.npy). Fold retriever: test 0.1916/0.3092/0.4881
    (full mlp.pt 0.1944/0.3116/0.4937); half B unseen 0.2004/0.3321/0.5275 (full mlp.pt on B, fitted: 0.2894/0.4611/0.6722).
Note: a CPU dry run with out_dir in the HOME job tmp filled the home quota (~10 GB resume state); deleted, plus 4.3 GB of
old smoke outputs; dry runs now go to scratch.
365489 step 800: 0.1486/0.2777/0.4583. Cancelled at the user's request ("stop drift run"); resume_state.pt to be deleted,
train.log kept. Verdict: Gumbel-softmax with the recommender's signal only degrades the retriever (top-1 -0.046).

### 365480 — decode of 365364 (joint cls retriever + recommender, no anchor) picks with its own reader
HR@1/5/10 0.0408 / 0.0855 / 0.1116, NDCG@1/5/10 0.0408 / 0.0638 / 0.0723 — same as joint_k3 (0.1116 / 0.0721) although
its retriever got worse (test top-3 0.3116 -> 0.2780): the recommender's quality is independent of the motives it is
given (consistent with it ignoring them); the +0.004 over the saved 364239 (0.1077) is the extra pass of reader training.
365490 (Gumbel + sup 1) step 200: 0.1955/0.3171/0.4989 = control. 365505 (SGD) running.

### Drift diagnosis conclusion + user decision (2026-11-14)
365505 (SGD 0.04): test retrieval unchanged through step 600 (0.1944/0.3119/0.4932). 365506 (fold-A retriever on half B,
AdamW): still degrades (step 200 0.1585/0.2995/0.4775 from 0.1916/0.3092/0.4881) -> the fitted start is NOT the cause.
365364 retriever weights moved 16% (first layer, |dW|/|W| 0.163) under AdamW from a gradient of norm ~0.003.
Chain (user's summary): the recommender ignores its latent -> its gradient to the retriever is uninformative -> AdamW
(step ~lr regardless of gradient size) turns it into full-size updates -> retriever degrades; SGD keeps steps
proportional -> nothing changes; the retrieval loss holds it at the control. Root cause = latent-ignoring.
DECISION: keep AdamW (the default) for consistency; resolve latent-ignoring later.

- 365506 (Gumbel, fold-A retriever trained on half B, AdamW) final: test retrieval top1/top3/top10 0.1248 / 0.2862 / 0.4743 (start 0.1916 / 0.3092 / 0.4881); 134 distinct motives in picks. A retriever that has not seen these rows still drifts under AdamW, so the "already fitted" start is not the cause. No decode (the recommender ignores motives).
- 365490 (Gumbel + sup1) final: 0.1988 / 0.3177 / 0.4978; 280 distinct motives. Same as the retrieval-loss-only control (0.1985 / 0.3177 / 0.4981). Decode 365531 running.
- 365505 (Gumbel + SGD 0.04) final: 0.1944 / 0.3120 / 0.4934 (start 0.1944 / 0.3116 / 0.4937), 227 distinct motives. SGD stops the drift but changes nothing, because the recommender's gradient carries no information. No decode.
- 365391 (4-set marginal + sup1) final: 0.1982 / 0.3171 / 0.4968, 281 distinct motives; equals control. Decode 365540 submitted.
- Decode 365531 (Gumbel + sup1, own reader): HR@1/5/10 0.0410 / 0.0859 / 0.1118; NDCG 0.0410 / 0.0641 / 0.0724. Same as 365480 (.1116) and 364239 epoch 3 (.1123): the joint run gives no gain, because the recommender ignores its motives.
- Decode 365540 (4-set + sup1, own reader): HR@1/5/10 0.0407 / 0.0855 / 0.1119; NDCG 0.0407 / 0.0639 / 0.0724. Conclusion for phase 5: every joint retriever+recommender variant lands at HR@10 0.1116-0.1119, no better than 364239 (0.1123) or O0 (0.1111). Blocked by the recommender ignoring its latent (to be resolved later, per user).

## Phase 6 (2026-11-14): teach the recommender to use noisy retrieved motives
- Prior evidence: 357938-357942 (noisy oracle + gated BCE selection) learned a good gate (valid AUC 0.846 on 357939) but HR@10 stayed at 0.110-0.112, because the gate head sits beside the LM (rel_q/rel_k on the pre-injection query), so the LM still ignored <|Thought|>.
- New code: scratch/IntuRec/code/collab-motivehead (copy of collab-refined-infonce-recovered; 364239's code is untouched).
  - motive_source insert | insert_similar: valid/test = retriever top-3; train, with prob NOISY_P, a row whose top-3 misses its true motive gets true motives + fillers (retriever's non-true picks | prototypes nearest the true motive in motive_text_emb), shuffled. NOISY_P_START gives a per-epoch curriculum.
  - motive head (MOTIVE_HEAD_W): CE of the true motive (882-way) from the LM's FINAL hidden state at the last prompt position, so it must pass through the LM reading the thought. A slot-index head is impossible (the thought readout is permutation-invariant).
  - Smoke test: train true-in-slots 0.9965 at p=1 / 0.3695 at p=0; valid = top-3; head loss 6.48 at init (ln 882 = 6.78).
- Runs (EXPERIMENT=CollabRec_motivehead, k=3, no InfoNCE, MOTIVE_HEAD_W=0.1): A insert p0.5 + head; B insert p0.8 + head; C insert_similar curriculum 1.0->0.5->0 + head; C0 the same curriculum without the head. Then: decode with true / top3 / null motives; Gumbel joint training from C (user's curriculum -> joint plan).
- Submitted 2026-11-14: A 365708, B 365709, C 365710, C0 365711 (pending on researchshort).
- C0 365711 (curriculum 1->0.5->0, no head) test (best = epoch 2): HR@1/5/10 0.0414 / 0.0855 / 0.1116; NDCG 0.0414 / 0.0641 / 0.0725; valid NTP 0.6856 / 0.5896 / 0.6143 (364239: 0.6781 / 0.5866 / 0.6143). No gain. True/null decodes submitted.
- Valid retriever top-1 / top-3 hit = 0.1759 / 0.2985, the bar for the motive head. Epoch 1 head hit: A 0.1616, B 0.1422, C 0.1639; epoch 2: A 0.1685 (valid head CE went up 0.5423 -> 0.5505).
- C0 decodes: true HR@1/5/10 0.0407 / 0.0871 / 0.1128 (NDCG 0.0407 / 0.0647 / 0.0730); null 0.0381 / 0.0775 / 0.1026 (NDCG 0.0381 / 0.0586 / 0.0667); top3 0.1116. Unlike 364239 (null = top3), C0 depends on the slots (removing them costs 0.009 HR@10), but true is only +0.0012 over top3: it reads the slots without telling right from wrong motives (or 'No specific motive' x3 is just out-of-distribution for it).
- A 365708 (insert p0.5 + head) test: HR@1/5/10 0.0426 / 0.0839 / 0.1107; NDCG 0.0426 / 0.0640 / 0.0726. Head never passed 0.1759. C epoch 2 head 0.1814 (first above the bar). True/null decodes queued for A, B, C.
- B 365709 (insert p0.8 + head) test: HR@1/5/10 0.0448 / 0.0827 / 0.1069; NDCG 0.0448 / 0.0644 / 0.0722 (worst HR@10). Decodes B_true 365970, B_null 365971; A_true 365902, A_null 365903.
- C 365710 (curriculum + head) test: HR@1/5/10 0.0439 / 0.0859 / 0.1107; NDCG 0.0439 / 0.0655 / 0.0735. Decodes C_true 365972, C_null 365973.
- A decodes: true HR@1/5/10 0.0392 / 0.0843 / 0.1107 (NDCG 0.0392 / 0.0625 / 0.0709); null 0.0276 / 0.0656 / 0.0861 (NDCG 0.0276 / 0.0468 / 0.0534); top3 0.1107. A depends on its slots (null -0.025) but true = top3. To tell content use from out-of-distribution sensitivity to 'No specific motive' x3, added collab/select_picks/k3_random.npy (3 random motives per row, seed 0; overlap with true 0.010) and queued random decodes for C, C0, A, B.
- **B breaks latent-ignoring.** B decodes: true HR@1/5/10 0.0521 / 0.1141 / 0.1499 (NDCG 0.0521 / 0.0838 / 0.0954); top3 0.1069; null 0.0923 (NDCG 0.0346 / 0.0536 / 0.0605). true > top3 > null, and true beats 351000 (0.1357). C decodes: true 0.0460 / 0.0963 / 0.1296 (NDCG 0.0460 / 0.0720 / 0.0827), top3 0.1107: gap 0.019 vs B's 0.043.
- Next: Gumbel joint training (AdamW) from reader B, with and without --sup_weight 1 (joint_gumbel_st_tau1to0.3_readerB[_sup1]). Afterwards decode its reader with its own picks and with true motives, to check the reader did not drift back to ignoring motives.
- 2026-11-15 scratch at 484 G: with user OK, deleted reader/ of 5 finished joint runs (15 G; logs/picks/retriever kept). The readerB joint runs use --resume_every 0 (user choice) to avoid 10 G resume states.
- C null: HR@1/5/10 0.0273 / 0.0647 / 0.0871 (NDCG 0.0273 / 0.0466 / 0.0538). C: true 0.1296 > top3 0.1107 > null 0.0871. Joint readerB jobs: 366067 (no sup), 366068 (sup1).
- C random: HR@1/5/10 0.0254 / 0.0554 / 0.0726 (NDCG 0.0254 / 0.0406 / 0.0461), below null 0.0871: C reads motive content (true 0.1296 > top3 0.1107 > null 0.0871 > random 0.0726). Joint readerB: retr_grad 2.85 at step 20 (vs ~0.003 with reader 364239).
- Joint from reader B, final retriever on test: 366067 (no sup) 0.1794 / 0.3098 / 0.4950, 69 distinct motives in picks (retriever sharpened onto few motives, top-1 -0.015); 366068 (sup1) 0.1994 / 0.3164 / 0.4960, 286 distinct = the control. Retrieval metrics alone show no gain; end-to-end decodes (own / B + joint picks / own + true) pending: jB_own 366081, jB_Bpicks 366088, rest queue-limited.
- 2026-11-15 scratch at 490 G: with user OK, deleted model.safetensors of C (365710); A and C0 weights to be deleted after their random decodes.
- C0 random: HR@1/5/10 0.0365 / 0.0770 / 0.0971 (NDCG 0.0365 / 0.0579 / 0.0644): true 0.1128 > top3 0.1116 > null 0.1026 > random 0.0971. Deleted C0 model.safetensors (user OK).
- A random: HR@1/5/10 0.0334 / 0.0757 / 0.0996 (NDCG 0.0334 / 0.0547 / 0.0624); A: true 0.1107 = top3 0.1107 > random 0.0996 > null 0.0861. Deleted A model.safetensors (user OK). Freed 9 G total (C, C0, A).
- B random: HR@1/5/10 0.0202 / 0.0467 / 0.0617 (NDCG 0.0202 / 0.0335 / 0.0384). B: true 0.1499 > top3 0.1069 > null 0.0923 > random 0.0617, the widest spread of any reader (true - random = 0.088 HR@10).
- jB_own (366067 reader + its picks): HR@1/5/10 0.0417 / 0.0837 / 0.1075; NDCG 0.0417 / 0.0634 / 0.0711. Same as B + cls top-3 (0.1069): no end-to-end gain.
- jB_Bpicks (B + 366067 picks): 0.0414 / 0.0833 / 0.1079 (NDCG .0414/.0631/.0711) = B + cls top-3: the new retriever is neutral. jB_true (366067 reader + true): 0.0429 / 0.0905 / 0.1177 (NDCG .0429/.0673/.0761), down from B's 0.1499: joint training on noisy Gumbel picks (pick_true ~0.3) eroded the reader's motive use (true - own gap 0.043 -> 0.010).
- jBsup_own (366068 reader + picks): 0.0413 / 0.0841 / 0.1075 (NDCG .0413/.0636/.0711) = same.
- Proposed next (awaiting user): joint training with reader B frozen (--lr_reader 0), so the retriever's gradient stays informative.
- jBsup_Bpicks (B + 366068 picks): 0.0404 / 0.0847 / 0.1091 (NDCG .0404/.0632/.0710), +0.0022 over cls top-3, within noise (SE ~0.004). jBsup_true (366068 reader + true): 0.0429 / 0.0873 / 0.1118 (NDCG .0429/.0658/.0737): reader erosion even stronger (true - own gap 0.004).
- Phase 6 conclusion: B (insert 0.8 + LM-side motive head) is the first reader that uses motives (true 0.1499 > top3 0.1069 > null 0.0923 > random 0.0617). Joint Gumbel training from B (with or without the retrieval loss) erodes that within one pass (true 0.1177 / 0.1118) and yields no end-to-end gain (0.1075). Next candidate: freeze the reader (--lr_reader 0) or keep the motive-head + insertion losses during joint training.
- 2026-11-15 correction (user): the requested pipeline was curriculum reader -> joint training FROM that reader; I had run joint from B instead (my deviation), and C (head) was my addition. Rerunning the user's design: added --save_last / SAVE_LAST=1 to collab-motivehead (saves the final epoch, trained at the deployment noise ratio; the test decode always used the last epoch), reader RUN_TAG=micro2-cur1to0-last (= C0 recipe), then Gumbel joint (AdamW, --resume_every 0) from it: joint_gumbel_st_tau1to0.3_readerCur.
- CAVEAT (2026-11-16): each run's own test line = last-epoch in-memory model, but the true/null/random decodes load the saved best-valid checkpoint (epoch 2 for A/B/C/C0). So true-vs-top3 gaps mix checkpoints; true/null/random are mutually consistent. Queued B_top3 (k3_top3 on B's saved weights) to fix B's comparison; run 366304 saves its last epoch, so its decodes will be consistent.
- B_top3 (B saved ckpt + cls top-3): HR@1/5/10 0.0411 / 0.0834 / 0.1092; NDCG 0.0411 / 0.0629 / 0.0713. Single-checkpoint B: true 0.1499 > top3 0.1092 > null 0.0923 > random 0.0617: gap 0.041 holds. Joint picks on the same B: 0.1079 (366067), 0.1091 (366068) vs 0.1092: the joint retrievers give no gain.
- Coverage split (test rows where cls top-3 holds a true motive = 0.3100), HR@10 all / covered / missed: 364239 top3 0.1077 / 0.2006 / 0.0660; B top3 0.1092 / 0.2049 / 0.0663; B true 0.1499 / 0.2164 / 0.1200; B null 0.0923 / 0.1790 / 0.0533; B random 0.0617 / 0.1077 / 0.0411. Wrong retrieved motives do NOT hurt B (missed rows = 364239); correct retrieved motives add only +0.004 on covered rows. B's true-motive gain is almost all on rows the retriever misses (hard rows): those motives carry information the history lacks, which a history-based retriever cannot provide. A better retriever on the same features is capped near +0.001 HR@10. Script: job tmp split.py (meta.json per-row hits).
- Confidence analysis (cls retriever prob. mass on its top-3; job tmp conf.py): AUC 0.693 for "top-3 holds a true motive". By confidence quintile, B+top3 beats B+null in every quintile and equals 364239+top3 (top quintile 0.2315 vs 0.2307), so gating motives by confidence only lowers HR@10 (0.1092 -> 0.0972 at 20%). Confident rows are easy users (HR 0.23 vs 0.05), not rows where motives add information. Confidence cannot fix it.
- 366304 curriculum reader (save last) test, retriever top-3: HR@1/5/10 0.0423 / 0.0830 / 0.1106; NDCG 0.0423 / 0.0636 / 0.0725; valid NTP 0.6920 / 0.5901 / 0.6134. Submitted joint 367072 (readerCur), decodes Cur_true 367073, Cur_null 367074.

### Joint training FROM SCRATCH (user request 2026-11-16: "try both of them")
All earlier joint runs started from a trained reader (364239 / B / curriculum 366304) + trained cls retriever, one pass.
New: the retriever is inside the normal CollabRec training (base Qwen2.5-1.5B, 3 epochs, batch 128, lr 3e-4, MICRO 2,
k=3 inject, no InfoNCE, = 364239's recipe) via motive_source `gumbel` (code copy
/common/scratch/.../code/collab-gumbel-scratch: gumbel_joint.py + patched run_collabrec.py / CollabRecDataset.py).
Train: straight-through Gumbel top-3 (tau 1.0 -> 0.3 over all steps); valid: deterministic top-3 of the current
retriever; test: final retriever's top-3 (picks_test.npy). Retriever in the same AdamW + linear schedule, own lr.
- (a) RETRIEVER_INIT=cls, LR_RETRIEVER=1e-4 (lr as all earlier joint runs)
- (b) RETRIEVER_INIT=random, LR_RETRIEVER=1e-3 (my choice: the cls recipe's lr, since it must learn from zero)
Checks: CPU smoke (6 rows) runs train -> valid -> save (+retriever.pt) -> test picks -> decode; ST forward NTP equals the
hard-token NTP exactly (5.5817 / 6.5297); retriever grad norm 0.63 (cls) / 0.42 (random); cls init valid top-1 0.1767.
Compare against 364239 (fixed cls top-3 from the start): test 0.0441/0.0867/0.1123 (own line).
- 367072 (Gumbel joint from curriculum reader 366304, AdamW) final retriever on test 0.1939 / 0.3041 / 0.4838 (start
  0.1944 / 0.3116 / 0.4937), 132 distinct motives; last steps pick_true 0.29, max_p 0.32 (up from ~0.13): the retriever
  sharpens without getting more accurate, as from B. Decodes requested: jCur_own, jCur_Rpicks (366304 + joint picks), jCur_true.
- Cur_true 367073 (366304 saved last-epoch ckpt + true motives): HR@1/5/10 0.0444 / 0.0849 / 0.1137; NDCG 0.0444 / 0.0655 / 0.0747.
  Cur_null 367074: 0.0377 / 0.0800 / 0.1036; NDCG 0.0377 / 0.0597 / 0.0674. Own top-3 (same ckpt): 0.1106.
  true - top3 = +0.003, top3 - null = +0.007: the curriculum reader (like C0) depends a little on its slots but barely
  reads their content; unlike B (true 0.1499). Assumption 1 not met for this reader.
- 367072 decodes: jCur_own 367205 HR@1/5/10 0.0401 / 0.0785 / 0.1043 (NDCG .0401/.0601/.0684); jCur_Rpicks 367206 (366304 +
  joint picks) 0.0432 / 0.0825 / 0.1095 (NDCG .0432/.0636/.0724); jCur_true 367207 0.0404 / 0.0795 / 0.1055 (NDCG .0404/.0609/.0692).
  vs 366304 + cls top-3 0.1106: joint picks neutral on the same reader (-0.001); joint reader worse (-0.006) and
  ignores motive content (true - own +0.001; 366304 had +0.003). Curriculum -> joint gives no gain, as from B / 364239.
- RESULTS (8h12m each, lunar): 367195 (cls init, lrr 1e-4) valid NTP 0.6746/0.5848/0.6187; test HR@1/5/10 0.0427 / 0.0849 / 0.1103,
  NDCG 0.0427 / 0.0647 / 0.0729; retriever test top-1/3/10 0.1944/0.3116/0.4937 -> 0.1959/0.3122/0.4940 (unchanged), 224 motives;
  retr_grad 0.08 (start) -> 0.001 (end), pick_true 0.20-0.23.
  367196 (random init, lrr 1e-3) valid NTP 0.6794/0.5866/0.6176; test 0.0421 / 0.0844 / 0.1100, NDCG 0.0421 / 0.0636 / 0.0719;
  retriever stays at chance (test top-10 0.0193 -> 0.0197), retr_grad 0.05 -> 0.0003, pick_true 0.003.
  => random motives give the same HR as cls motives (0.1100 vs 0.1103 = 364239 0.1123 = O0 0.1112): trained from scratch with
  the retriever in the loop, the recommender still learns to ignore the slots, and the gradient to the retriever vanishes
  (end-to-end NTP signal alone cannot bootstrap motive use). Motive use needs an explicit signal (insertion + LM motive head, B/C).
- 2026-11-18 (user OK): deleted model weights of failed runs 367195, 367196, 366304 (model.safetensors) and joint readerB / readerB_sup1 / readerCur (reader/); logs, picks, retriever.pt, results kept. B 365709 and 364239 kept.
