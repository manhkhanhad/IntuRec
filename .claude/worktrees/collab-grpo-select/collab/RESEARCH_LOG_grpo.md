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
