# MotiveRec research log (worktree `motiverec`, branch `motiverec`)

MotiveRec = CollabRec (text mode) with predicted motive-prototype supports (`motive_corpus/stage2`), without the
collaborative InfoNCE of `collab-refined-infonce`. Paper effort started 2026-11-13. Metrics are values, not %.
Oracle-motive runs are development-only ceilings and are not used as paper evidence (user, 2026-11-13).

## 2026-11-13 — pilot studies (motivation section)
Code: `pilot/` (`common.py`, `pilot_a_motive_breakdown.py`, `pilot_bc.py`, `pilot_c_r3.py`, `submit_pilot_c.sbatch`).
Outputs: `<scratch>/IntuRec/checkpoints/<dataset>/MotiveRec_pilot/` (`pilot_a.json`, `pilot_b.json`, `pilot_c.json`, `r3/`).
Models (all Qwen2.5-1.5B, same test split; R3 and IntuRec CSVs are byte-identical): O0 = backbone with a
thought slot but no supports (351003 / 350999), IntuRec F2 micro 2 (349506 / 348597), original LatentR3 stage 1
SFT (`R3_ori/.../latent`) and stage 2 RL (`R3_ori/.../nomul_attn_lr_5e-5`). Motive facets (13 coarse purchase-motive
types per dataset, from v5 rationales) are analysis labels only.

### Pilot A — accuracy by purchase-motive facet (CPU)
HR@10 per facet of the target purchase (a row with 2 facets counts in both):

CDs (n 6157): O0 / IntuRec / R3-SFT / R3-RL
- occasion_gift_seasonal (223) 0.2915 / 0.2780 / 0.2825 / 0.2870; artist_loyalty (179) 0.1955 / 0.2123 / 0.1676 / 0.2123;
  collection_completion (928) 0.1994 / 0.2004 / 0.1929 / 0.2015
- discovery_novelty (804) 0.0759 / 0.0796 / 0.0672 / 0.0784; live_performance (346) 0.0925 / 0.0751 / 0.0665 / 0.0751;
  compilation_best_of (2204) 0.1003 / 0.0998 / 0.0649 / 0.0789
- overall 0.1178 / 0.1202 / 0.0989 / 0.1095
Toys (n 6738):
- franchise_brand_loyalty (402) 0.3134 / 0.2910 / 0.2935 / 0.2886; collection_completion (1898) 0.2445 / 0.2455 / 0.2276 / 0.2313;
  interest_theme_affinity (526) 0.1692 / 0.1692 / 0.1654 / 0.1711
- active_outdoor_play (605) 0.0397 / 0.0397 / 0.0281 / 0.0331; quality_safety (1042) 0.0518 / 0.0585 / 0.0451 / 0.0461;
  age_stage_fit (144) 0.0625 / 0.0486 / 0.0417 / 0.0417; value_price (874) 0.0595 / 0.0641 / 0.0538 / 0.0549
- overall 0.1112 / 0.1112 / 0.1023 / 0.1055

- Across-facet std of HR@10: CDs 0.057–0.061, Toys 0.076–0.079 vs permutation null 0.012–0.016 (95% ≤ 0.024), p < 0.0005 for every model; best/worst facet ratio 3.7–4.4 (CDs), 7.3–10.5 (Toys).
- Controlled: logistic hit@10 ~ target–history title Jaccard + log target popularity + log history length, adding facet indicators: LR test p ≤ 1e-11 for every model and dataset (CDs LR 131–221, Toys 77–87, df 13). Facet effects are not explained by lexical similarity / popularity / history length.
- Same facet profile for all four models; latent reasoning (IntuRec, R3) does not close the gap on the hard facets (Toys active_outdoor_play: 0.0397 / 0.0397 / 0.0281 / 0.0331).
- Motive continuation vs shift (target facet already in the user's own train-row facets?): not significant (CDs +0.019 to +0.027, CIs include 0; Toys −0.008 to −0.013, n.s.). Not usable as a pilot claim. Rows of users without train rows have higher HR (0.12–0.18) — unexplained, not investigated.
- Caveat: facets come from an LLM that saw the target and review, so a facet is partly a description of the target's purchase type; they are labels for stratification, as popularity buckets are.

### Pilot B — does the latent thought encode the purchase motive? (CPU)
Linear probe (13 sigmoid outputs, standardised features, 5-fold GroupKFold by user, weight decay best of 1e-4/1e-3/1e-2), macro ROC-AUC over facets:

| representation | CDs | Toys |
|---|---|---|
| Gaussian noise (floor) | 0.5031 | 0.5084 |
| history text (gte-large) | 0.6237 | 0.6632 |
| IntuRec thought | 0.5855 | 0.6627 |
| LatentR3-SFT thought | 0.5372 | 0.4997 |
| LatentR3-RL thought | 0.5178 | 0.5026 |
| history + IntuRec thought | 0.6224 | 0.6649 |
| history + R3-SFT thought | 0.6224 | 0.6601 |
| history + R3-RL thought | 0.6175 | 0.6606 |

- R3 thoughts carry (almost) no motive information (Toys at chance; CDs 0.52–0.54).
- IntuRec's thought, despite effective rank ≈ 1, is linearly decodable to about the history's level (Toys 0.6627 vs 0.6632): the information sits in tiny-variance directions (the probe standardises), and is a copy of what the history says.
- No thought adds motive information beyond the history (history + thought − history: −0.0062 to +0.0017).
- History alone predicts the motive facet only weakly (0.62–0.66), consistent with the retriever ceiling (P@1 ≈ 0.2–0.3).

### Pilot C — latent collapse (geometry CPU; R3 interventions GPU, jobs 364570–364573)
Geometry of test thoughts:

| model | dataset | eff. rank | PC1 var | random-pair cos |
|---|---|---|---|---|
| IntuRec | CDs | 1.02 | 0.998 | 0.9524 |
| LatentR3-SFT | CDs | 9.01 | 0.307 | 0.9444 |
| LatentR3-RL | CDs | 1.55 | 0.905 | 0.9947 |
| IntuRec | Toys | 1.05 | 0.992 | 0.9955 |
| LatentR3-SFT | Toys | 2.27 | 0.631 | 1.0000 |
| LatentR3-RL | Toys | 2.30 | 0.607 | 1.0000 |

(Toys R3 pair cosine 1.0000: one constant vector; the residual rank is fp16 noise — the K = 1 limit of R3's pooling collapse, see R3 Latent-utilization log U3.)

IntuRec interventions (existing decodes 355229 / 355228), ΔHR@10 vs raw [95% CI]:
- CDs raw 0.1191: mask −0.0050 [−0.0076, −0.0026], mean −0.0029 [−0.0054, −0.0005], shuffle −0.0016 [−0.0041, +0.0006], true target put into the intuition list +0.0000 [−0.0016, +0.0016], target removed −0.0016 [−0.0037, +0.0005].
- Toys raw 0.1119: mask −0.0013 [−0.0031, +0.0004], mean −0.0010 [−0.0025, +0.0004], shuffle −0.0016 [−0.0031, −0.0001], target put in −0.0013 [−0.0027, −0.0001], target removed −0.0010 [−0.0022, +0.0000].
R3 interventions (raw / mask / train-mean / shuffle_user / shuffle_target, R3's own `experiment.py`, outputs redirected to scratch): submitted via JobManager 364570 (CDs SFT), 364571 (CDs RL), 364572 (Toys SFT), 364573 (Toys RL). Partial results (JobManager):
- raw: CDs SFT 0.0989 / NDCG 0.0717, RL 0.1100 / 0.0808; Toys SFT 0.1011 / 0.0654, RL 0.1049 / 0.0673 (CDs matches `final_result.json`, which gives 0.0989 / 0.1095; CDs catalog violations 80).
- CDs mask: SFT and RL both give HR@10 0.07309, NDCG@10 0.04196, identical to every digit. This is expected. A tensor-by-tensor comparison of the two `model.safetensors` shows that only 6 of 344 tensors differ, all in the pooling module `attention.{query,key,value}.{weight,bias}`. RL stage 2 trains only that module, so the LLM decoder is the same in both, and masking the thought (zero vector) gives the same model.
  - Consequence: the whole RL gain (0.0989 → 0.1100 on CDs) passes through the thought vector, even though the RL thought has effective rank 1.55 and pair cosine 0.9947.
  - Caveat: zero is off-distribution, so the train-mean condition is the fairer "no information" control.
- CDs train-mean thought (one constant vector for every user): SFT 0.0994 / 0.0717 (raw 0.0989 / 0.0717), RL 0.1104 / 0.0808 (raw 0.1100 / 0.0808). The constant mean thought matches the per-user thought, so R3's thought carries no usable per-user information. RL's gain is a learned constant offset (a soft prompt), not per-user reasoning.
- Toys mask: SFT and RL both 0.0938 / 0.0525 (raw 0.1011 / 0.0654 and 0.1049 / 0.0673). Again identical, because on Toys too RL changes only the pooling attention module. Masking costs 0.007–0.011 HR@10, less than on CDs (0.026–0.037).
- CDs shuffle_user (each row gets another user's thought): SFT 0.0997 / 0.0718, RL 0.1101 / 0.0809, against raw 0.0989 / 0.0717 and 0.1100 / 0.0808. The model is unaffected, so R3's thought carries no per-user information.
- CDs complete (jobs 364570 / 364571, exit 0). ΔHR@10 vs raw [95% CI] from `pilot_bc.py collapse`:
  - SFT (raw 0.0989): mask −0.0258 [−0.0313, −0.0210]; mean +0.0005 [−0.0005, +0.0015]; shuffle_user +0.0008 [−0.0003, +0.0019]; shuffle_target +0.0008.
  - RL (raw 0.1100): mask −0.0369 [−0.0429, −0.0315]; mean +0.0005 [−0.0006, +0.0018]; shuffle_user +0.0002 [−0.0013, +0.0016]; shuffle_target +0.0002.
- shuffle_target gives the same hit rank as shuffle_user on 100% of rows. This is by construction in R3's `experiment.py` (lines 641–650): both modes take the same hashed donor index, `(sample_id·104729 + seed·7919) mod n_train`, and step forward only when the user or the target collides. Collisions are rare, so the donor is almost always the same row. Report a single "shuffle" row in the paper.
- Toys train-mean: SFT 0.1014 / 0.0655 (raw 0.1011 / 0.0654), RL 0.1048 / 0.0673 (raw 0.1049 / 0.0673). Same result as CDs: a single constant thought matches the per-user thought.
- Remaining: Toys shuffle_user / shuffle_target.

### MotiveRec's own fused thought (same tests as Pilot C; development analysis, not part of the pilot)
Thought = InjectNet(history query, 10 predicted motive prototypes). Toys run 345291: thought dump and decodes come from the earlier `collab-refined-infonce/collab/latent_diagnostics.py` job 349725. Geometry uses `pilot/common.collapse_stats`.

| Toys | eff. rank | PR | PC1 var | PCs 90% | pair cos | cos to mean |
|---|---|---|---|---|---|---|
| MotiveRec thought | 15.53 | 9.56 | 0.200 | 14 | 0.5397 | 0.7348 |
| history query (pre-injection) | 131.96 | 30.19 | 0.135 | 337 | 0.5107 | 0.7144 |
| IntuRec thought (reference) | 1.05 | 1.02 | 0.992 | 1 | 0.9955 | 0.9976 |

Interventions, Toys (raw 0.1101 / NDCG 0.0702), dHR@10 [95% CI]:

| intervention | dHR@10 | top-10 overlap with raw |
|---|---|---|
| mask | +0.0004 [−0.0012, +0.0019] | 0.922 |
| mean | −0.0007 [−0.0019, +0.0004] | 0.932 |
| shuffle | −0.0006 [−0.0019, +0.0007] | — |
| swap motives | −0.0004 [−0.0018, +0.0009] | — |
| random motives | −0.0007 [−0.0021, +0.0006] | — |
| (dev only) oracle motives | +0.0001 | — |

Reading: the thought is not geometrically collapsed (rank 15 vs IntuRec's 1). It varies between rows, and the earlier summary shows it groups rows by user and by motive. But the decoder does not use it: removing it or swapping its motives changes no metric. The IntuRec critique therefore currently applies to MotiveRec too.

CDs run 359280 (job 364681): thoughts dumped; decodes still running.

| CDs | eff. rank | PR | PC1 var | PCs 90% | pair cos | cos to mean |
|---|---|---|---|---|---|---|
| MotiveRec thought | 8.78 | 5.39 | 0.353 | 8 | 0.6865 | 0.8281 |
| history query | 137.97 | 28.83 | 0.140 | 354 | 0.6857 | 0.8278 |

On CDs, MotiveRec's effective rank (8.78) is about the same as R3-SFT's (9.01). The difference is in pair cosine: 0.69 against 0.94 for R3-SFT and 0.95–0.99 for IntuRec and R3-RL. So the paper's collapse claim on CDs should rest on pair cosine, cosine to mean and PC1 share, not on rank alone. Mean thought norm 0.19.

## 2026-11-13 — paper framing (author decisions) and MotiveRec-E
- Framing: motivation = latent collapse, geometry only (eff. rank, PC1, pair cosine); interventions are left for a separate paper. Design = motive mining → motive retrieval → motive integration (E: embedding, T: text). The two-level reader of the T variant is cited as IntuRec's pooling design. Novelty = motive grounding. Draft: `paper/main.tex` (v0) + `paper/refs.bib`. No LaTeX on the cluster; compile locally or on Overleaf.
- Accuracy check (HR@10 / NDCG@10): CDs O0 0.1178 / 0.0864, IntuRec 0.1202 / 0.0860, MotiveRec-T 0.1174 / 0.0838; Toys O0 0.1112 / 0.0729, IntuRec 0.1112 / 0.0721, MotiveRec-T 0.1072 / 0.0705. MotiveRec-T is below IntuRec on both datasets.
- The old `emb_k10_*` runs (335693 / 345290) used `collab/stage2` (collaborative rationales), not motives. A MotiveRec-E run did not exist yet.
- Built `collab/build_motive_emb_support.py` → `motive_corpus/stage2_emb/`:
  - table = [train rationales (= stage2 table) ; prototypes], with prototypes embedded by all-mpnet-base-v2 and centred on the train-rationale mean;
  - support ids shifted by n_train; own_rationale copied;
  - no code change needed.
  - Check: mean cosine from each prototype to its nearest rationale is 0.635 (CDs) / 0.679 (Toys).
- Training: 364727 (CDs), 364728 (Toys, 48 GB GPUs) via JobManager; run folders `emb-motive_k10_beta0.1_rs1_lr0.0003_job<id>`.

## 2026-11-13 — E8a motive-mining diagnostics (CPU)
`pilot/mining_diagnostics.py`, `pilot/mining_merge_sweep.py`; report `Report/experiments/06_motive_mining/report_mining_diagnostics.md`.
- From history, discrete labels are predicted no better than popularity (Toys R@10 0.265 vs 0.257; CDs 0.362 vs 0.384); user's past motives also ≈ popularity.
- Continuous rationale space has target-free signal: cos to history items' other buyers 0.18–0.21 vs random 0.00; same user 0.12–0.15; same target item 0.26–0.43.
- Merging synonyms (semantic/behavioural, K 600→50) does not raise the history lift and lowers oracle next-item HR@10 (Toys 0.138 → 0.075 at K=100).
- 29–41 generic prototypes cover 50% of labels. Oracle motives alone rank next item at HR@10 0.138 (Toys), 0.076 (CDs).
- Proposed: soft labels, utility-scored vocabulary, persistent vs situational motives.

## 2026-11-13 — E8c motive–interaction aligner on v5 (CPU, Toys)
`collab/train_motive_aligner.py` (IRLLRec-inspired: history q, motive text t / prototypes p, item e; ce + hm + mi + match).
Test MRR 0.290–0.296 for all variants (cls retriever 0.2895): alignment does not lift v5 label retrieval (labels are the ceiling).
mi loss makes retrieved motives point to the next item in SASRec space: item HR@10 0.0024 (ce only) → 0.0407 (full).
Next: run on v6 motives when job 364815 finishes. Report: Report/experiments/06_motive_mining/report_aligner_v5.md.

## 2026-11-13 — E8d interaction-fitted relabelling on v5 (CPU, Toys)
Label = top-2 of p(m|item)·p(m|history, OOF)/prior^γ, 2 rounds. γ=0 collapses (170 motives, 28% one motive). γ=1 keeps 849.
Fixed scorer (Σπ p(item|m), v5 links) test HR@10: cls retriever 0.0439, aligner r0 0.0350, relabel γ=1 r2 0.0413 → no gain over cls.
Learned item tower is unstable (valid/test disagree). Next: apply to v6. Report: Report/experiments/06_motive_mining/report_relabel_v5.md.

## 2026-11-13 — E8e motive reliability by history length (Toys)
History length 1–10 (capped). GEARec reachability flat on train (rank≤20 0.24–0.26), test best at len 2 (0.298), worst 6–10 (0.215).
v5 retrieval lift over popularity largest at len 1 (MRR +0.115), smallest at 6–10 (+0.023). Test len-1 GEARec coverage only 38%.
HR@10 6–10: O0 0.0935, IntuRec 0.0926, MotiveRec-T 0.0877, curriculum 0.0988. Long histories are the weak spot, not short ones; no min-history filter.

## 2026-11-13 — E8b v6 mining results (job 364815, Toys)
P1000 K=998, largest share 0.26%. History-item profile R@10 0.215 vs popularity 0.026 (8×; v5 1.03×), 0.90 of target-item oracle (v5 0.66).
Oracle true-motive next-item HR@10 0.181 (v5 0.138). Prototype reachability top-20 0.094 / top-100 0.275 (v5 0.059 / 0.140). Choose P1000.
v6 P1000 retrieval: cls MLP test R@10 0.237 / MRR 0.206 (pop 0.026 / 0.027). Fixed-scorer next-item HR@10 test: v6 cls 0.0695 vs v5 cls 0.0439.
Aligner (A 0.0626, D 0.0623) and relabel γ=1 (0.0622) do not beat the cls retriever on v6 → aligner dropped.

## 2026-11-13 — E8g aligner v2 (user × item motives, averaged v6 texts, CPU, Toys)
Fixed-scorer test HR@10: best 0.0675 (ce only) / 0.0672 (all) vs plain cls 0.0695 → no gain; generated per-user/item texts (b) not pursued.
User+item alignment raises v6 label retrieval (R@10 0.2508, MRR 0.2169 vs cls 0.2368 / 0.2058). Joint read-out worse. Report: report_aligner2_v6.md.

## 2026-11-13 — E8h IRLLRec-faithful motive model (submitted)
- `collab/irllrec_motive.py` (port of official lightgcn_int: LightGCN d32 L3, BPR, RLMRec kd 1e-2, K=128 intents + adaptive mask, kd_int 2e-2 vs v6 need/kind texts, kd_int_2/itm 1e-7). Rows = user nodes; graph = row→history items; BPR on train targets.
- Read-outs on v6 P1000 prototypes: need (user motive), kind (item motive from top-50 next items), need+kind, need+cond_kind.
- Prep done (59,494 needs / 74,639 kinds mxbai). CPU 68 s/epoch → moved to GPU (device support added).
- 3-epoch CPU smoke (test): need R@10 0.1283 / fixed HR@10 0.0476; kind 0.1155 / 0.0309; need+kind 0.1475 / 0.0435; LightGCN HR@10 0.0462. Reference plain cls retriever R@10 0.2368 / fixed HR@10 0.0695.
- Job 365134 (`submit_irllrec_motive.sbatch`, tags irllrec + plain ablation) PENDING via JobManager.
- 365134 done (17 min). Test: need+kind R@10 0.1781, fixed HR@10 0.0560 (best IRLLRec read-out); need+cond_kind 0.0484; cls retriever 0.2368 / 0.0695 → IRLLRec dropped. LightGCN rec test HR@10 plain 0.0974 vs IRLLRec 0.0775 (alignment hurts). Report: Report/experiments/06_motive_mining/report_irllrec_v6.md.
- 365120 MotiveRec-T v6 Toys done (epoch-3 model): HR 0.0401/0.0818/0.1113, NDCG 0.0401/0.0613/0.0709 (v5 0.0410/0.0846/0.1072 | 0.0410/0.0632/0.0705; IntuRec 0.0420/0.0833/0.1112 | 0.0420/0.0632/0.0721). HR@10 +0.0041 vs v5, @1/@5 lower; within ~1 SE. Valid NTP best at epoch 2.

## 2026-11-14 — MotiveRec-T v6 with K=3 supports (cls vs aligner)
- Run 1: cls retriever supports, SUPPORT_K=3 (first 3 cols of stage2) → job 365345.
- Run 2: aligner v2 user+item (w_inter 0), 5-fold cross-fitted for train rows (`train_motive_aligner2.py --crossfit-folds 5 --tag user_item_cf`) → supports `motive_corpus_v6/P1000/stage2_aligner/` → submitted via JobManager (RUN_TAG motivev6aligner).
  - Aligner test: R@1 0.0609, R@10 0.2536, MRR 0.2165, fixed HR@10 0.0640 (cls: 0.0574 / 0.2368 / 0.2058 / 0.0695). Train OOF R@10 0.3025.
  - True motive in top-3 / top-10: aligner train 0.294/0.492, valid 0.245/0.427, test 0.237/0.425; cls train 0.264/0.449, valid 0.225/0.410, test 0.225/0.396.
  - Caveat: item-motive texts include held-out rows' texts (mild leak) → aligner's train-test gap in top-3 hit is 0.057 vs 0.039 for cls.

## 2026-11-14 — InjectNet v2: LLM last-layer mean pooling of motives (E9a, submitted)
- Before: InjectNet pools each motive text by one attention step over raw token embeddings (query = last token), then the history query attends over the pooled supports.
- New `motive_pool=llm_mean` (CollabRecF2, worktree only): each motive/rationale text → Qwen backbone → masked mean of last-layer hidden states (positions from 0 per text, duplicates encoded once, all-pad columns trimmed) → InjectNet's second stage (W_q_2/W_k_2/W_v_2) unchanged. Applied to supports, own rationale and negatives (user choice). Negatives: the micro-batch shares one row's 63 random negatives (cost).
- `inject` mode verified bit-identical to the main-repo CollabRecF2; llm_mean checked on a tiny Qwen (finite loss, backbone gets gradients, pad- and duplicate-invariant).
- sbatch: new CODE_DIR (default main repo) and MOTIVE_POOL (default inject, flag passed only when not default).
- Run: Toys, v6 cls supports K=10, β 0.1, MICRO 4 (accum 32), CODE_DIR = worktree → compare with 365120 (0.0401/0.0818/0.1113 | 0.0401/0.0613/0.0709).
- 365355 (llm_mean, no clipping) diverged in epoch 1: running-avg NTP tracked 365120 to step 320 (1.2988 vs 1.2867), then 1.3283 / 1.4282 / 1.6521 at steps 330-350 (per-step NTP ~9). Rationale loss lagged throughout (3.94 vs 3.73 at step 320). Cancelled (user OK).
- Added `--grad_clip` (default 0 = off) to run_collabrec.py and GRAD_CLIP to the worktree sbatch; resubmitting llm_mean with GRAD_CLIP=1.0.
- 365345 (v6, cls, K=3) done: HR 0.0424/0.0847/0.1097, NDCG 0.0424/0.0642/0.0722 (K=10 365120: 0.0401/0.0818/0.1113 | 0.0401/0.0613/0.0709; IntuRec 0.0420/0.0833/0.1112 | 0.0420/0.0632/0.0721). K=3 ≥ IntuRec on @1/@5 and NDCG@10, −0.0015 HR@10; all within ~1 SE.
- Why llm_mean diverged: not a pooling bug (llm_mean matches per-text unpadded reference to 2e-5). Qwen2.5-1.5B final RMSNorm weight mean 3.46 (max 8.9) → every last-layer token has norm 170-220, pooled ~170 vs token embeddings ~1.0; a few dims dominate (110/67/60 vs median 0.68), cosine between motives 0.967 (token-emb mean 0.484). The thought (built from these) was inserted as an input embedding ~170x too large.
- New `motive_pool=llm_mean_ln`: same mean, then parameter-free LayerNorm / sqrt(d) → pooled norm 1.0, thought norm 0.75 (token emb 1.016). Shared direction remains (cosine 0.967). Submitted without clipping (user: keep consistency).
- 365351 (v6, aligner, K=3) done: HR 0.0405/0.0824/0.1070, NDCG 0.0405/0.0621/0.0700 → below cls K=3 on all metrics. Report section E8i added to report_mining_v6.md.
- Submitted MotiveRec-T v5 K=3 (as 345291 but SUPPORT_K=3, motive_corpus/stage2) to separate the K=3 gain from v6.
- 365464 (llm_mean + clip 1.0) cancelled at user request; 365470 (llm_mean_ln, no clip) is the comparison.

## 2026-11-14 — MotiveRec objective = NTP only (user decision)
- User: MotiveRec uses only the next-item loss; remove L_align (β → 0) and L_collab; L_support γ stays 0.
- Worktree defaults changed: run_collabrec.py --beta 0.0, submit_collabrec.sbatch BETA=0 (main repo untouched). Report/main.md §2.4 rewritten.
- All earlier MotiveRec-T runs (incl. running 365470, 365477) used β 0.1.
- Deleted (user request, wrong loss β 0.1) run folders: Toys 364728 (E), 365120, 365345, 365351, 365355, 365464; CDs 364727 (E). 365470 / 365477 being cancelled, folders to delete after. Kept: 345291 (older, GRPO init), β-0 k-sweep runs (other session). Reports marked superseded.
- NTP-only reruns (β 0, CODE_DIR worktree) requested via JobManager: a v6 K10, b v6 K3, c v6 aligner K3, d v5 K10, e v5 K3, f v6 K10 llm_mean_ln, g E Toys, h E CDs.
- 365470 / 365477 cancelled and folders deleted. Reruns: a=365544 b=365542 c=365546 d=365545 e=365543 f=365549 g=365547 h=365548 (β 0).
- b 365542 (v6 K3, β0) done: HR 0.0421/0.0837/0.1128, NDCG 0.0421/0.0637/0.0730 (IntuRec 0.0420/0.0833/0.1112 | 0.0420/0.0632/0.0721; O0 HR@10 0.1112, NDCG@5/10 0.0650/0.0729). Valid NTP 0.6347/0.5554/0.5824.
- d 365545 (v5 K10, β0) done: HR 0.0404/0.0844/0.1101, NDCG 0.0404/0.0636/0.0719. Valid NTP 0.6394/0.5542/0.5802. f 365549 started.
- g 365547 (MotiveRec-E v5 K10 Toys, β0) done: HR 0.0402/0.0839/0.1094, NDCG 0.0402/0.0628/0.0710. Valid NTP 0.6425/0.5568/0.5828.
- c 365546 (v6 aligner K3, β0) done: HR 0.0404/0.0810/0.1082, NDCG 0.0404/0.0616/0.0703 → below b (cls K3) on all metrics again.
- h 365548 (MotiveRec-E v5 K10 CDs, β0) done: HR 0.0557/0.0901/0.1135, NDCG 0.0557/0.0736/0.0810 (CDs IntuRec 0.0580/0.0981/0.1202 | 0.0580/0.0789/0.0860). Valid NTP 1.0959/0.9837/1.0997.
- e 365543 (v5 K3, β0) done: HR 0.0423/0.0844/0.1122, NDCG 0.0423/0.0639/0.0728 ≈ b (v6 K3) 0.0421/0.0837/0.1128 | 0.0421/0.0637/0.0730 → v5 = v6 at K=3.
- a 365544 (v6 K10, β0) done: HR 0.0417/0.0819/0.1097, NDCG 0.0417/0.0624/0.0713 vs d (v5 K10) 0.0404/0.0844/0.1101 | 0.0404/0.0636/0.0719 → v5 ≈ v6 at K=10 too; K=3 > K=10 for both.
- f 365549 (v6 K10 llm_mean_ln, β0) done: HR 0.0398/0.0839/0.1095, NDCG 0.0398/0.0625/0.0708. All 8 reruns done → Report/experiments/07_ntp_only/report_ntp_only.md (E9).

## 2026-11-15 — L_align with a v6 target (E9b, submitted)
- User: try L_align again (β 0.1) but keep code default β 0.
- Until now L_align always used v5 reason sentences (`--rationale_tag v5`), even for v6 runs. Built `data/Toys_and_Games/collab/rationales_facet_{train,valid}_v6.jsonl`: each row's own v6 extraction, "need -> item kind; need -> item kind" (lower-case). Rows without a GEARec reasoning fall back to the v5 reason (train 741 = 1.4%, valid 457 = 6.8%; field `source`). Tokens median 26, p95 33 (cap 64).
- sbatch: new RATIONALE_TAG (default v5; passed and added to run name as -rat<tag> only when not v5). run_collabrec auto name likewise.
- Run: v6 cls K=3, BETA=0.1, RATIONALE_TAG=v6 → compare with b 365542 (β 0: 0.0421/0.0837/0.1128 | 0.0421/0.0637/0.0730).
- 2026-11-15 user: all MotiveRec runs from now use 3 motives. Worktree defaults: SUPPORT_K=3 (sbatch), --support_k 3 (run_collabrec).
- 2026-11-15 user: rerun LLM mean-pool+LN and MotiveRec-E with 3 motives (β 0): f3 (v6 cls K3 llm_mean_ln, micro 8), g3 (E Toys v5 K3), h3 (E CDs v5 K3) submitted via JobManager.
- Job IDs: f3=365890, g3=365891, h3=365892 (pending).
- 2026-11-15 user: run b with micro-batch 2 (effective batch 128 kept; RUN_TAG motivev6-micro2). Note: the β-0 k-sweep and O0 used micro 2 (k3 364239: 0.0441/0.0867/0.1123 | 0.0441/0.0662/0.0744).
- 365889 (v6 K3, β0.1, v6 target) done: HR 0.0426/0.0831/0.1103, NDCG 0.0426/0.0635/0.0723 → no gain over NTP-only b; added to report as E9b.
- h3 365892 (E CDs v5 K3, β0) done: HR 0.0575/0.0911/0.1140, NDCG 0.0575/0.0751/0.0824 (h K10: 0.0557/0.0901/0.1135 | 0.0557/0.0736/0.0810; CDs IntuRec 0.0580/0.0981/0.1202 | 0.0580/0.0789/0.0860).
- f3 365890 (v6 K3 llm_mean_ln, β0, micro 8) done: HR 0.0413/0.0840/0.1080, NDCG 0.0413/0.0633/0.0710 → below b (InjectNet) on HR@10/NDCG@10, ≈ on HR@5.

- 2026-11-15 g3 365891 (E Toys emb K3 β0): test 0.0390/0.0831/0.1091 | 0.0390/0.0616/0.0700 — below g (K10) and b. micro2 365893 still running.
- 2026-11-15 b-micro2 365893 (v6 K3 β0 MICRO 2): test 0.0430/0.0879/0.1129 | 0.0430/0.0662/0.0743 — beats b (MICRO 8) and IntuRec Toys on all six metrics. Single seed.

### 2026-11-16 — reader-B recipe on v6 (user: make the model use motives; "try it with v6")
- Port of GRPO session's reader B (365709: insert p 0.8 + LM-side motive head 0.1, v5) into the motiverec worktree:
  `--motive_corpus motive_corpus_v6/P1000 --insert_p 0.8 --insert_fill {pred,similar} --motive_head_weight 0.1`
  (data/CollabRecDataset.py motive_slots; model/CollabRecF2.py motive_head over the final hidden state; 999-way incl. "no motive").
  Total loss = NTP + 0.1·motive-head CE (user OK'd the head as the exception to NTP-only for this experiment).
- Decode diagnostics: `--eval_split valid --eval_motives {top,true,null,random}` (true = oracle, valid only), results in result/standard/<split>_<motives>/.
- CPU checks: train true-in-slots 0.82 at p 0.8 vs 0.24 at p 0 (retriever); similar fill p 1 → 0.98; head CE at init 6.88 (ln 999 = 6.91); valid true/null/random slots hit 0.94/0/0.01.
- Runs (Toys, v6, K 3, micro 2, β 0): B6 (fill pred), B6sim (fill similar). Compare with b-micro2 365893 (0.0430/0.0879/0.1129 | 0.0430/0.0662/0.0743).
- Submitted 2026-11-16: B6 366598, B6sim 366599 (hady a40, preemptible; scratch 422 GB).
- 2026-11-16 user: curriculum reader C on v5 and v6. Added `--insert_p_start` (per-epoch linear insert prob., 1.0 → 0.5 → 0.0) and `--save_last` (decodes use the epoch-3 model trained at the deployment ratio). CPU check, v5: true-in-slots 0.995 / 0.72 / 0.38 at p 1 / 0.5 / 0. Runs C5 (v5, motive_corpus/stage2) and C6 (v6 P1000): micro 2, K 3, β 0, motive head 0.1, fill pred. Reference: GRPO session C (v5) 365710.
- Need/kind two-stage pilot (`collab/need_kind_analysis.py`, mpnet + k-means on need phrases): flat cls R@3 0.225; two-stage (top-1 need group, retriever inside, prototype-need groups) 0.162 (G 100) / 0.155 (G 300); need group predictability only 2.0× popularity (full motive 13×); oracle need group → R@3 0.550. Candidate sets used the prototype's need text (matches the row's need group on 43% of rows) — mining-based candidate lists not yet tried.
- Submitted 2026-11-16: C5 366678, C6 366679 (pending on researchshort).

## 2026-11-17 — B6 / B6sim results (v6 reader B)
- B6 366598 test: 0.0386/0.0824/0.1118 | 0.0386/0.0609/0.0705; B6sim 366599: 0.0392/0.0840/0.1121 | 0.0392/0.0621/0.0712.
  Both < b-micro2 (0.0430/0.0879/0.1129 | 0.0430/0.0662/0.0743). Epoch-3 model; valid best at epoch 2.
- Epoch-3 predictions backed up to result/standard_epoch3_trainrun before the epoch-2 test-top re-decode.
- Requested 10 decodes via JobManager (epoch-2 ckpt): test top + valid top/true/null/random for each run.
- C5 366678 started on album (hady), epoch 1; C6 366679 pending.

## 2026-11-17 — motive v6 for CDs_and_Vinyl (user request)
- No GEARec reasoning exists for CDs (GEARec_R3_toys is Toys only). Ported the GEARec v2 generator into the worktree:
  `collab/gearec_v2/target_reasoning.py` (copy; GEARec's file untouched) + `--domain cds` (music wording: genres, eras,
  artist styles, formats; never name title/artist/album/label; music stop words for the leak check) + stages
  `items` (items.json from split CSVs + R3/data/meta_CDs_and_Vinyl.json.gz; 5,841 items, all with metadata) and
  `item_emb` (mxbai over item documents, as GEARec).
- `collab/gearec_v2/submit_gearec_cds.sbatch`: gemma-3-4b-it Best-of-8, full history (CDs max 10), T 0.8, seed 42,
  generate + validate (rank by mxbai; no judge filter, as Toys v6). Jobs: valid 367720, test 367721, train 367722.
- `collab/build_motive_corpus_v6.py`: `domain(args)` picks CDs reasoning dir (data/CDs_and_Vinyl/collab/gearec_v2/
  full_<split>_bo8) and music versions of the extract/name prompts; Toys path unchanged.
- `collab/build_motive_text_support.py`: stage2 text supports from a corpus + retriever scores (Toys v6 format).
- Remaining: corpus (submit_motive_corpus_v6.sbatch DATASET=CDs_and_Vinyl) -> retriever
  (train_motive_retriever.py --dataset CDs_and_Vinyl --corpus motive_corpus_v6/P1000 --head cls --crossfit-folds 5)
  -> build_motive_text_support.py -> MotiveRec CDs run.

## 2026-11-18 — C5/C6 valid diagnostics
C5 (v5 curriculum) true/top/null/random HR@10 0.1272/0.1266/0.1208/0.1268 → ignores motive content (format effect only).
C6 (v6 curriculum) true/top/null/random HR@10 0.1383/0.1185/0.1122/0.0923 → reads content; retrieved motives +0.006 over null, true +0.026.
Curriculum does not by itself make a motive-reader that beats a motive-ignorer; retrieval accuracy is the bottleneck. Details: Report/experiments/07_ntp_only E9d.

## 2026-11-18 — CDs v6 motive corpus built (job 367973)
Extract rows train/valid/test 44775/5851/5921. P1000 → 1005 prototypes (rows/proto min/median/max 9/84/315, largest share 0.35%);
P2000 → 1767. Rows with 0 motives: train 9.1%, valid 5.0%, test 3.8%. Naming errors 0.
Next: retriever (cls, crossfit 5) + supports via collab/submit_motive_retriever_v6.sbatch.
CDs v6 retriever (job 368132, cls, crossfit 5), test recall@1/5/10/20, mrr: mlp 0.0499/0.1497/0.2216/0.3113, 0.1872;
popularity 0.0046/0.0139/0.0263/0.0469, 0.0296 (Toys v6 mlp: 0.0574/0.1647/0.2368/0.3338, 0.2058). Supports written to P1000/stage2.
Submitted CDs v6 plain MotiveRec (K3, NTP only, micro 2, RUN_TAG=motivev6-micro2). Compare: CDs v5 predicted K3 micro2 (job364978)
0.0614/0.0979/0.1169 | 0.0614/0.0802/0.0862; IntuRec 0.0580/0.0981/0.1202 | 0.0580/0.0789/0.0860; O0 0.0604/0.0968/0.1178 | 0.0604/0.0796/0.0864.

## 2026-11-18 — checkpoint cleanup
Removed model.safetensors of 12 superseded MotiveRec runs (user OK). Their metrics: Report/deleted_checkpoints_2026-11-18.md. Logs kept on scratch.

## 2026-11-18 — CDs v6 plain MotiveRec result (job 368133; K3, NTP only, micro 2)
Test (end-of-train, epoch-3 in-memory model): 0.0609/0.0945/0.1158 | 0.0609/0.0786/0.0854. Valid NTP 1.1605/1.0529/1.1918 (saved ckpt = epoch 2).
vs CDs v5 same setup (364978) 0.0614/0.0979/0.1169 | 0.0614/0.0802/0.0862; IntuRec 0.0580/0.0981/0.1202 | 0.0580/0.0789/0.0860; O0 0.0604/0.0968/0.1178 | 0.0604/0.0796/0.0864.
→ v6 ≈ v5 on CDs too (slightly lower), below IntuRec at @5/@10, above at @1.

## 2026-11-18 — user: CDs curriculum + b-micro2
CDs b-micro2 (plain v6 K3 micro 2) = 368133 (done, 0.1158 HR@10). Submitted CDs C6 (v6) and C5 (v5): insert 1.0→0.5→0, head 0.1, save last, K3, micro 2, β 0 (Toys C5/C6 recipe).

## 2026-11-18 — user: b-micro2 + curriculum only (Toys)
Same as b-micro2 365893 (v6 P1000, K3, micro 2, β 0, NTP only, inject, save best) + train-only true-motive insertion 1.0→0.5→0 (fill pred). No motive head. Differs from C6 only by head 0 and save best (C6: head 0.1, save last).

## 2026-11-18 — user: reproduce CollabRec 356575 in the MotiveRec code
Ported from collab-refined-infonce (that worktree untouched): `--readout gated` (+ `--sel_weight`; model.gated_readout, selection BCE vs "slot is a true motive"; valid prints gate true/wrong means + AUC) and `--noisy_precision` / `--noisy_precision_start` (dataset: train/valid slots keep each true motive with prob p else top non-true prediction, shuffled; train re-drawn, valid fixed per row; test = retriever top-k). Sbatch: READOUT, SEL_WEIGHT, NOISY_P, NOISY_P_START (defaults off → existing runs unchanged).
CPU checks (Toys v5, K2): train slot precision 1.000 / 0.623 / 0.256 at p 1 / 0.625 / 0.25; valid fixed 0.259; test top-2 0.151; tiny bf16 model: gated loss = NTP + BCE, rel_q gets grads, inference OK.
Run: Toys, v5 motive_corpus, K2, micro 2, β 0, gated 1.0, noisy 1.0→0.25, RUN_TAG=micro2-repro356575. Reference 356575: retest (epoch-2 ckpt, cls top-10) 0.0421/0.0852/0.1126 | 0.0421/0.0643/0.0732; rerun 357942 top-2 HR@10 0.1121, NDCG@10 0.0740.
Note: this objective adds the selection BCE (not NTP-only) — user-requested reproduction.

## 2026-11-19 — CDs C6 result (job 368590; v6 K3 micro 2, insert 1.0→0.5→0, head 0.1, save last)
Test (epoch-3 model = saved last): 0.0578/0.0914/0.1127 | 0.0578/0.0755/0.0824.
Valid total/NTP/motive/head-top1: ep1 1.7425/1.1660/5.7654/0.0999; ep2 1.6006/1.0467/5.5392/0.1231; ep3 1.7197/1.1541/5.6559/0.1217.
→ below CDs plain v6 368133 (0.1158), v5 364978 (0.1169), IntuRec (0.1202). Curriculum + head hurts on CDs, as on Toys (C6 0.1109 < b-micro2 0.1129).

## 2026-11-19 — CDs C5 result (job 368591; v5 K3 micro 2, insert 1.0→0.5→0, head 0.1, save last)
Test (epoch-3 model): 0.0546/0.0869/0.1070 | 0.0546/0.0714/0.0779.
Valid total/NTP/motive/head-top1: ep1 1.7229/1.1730/5.4995/0.1538; ep2 1.5891/1.0620/5.2715/0.1563; ep3 1.6911/1.1655/5.2558/0.1540.
→ worst CDs run; v5 plain same setup (364978) 0.1169. On CDs the C5/C6 recipe (insertion curriculum + head + save last) hurts both corpora (−0.010 v5, −0.003 v6 HR@10).

## 2026-11-19 — Toys b-micro2 + insertion curriculum only (job 368636; v6 K3 micro 2, insert 1.0→0.5→0, no head, save best)
Test (end-of-train epoch-3 in-memory model; run_collabrec.py does not reload the saved best before evaluate): 0.0430/0.0841/0.1109 | 0.0430/0.0643/0.0728.
Valid NTP 0.6709/0.5854/0.6108 (saved ckpt = epoch 2, not the one tested).
vs b-micro2 365893 0.0430/0.0879/0.1129 | 0.0430/0.0662/0.0743; C6 0.1109. → the insertion curriculum alone gives the C6 drop; the head is neutral.
Possible follow-up: decode the saved epoch-2 ckpt (EVAL_CKPT unset, TRAIN_OR_TEST=test).

## 2026-11-19 — 356575 reproduction in MotiveRec code (job 368876; Toys v5 K2 micro 2, gated 1.0, noisy 1.0→0.25)
Test (end-of-train epoch-3 model, retriever top-2): 0.0421/0.0876/0.1125 | 0.0421/0.0656/0.0736.
Valid total/NTP/selection: ep1 3.4353/0.6739/2.7615; ep2 1.1953/0.5850/0.6103; ep3 1.0193/0.6073/0.4120.
Gate AUC 0.7780/0.8111/0.8157 (true 0.985/0.738/0.561, wrong 0.964/0.419/0.170).
vs 356575 retest 0.0421/0.0852/0.1126 | 0.0421/0.0643/0.0732, AUC 0.775/0.813/0.813; 357942 HR@10 0.1121, NDCG@10 0.0740.
→ reproduced (HR@10 within 0.0004, same AUC trajectory). Port is consistent with CollabRec.

## 2026-11-19 — user: does the motive head alone (no curriculum) give motive usage?
Run "head only": Toys v6 P1000, K3 retrieved motives, micro 2, β 0, motive head 0.1, save last, no insertion (= C6 minus curriculum, = b-micro2 365893 + head).
Then 4 valid decodes (true/top/null/random), compared with C6 (true/top/null/random HR@10 0.1383/0.1185/0.1122/0.0923).
Submitted: job 369390 (pending, researchshort, l40|a40|l40s).
Result 369390 (head only, no curriculum), test (epoch-3 model): 0.0401/0.0831/0.1088 | 0.0401/0.0623/0.0707.
Valid NTP/motive/head-top1: ep1 0.6751/5.4828/0.1337; ep2 0.5893/5.4041/0.1431; ep3 0.6166/5.6766/0.1275.
vs b-micro2 365893 (no head) 0.1129, C6 (head + curriculum) 0.1109, 368636 (curriculum only) 0.1109. → the head alone costs −0.004 HR@10.
Valid decodes submitted: top 369712, true 369713, null 369714, random 369715 (hady A40, preemptible).
369390 valid decodes: top (369712) 0.0552/0.0953/0.1219 | 0.0552/0.0758/0.0843 (C6 top 0.1185).
369390 valid decodes: true (369713) 0.0551/0.0951/0.1204 | 0.0551/0.0755/0.0836 (≤ top → no gain from true motives; C6 true 0.1383).
369390 valid decodes: null (369714) 0.0549/0.0951/0.1210 | 0.0549/0.0755/0.0838 → top/true/null all ≈0.121: ignores motive content.
369390 valid decodes: random (369715) 0.0554/0.0962/0.1220 | 0.0554/0.0761/0.0843.
→ Conclusion: head only (no curriculum) ignores motive content: true/top/null/random HR@10 0.1204/0.1219/0.1210/0.1220 (spread 0.0016),
like C5. C6 (curriculum + head) 0.1383/0.1185/0.1122/0.0923. Motive usage comes from the insertion curriculum, not the head.

## 2026-11-20 — user: 356575 recipe with v6 motives
Same as 368876 (K2, gated 1.0, noisy 1.0→0.25, micro 2, β 0, save best) but MOTIVE_CORPUS=motive_corpus_v6/P1000. RUN_TAG=motivev6-micro2-repro356575.
Compare: 368876 (v5) 0.0421/0.0876/0.1125 | 0.0421/0.0656/0.0736; b-micro2 365893 0.1129; 368636 0.1109.
Submitted: job 371436.
371436 failed at launch (my submit line lacked CODE_DIR → main-repo run_collabrec.py rejected the new flags). JobManager resubmitted with CODE_DIR=motiverec worktree as 371443.

## 2026-11-20 — user: MotiveRec on Steam and Movielens (plain v6 K3; full v6 pipeline; no IntuRec baseline here)
Data: R3 splits (Steam_5_2016-1-2017-12: 82419/10302/10303 rows, 4419 items; Movielens_5_2014-1-2015-3: 81308/10163/10164, 2916 items;
history ≤ 10). IntuRec/data/Steam already existed (= R3 copy); IntuRec/data/Movielens copied from R3 (R3 untouched); worktree symlinks.
Code (worktree only):
- collab/gearec_v2/target_reasoning.py: --domain steam|movielens (game / movie wording in system prompt, header, section labels,
  task; stop words); --stage items reads R3 item_meta.json (genres → categories, Steam developer → brand, tags kept);
  item_block drops an empty brand and prints tags. Toys/CDs prompts unchanged. Backup in scratchpad.
- collab/build_motive_corpus_v6.py: extract/name prompts for Steam and Movielens; reasoning root data/<ds>/collab/gearec_v2.
- data/CollabRecDataset.py CATEGORY: Steam → games, Movielens → movie.
- New collab/gearec_v2/submit_gearec.sbatch (DATASET, DOMAIN, SPLIT) and collab/submit_sasrec.sbatch (Verifiable-Rec sasrec.py
  read-only, run_sasrec_toys.sh settings, output data/<ds>/collab/sasrec/). Retriever sbatch: SASREC_CKPT, auto-picks that pkl.
items.json built on CPU (Steam 4419, Movielens 2916 items, 0 without metadata). SASRec 1-epoch CPU smoke on Steam OK.
Chain per dataset: GEARec train/valid/test → corpus v6 → retriever (+ SASRec) → MotiveRec (365893 recipe).
Open check: submit_motive_retriever_v6.sbatch never passed --sasrec-ckpt, so its default (the CDs SASRec) applies unless the
Toys v6 retriever was run another way; to verify for Toys.
Steam chain: GEARec 371957/371958/371959, SASRec 371960, corpus 371961, retriever 371962, MotiveRec 371963. Movielens: waits for queue slots (cap 16).
Toys SASRec check: session transcripts show Toys retriever commands with --sasrec-ckpt .../Toys_and_Games_5_2017-1-2018-11/models/SASRec/2026-10-31_12-00-01-480407.pkl → Toys retriever used the Toys SASRec (concern resolved).

## 2026-11-20 — 356575 recipe with v6 result (job 371443; K2, gated, noisy 1.0→0.25)
Test (epoch-3 model): 0.0442/0.0868/0.1101 | 0.0442/0.0665/0.0740. Valid NTP 0.6785/0.5889/0.6115; selection 3.139/0.763/0.520.
Gate AUC 0.627/0.656/0.649 (v5 368876: 0.778/0.811/0.816). vs v5 368876 0.0421/0.0876/0.1125 | 0.0421/0.0656/0.0736; v6 plain K3 365893 0.1129.
→ v6 gate cannot separate true from wrong motives (AUC ~0.65); best HR@1 / NDCG@5 of all Toys runs but HR@10 −0.0024 vs v5.
Movielens chain: GEARec 372421/372422/372423, SASRec 372424, corpus 372425, retriever 372426; MotiveRec step waits for a free slot (16/16).
MotiveRec Movielens = 372745 (afterok:372426). Steam GEARec valid 371958: 10302 rows, with best 9964, median rank 225, top20 0.198, top100 0.369 (CDs valid: median 205, top20 0.241, top100 0.401).
Steam GEARec test 371959: 10303 rows, with best 10070, median rank 214, top20 0.203, top100 0.378. SASRec Steam 371960: test R@10 0.0214 NDCG@10 0.0118 (valid 0.0497/0.0347; time shift, test = Nov–Dec 2017).
Movielens GEARec valid 372422: 10163 rows, with best 10156, median rank 22, top20 0.489, top100 0.700.
Movielens GEARec test 372423: 10164 rows, with best 10160, median rank 21, top20 0.498, top100 0.707.
Leak check (valid): 30% of best reasonings name the target movie (titles ≥ 8 chars; Steam 7%, CDs 6%); leaky best median rank 2 vs 59.
Asked the user whether to re-validate Movielens with --filter_leak before the corpus step.
SASRec Movielens 372424: test R@1/5/10 0.0129/0.0578/0.0988, NDCG@1/5/10 0.0129/0.0352/0.0483, MRR 0.0434 (valid R@10 0.1058, NDCG@10 0.0533).
Steam GEARec train 371957 (12h34m): 82419 rows, with best 71671 (87%; valid/test 97–98%), median rank 235, top20 0.194, top100 0.364. Corpus 371961 released.
Fix (2026-11-21): target_reasoning.py validate now drops out-of-range support citations instead of rejecting the candidate (Steam 1-game histories cited "1, 2" -> 10,569 train rows had no best). Offline estimate: Steam train coverage 87% -> ~100%. Old Steam validated.jsonl backed up as validated.pre_supfix.jsonl. Re-validate of Steam splits + hold of corpus 371961 NOT yet submitted (blocked pending user approval). Movielens train 372421 will validate with the patched code.
Steam re-validate submitted: train 373316, valid 373317, test 373318; corpus 371961 now afterok on them (old validated files kept as validated.pre_supfix.jsonl).
Steam train rerun 373316: 82419 rows, with best 82404 (was 71671), median rank 233, top20 0.199, top100 0.367.
@3 added (recomputed from result/standard/meta.json, same matching as calRecallAndNdcgFromFile; @1/5/10 reproduce exactly). HR@3 | NDCG@3:
Toys: 365893 0.0723|0.0599, 368636 0.0684|0.0578, 366679 0.0702|0.0586, 369390 0.0669|0.0556, 368876 0.0706|0.0586, 371443 0.0729|0.0608, IntuRec 348597 0.0663|0.0562.
CDs: 368133 0.0835|0.0741, 364978 0.0838|0.0744, 368590 0.0807|0.0711, 368591 0.0749|0.0665, IntuRec 349506 0.0830|0.0727.
Latent-dependence diagnostics (Toys valid, HR@10 true/top/null/random): 366679 C6 0.1383/0.1185/0.1122/0.0923 (uses); 366598 B6 ins0.8+mh 0.1510/0.1170/0.1115/0.0928 (uses); 366599 B6sim 0.1511/0.1208/0.0810/0.0818 (uses, collapses without motives); 366678 C5 v5 0.1272/0.1266/0.1208/0.1268 (ignores); 369390 mh only 0.1204/0.1219/0.1210/0.1220 (ignores). No diagnostics for 365893, 368636, 368876, 371443 or any CDs run. Test of B6 366598: 0.0399/0.0675/0.0840/0.1067 | 0.0399/0.0561/0.0629/0.0704; B6sim 366599: 0.0398/0.0665/0.0841/0.1072 | 0.0398/0.0549/0.0622/0.0696; C5 366678: 0.0430/0.0692/0.0849/0.1109 | 0.0430/0.0583/0.0647/0.0731.
Steam reruns: valid 373317 10302 rows, with best 10299, median 222, top20 0.201, top100 0.373; test 373318 10303/10299, median 213, top20 0.204, top100 0.379.
Movielens GEARec train 372421 (12h36m, patched check): 81308 rows, with best 81308, median rank 25, top20 0.477, top100 0.683. No leak-filter decision from the user -> corpus 372425 runs unfiltered. Both corpus jobs (371961, 372425) pending on Priority at 17:12.
Movielens corpus done (372425; hady twin 373468 also ran, final files all from 372425 after 21:05): P1000 -> 999 prototypes, rows/proto min/median/max 22/149/1178, largest share 0.0073, rows with 0 motives valid 0.0007 / test 0.0004.
Movielens retriever 372426 (and hady twin 373469) FAILED: (1) missing hyp_cache/hist_gte-large-en-v1.5*.npy (never built for new datasets); (2) SASRec 0/2916 items mapped (int asins vs str iid_map keys). Fixes: new collab/build_hist_text_emb.py (gte-large, "; "-joined history titles, as tune_latent.py), called first in submit_motive_retriever_v6.sbatch; build_support.SasrecEncoder falls back to str(asin) (now 2912/2916 Movielens, 4386/4419 Steam; Toys/CDs unaffected). Queued Steam retrievers 371962/373466 use the old sbatch copy -> need replacing. Awaiting user approval to resubmit.
Steam corpus done (hady 373465, 3h57m): 1003 prototypes, rows/proto min/median/max 22/152/734, largest share 0.0045, rows with 0 motives train/valid/test 0.0002/0.0003/0.0004. Old-sbatch retriever 373466 failed as expected. User approved resubmitting both retrievers with the fix + repointing MotiveRec.
Resubmitted retrievers: Steam 374112 (+hady twin 374113), Movielens 374114 (+twin 374115); MotiveRec Steam 371963/373467 -> afterok:374112, Movielens 372745/373470 -> afterok:374114.
Retrievers 374113/374115 FAILED: gte-large hub main moved to model_type 'gte' (unsupported by rrec transformers). build_hist_text_emb.py now pins revision 104333d6 (the snapshot used for Toys/CDs; reproduces CDs cache with cos 1.0000).
Retrievers resubmitted (gte pinned): Steam 374119/374120, Movielens 374121/374122; MotiveRec repointed.
Steam retriever 374120 done: SASRec 4386/4419 mapped; test recall@1/5/10/20 mlp 0.0383/0.1223/0.1817/0.2669 MRR 0.1625 (popularity 0.0043/0.0177/0.0322/0.0585, MRR 0.0330); valid mlp R@10 0.1985 MRR 0.1723.
Movielens retriever 374122 done: SASRec 2912/2916; test recall@1/5/10/20 mlp 0.0352/0.1053/0.1611/0.2356 MRR 0.1458 (popularity 0.0071/0.0256/0.0426/0.0689 MRR 0.0425; item_profile 0.0318/0.0992/0.1505/0.2143 MRR 0.1362).
MotiveRec 373467 (Steam) / 373470 (Movielens) FAILED in 17 s: no rationales_facet_train_v5.jsonl (IntuRec facet rationales; don't exist for new datasets). They are only L_collab targets (own/neg tokens, used only when beta > 0). Fix: CollabRecDataset text mode makes them optional when support_texts.json exists (Toys/CDs unchanged). CPU smoke test of valid/test datasets OK for both datasets. Awaiting user approval to resubmit.
MotiveRec plain v6 K3 resubmitted (optional-rationale fix): Steam 374160 (+hady 374161), Movielens 374162 (+hady 374163). Env adds MOTIVE_CORPUS=motive_corpus_v6/P1000 (only loads motive_target; unused without head/insertion, so same recipe as Toys 365893).
User: also run v6 + curriculum + motive head (C6 recipe of 366679/368590: INSERT_P_START=1.0 -> 0, MOTIVE_HEAD_W=0.1, SAVE_LAST=1) on Steam and Movielens; requested via JobManager.
C6 Steam 374229 (+hady 374230), Movielens 374231 (+hady 374232) submitted.
2026-11-22 ~04:47-04:51: scratch hit 500 GB quota (mostly other sessions' IntuRec F1 Steam/Movielens checkpoints with optimizer states) -> FAILED: Steam plain 374161 (after epoch 1, best valid NTP 1.2687 saved), Movielens plain 374163 (epoch 1 step ~470), Steam C6 374229 (epoch 1 step ~90). User cleaned scratch (now ~418 GB). Movielens C6 374232 running. Reruns await user approval.
User approved rerun of 374161, 374163, 374229 (same envs); requested via JobManager.
Reruns: Steam plain 374810/374811, Movielens plain 374812/374813, Steam C6 374814/374815 (twins). Scratch 319 GB.
All 4 Steam/Movielens MotiveRec runs running: Steam plain 374810, Movielens plain 374812, Steam C6 374814, Movielens C6 374232.

- Sun Nov 22 15:59 : Movielens plain 374812 and Steam C6 374814 preempted on lagoon at 15:05 during epoch 1 (no checkpoint yet), auto-requeued in researchshort. Est. start 374812 22:59 Nov 22, 374814 09:25 Nov 23. 374810 and 374232 still running.
- Sun Nov 22 23:06 : 374812 (Movielens plain) requeue started on avenue (from scratch; no epoch-1 checkpoint). 374814 still pending.
- Sun Nov 22 23:08 : 374814 (Steam C6) requeue started on avenue (from scratch). All 4 running.
- Mon Nov 23 04:09 : RESULT Movielens C6 374232 (exit 0, 19h11m, epoch-3 model) test HR@1/3/5/10 0.0271/0.0595/0.0857/0.1348 | NDCG 0.0271/0.0456/0.0563/0.0721 (n=10164). SASRec: HR 0.0129/-/0.0578/0.0988, NDCG 0.0129/-/0.0352/0.0483. Motive head valid top-1 0.0734 at ep3.
- Mon Nov 23 05:42 : RESULT Steam plain 374810 (exit 0, 18h47m, epoch-3 model) test HR@1/5/10 0.0119/0.0288/0.0429 NDCG 0.0119/0.0206/0.0252 (@3 in chat via metrics_at3). SASRec HR@10 0.0214 NDCG@10 0.0118.
- Mon Nov 23 10:02 : CDs epoch-2 test-only decodes submitted (best-valid saved model = epoch 2): plain v6 368133 -> 376945/376946, v5 364978 -> 376947/376948, IntuRec 349506 -> 376949/376950 (new collab/test_inturec_ckpt.sbatch, runs unmodified run_inturec_f2.py in test mode). Outputs result/standard/epoch_2/. Note: curriculum runs (SAVE_LAST=1) lost epoch 2; 368590 ep2 valid NTP 1.0467 was best of all CDs models (IntuRec ep2 1.0507).
- Mon Nov 23 10:06 : CDs curriculum tuning (user approved A–E). Code: run_collabrec.py --test_best (after last-epoch test, reload best-valid ckpt -> result/standard/best); submit_collabrec.sbatch LR (default 0.0003, in run name) + TEST_BEST. Backups in scratchpad (*.pre_testbest.py, *.pre_lr.sbatch). 7 runs, all v6 K3, curriculum start 1.0, SAVE_LAST=0, TEST_BEST=1, RUN_TAG=motivev6-micro2-best: A (->0, mh0.1), B (->0.25), C (fill similar), D03 (mh0.03), D0 (mh0), E (lr 2e-4), ABC (->0.25 + similar). Submitted via JobManager.
- Mon Nov 23 10:06 : CDs tuning job IDs (researchshort / hady twin): A 376951/376952, B 376953/376954, C 376955/376956, D03 376957/376958, D0 376959/376960, E 376961/376962, ABC 376963/376964. Scratch 370G.
- Mon Nov 23 10:36 : Tuning A started as hady twin 376952 (album); folder ..._job376952, settings verified.
- Mon Nov 23 10:50 : Tuning B started as 376953 (lunar), settings verified.
- Mon Nov 23 10:58 : Tuning C started as 376955 (lucas), settings verified.
- Mon Nov 23 11:08 : Tuning D03 started as 376957 (albert), settings verified.
- Mon Nov 23 11:08 : RESULT CDs v5 364978 epoch-2 test (376947): HR@1/3/5/10 0.0572/0.0760/0.0864/0.1085 | NDCG 0.0572/0.0683/0.0725/0.0797 — worse than its epoch-3 (0.0614/0.0838/0.0979/0.1169 | 0.0614/0.0744/0.0802/0.0862) despite lower valid NTP (1.0522 vs 1.1625). Caveat: 364978 trained with older code; decode used current code.
- Mon Nov 23 11:09 : Tuning D0 started as 376959 (avenue), settings verified.
- Mon Nov 23 11:17 : RESULT CDs IntuRec 349506 epoch-2 test (376949): HR@1/3/5/10 0.0594/0.0810/0.0955/0.1181 | NDCG 0.0594/0.0718/0.0777/0.0851 (epoch-3: 0.0580/0.0830/0.0981/0.1202 | 0.0580/0.0727/0.0789/0.0860). Epoch 2 only better at @1 for IntuRec too.
- Mon Nov 23 11:28 : RESULT CDs plain v6 368133 epoch-2 test (376945): HR@1/3/5/10 0.0572/0.0771/0.0897/0.1134 | NDCG 0.0572/0.0689/0.0740/0.0816 (epoch-3 0.0609/0.0835/0.0945/0.1158 | 0.0609/0.0741/0.0786/0.0854). Conclusion: on CDs epoch 3 beats epoch 2 on test for all three models despite higher valid NTP; best-valid selection (tuning A) unlikely to help.
- Mon Nov 23 16:34 : Tuning E started as 376961 (analog), LR=0.0002 verified.
- Mon Nov 23 16:55 : RESULT CDs tuning C 376955 (5h57m, lucas): last(ep3) HR@1/3/5/10 0.0580/0.0789/0.0911/0.1147 | NDCG 0.0580/0.0703/0.0753/0.0829; best(ep2) 0.0515/0.0755/0.0854/0.1009 | 0.0515/0.0654/0.0694/0.0744. Valid NTP 1.1679/1.0468/1.1532. Slightly above 368590 C6 (HR@10 0.1127) but below plain v6/v5/IntuRec.
- Mon Nov 23 17:11 : Tuning ABC started as 376963 (avenue), settings verified. All 7 tuning runs started (C done).
- Mon Nov 23 17:12 : RESULT Steam C6 374814 (18h04m avenue, ep3): see chat; HR@1/5/10 0.0123/0.0293/0.0425 NDCG 0.0123/0.0208/0.0251 (plain 374810: 0.0119/0.0288/0.0429 | 0.0119/0.0206/0.0252).
- Mon Nov 23 18:01 : RESULT Movielens plain v6 374812 (18h55m avenue, ep3): HR@1/5/10 0.0287/0.0900/0.1397 NDCG 0.0287/0.0594/0.0755 (C6 374232: 0.0271/0.0857/0.1348 | 0.0271/0.0563/0.0721). All 4 Steam/Movielens runs done.
- Mon Nov 23 18:24 : RESULT CDs tuning B 376953 (7h33m lunar): last HR@1/5/10 0.0570/0.0949/0.1184 NDCG 0.0570/0.0763/0.0839; best(ep2) 0.0529/0.0849/0.1061 | 0.0529/0.0700/0.0768.
- Mon Nov 23 20:05 : RESULT CDs tuning D03 376957 (8h57m albert): last HR@1/5/10 0.0591/0.0918/0.1166 NDCG 0.0591/0.0757/0.0838; best(ep2) 0.0541/0.0897/0.1075 | 0.0541/0.0725/0.0783.
- Mon Nov 23 22:53 : RESULT CDs tuning A 376952 (12h17m album): last HR@1/5/10 0.0578/0.0958/0.1171 NDCG 0.0578/0.0772/0.0840; best(ep2) 0.0539/0.0888/0.1059 | 0.0539/0.0719/0.0774. A's last-epoch = same recipe as 368590 (HR@10 0.1127) -> run-to-run spread ~0.004 HR@10; A/B/C/D03 within noise.
- Mon Nov 23 23:17 : RESULT CDs tuning D0 376959 (12h08m avenue): last HR@1/5/10 0.0580/0.0953/0.1153 NDCG 0.0580/0.0774/0.0838; best(ep2) 0.0560/0.0906/0.1119 | 0.0560/0.0741/0.0809. Motive head weight (0/0.03/0.1) has no measurable test effect.
- Tue Nov 24 03:39 : RESULT CDs tuning E 376961 lr2e-4 (11h04m analog): last HR@1/5/10 0.0572/0.0872/0.1114 NDCG 0.0572/0.0728/0.0806; best(ep2) 0.0513/0.0828/0.0984 | 0.0513/0.0674/0.0724. Lower LR hurts.
- Tue Nov 24 05:27 : RESULT CDs tuning ABC 376963 (12h15m avenue): last HR@1/5/10 0.0581/0.0923/0.1174 NDCG 0.0581/0.0758/0.0839; best(ep2) 0.0503/0.0848/0.1030 | 0.0503/0.0680/0.0738. SUMMARY: 6/7 curriculum variants within noise (HR@10 0.1147-0.1184, NDCG@10 0.0829-0.0840), all below IntuRec ep3 (0.1202/0.0860); lr 2e-4 worse; best-valid ckpt always worse on test.
