# Research log

Started fresh on 2026-10-23. The previous log (2026-10-21 → 10-23: rationale generation, hypothesis test,
retrieval, motive corpus, CollabRec v1 runs, oracle design) is archived in
`collab/archive/RESEARCH_LOG_2026-10-21_to_10-23.md`.

---

## Setup

- **Data:** Amazon CDs_and_Vinyl, 5-core, 2015-10 → 2018-11, time split — train 49,251 rows (6,865 users), valid 6,156, test 6,157; 5,841 items.
- **Models:** IntuRec (stage 1 `run_inturec_f1.py`, stage 2 `run_inturec_f2.py`), CollabRec (`run_collabrec.py`, `model/CollabRecF2.py`, `data/CollabRecDataset.py`); base LLM Qwen2.5-1.5B, full fine-tune bf16.
- **Env:** `~/.conda/envs/rrec` (Qwen3-family models need `~/.conda/envs/racer`).
- **Run protocol (all runs):** lr 3e-4, 20 warmup steps + linear decay, 3 epochs, effective batch 128 with **micro-batch 8**, GPUs ≥ 40 GB only (L40 / A100 / A40 / L40S), constrained top-10 beam decoding of the test set with the last-epoch model.
- **Checkpoints:** only the best by average valid NTP is saved (CollabRec and IntuRec stage 2).
- **Storage:** one folder per run under `/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints/<dataset>/<experiment>/<run>/` (checkpoint, `result/<decode>/meta.json`, `train.log`, `slurm-<job>.out`); the repo's `checkpoints/` is a symlink to it.
- **Submit:** `submit_inturec_f2.sbatch` (experiment `IntuRec_original`), `submit_collabrec.sbatch` (experiments `CollabRec`, `CollabRec_oracle`); both use `--requeue`.

---

## Ideas

## Experiments and results

### 2026-10-23 — submitted
| Job | Experiment / run | Purpose |
|---|---|---|
| 335690 | `IntuRec_original/IntuRecF2_micro8_job335690` | IntuRec stage-2 baseline under the fixed protocol |
| 335691 | `CollabRec/text-item-gamma1.0_title_k10_beta0.1_rs1_lr0.0003_job335691` | item-grounded peer supports + contrastive support loss, anchor = target title |
| 335692 | `CollabRec/text-item-gamma1.0_motive_k10_beta0.1_rs1_lr0.0003_job335692` | same, anchor = ground-truth motive prototypes |
| 335693 | `CollabRec/emb_k10_beta0.1_rs1_lr0.0003_job335693` | original CollabRec, emb integration (mpnet support vectors + learned projection), peer-rationale supports |
| 335694 | `CollabRec/text_k10_beta0.1_rs1_lr0.0003_job335694` | original CollabRec, text integration (support rationale tokens → InjectNet), peer-rationale supports |
| 335725 | `CollabRec_oracle/text-oracle_own_k1_beta0_rs1_lr0.0003_job335725` | **O1 oracle**: row's own rationale text as the support (β 0) — can rationale input help at all? |
| 335726 | `CollabRec_oracle/text-oracle_own_motive_k2_beta0_rs1_lr0.0003_job335726` | **O2 oracle**: row's own 1–2 motive prototype texts as the support (β 0) |

335691 / 335692 (support loss, title / motive anchor) cancelled after ~70 min: the support loss stayed at ln 2 (0.6931 → 0.6922 by step 200) while IntuRec's intuition loss fell to 0.39 — the retrieved-support thought is indistinguishable from a random-support thought. Suspected causes: last-token pooling of long `"title": rationale` texts hides the title; retrieved and random rationales are near-identical in register; raw-dot BPR has tiny margins. Candidate fix (not run): pool title and rationale separately + cosine/τ InfoNCE with channel-matched anchors.

### 2026-10-23 — results

| Run (job) | Support input | R@1 | R@5 | R@10 | N@5 | N@10 | valid NTP ep1/2/3 |
|---|---|---|---|---|---|---|---|
| O0 (archived, 335415) | none | 5.78 | 9.13 | 11.00 | 7.49 | 8.09 | 1.093 / 0.984 / 1.088 |
| **O1 oracle (335725)** | own rationale text (k 1) | 5.72 | 8.90 | 11.08 | 7.38 | 8.09 | 1.089 / 0.988 / 1.099 |
| **O2 oracle (335726)** | own 1–2 motive prototypes (k 2) | **5.99** | **9.89** | **12.29** | **8.02** | **8.79** | 1.086 / 0.963 / 1.071 |
| IntuRec F2 baseline (335690) | 15 intuition titles | 5.83 | 9.26 | 11.19 | 7.61 | 8.24 | 1.095 / 0.986 / 1.109 |
| CollabRec text (335694) | 10 peer rationales, text → InjectNet, β 0.1 | 5.67 | 9.10 | 11.35 | 7.48 | 8.21 | 1.085 / 0.985 / 1.104 |
| CollabRec emb (335693) | 10 peer rationales, mpnet + projection, β 0.1 | 5.52 | 9.11 | 11.42 | 7.37 | 8.11 | 1.098 / 0.986 / 1.097 |
| ref: IntuRec mean of earlier 3 runs | | 5.81 | 9.46 | 11.73 | 7.69 | 8.42 | |
| ref: SASRec | | 5.33 | 9.85 | 12.19 | 7.73 | 8.48 | |

Findings:
- **Motive information helps:** O2 − O0 = +1.29 R@10 / +0.70 N@10 (beyond the ±0.25–0.5 seed spread); best LLM run so far, ≥ SASRec. Prototypes carry no names → not target leakage.
- **Free-text rationale does not:** O1 (more information, incl. occasional artist names) = O0. A unique long text pooled into one thought vector is not usable; a shared vocabulary of 859 recurring motive units is.
- Explains earlier runs: all free-text support designs ≈ O0; predicted top-10 motive prototypes (archived 335237) = 11.06 vs oracle 12.29 → the gap is motive-prediction quality (true motive in top-10 for ~62% of rows; 10 candidates dilute 1–2 correct ones).
- Caveats: single seeds; O2 is an upper bound (motives derived from target + review).
- Original CollabRec reruns reproduce earlier numbers (emb 11.42 vs 11.48; text 11.35 vs 11.00) → seed spread ≈ ±0.2–0.35 R@10. All non-O2 runs (O0, O1, IntuRec, CollabRec emb/text) lie within 11.00–11.42 R@10: neither intuition titles nor peer rationales clearly beat the plain recommender. O2 is the only clear separation (+0.9 to +1.3 R@10, +0.55 to +0.70 N@10).

### 2026-10-23 — submitted: dilution test
| Job | Run | Purpose |
|---|---|---|
| 335881 | `CollabRec_oracle/text-oracle_own_motive_plus_pred_k10_beta0_rs1_lr0.0003_job335881` | **O2 + 8 predicted motives**: 2 true motive prototypes + the retriever's top-8 non-true predictions (train: out-of-fold), shuffled; β 0. Close to O2 (12.29) → the model finds true motives among distractors (motive recall matters); close to predicted top-10 (11.06) → distractors cancel the gain (precision matters). |

### 2026-10-25 — result: dilution test
| Run (job) | Supports | R@1 | R@5 | R@10 | N@5 | N@10 | valid NTP ep1/2/3 |
|---|---|---|---|---|---|---|---|
| O2 + 8 predicted (335881) | 2 true motive prototypes + top-8 non-true predicted, shuffled | 5.70 | 9.19 | 11.32 | 7.52 | 8.21 | 1.087 / 0.985 / 1.104 |

- 8 distractors erase O2's gain (12.29 → 11.32, back inside the 11.0–11.4 band of all non-oracle runs; valid NTP follows O0, not O2).
- **Precision, not recall, is the bottleneck:** even with the true motives guaranteed present, the model cannot tell which supports are right (InjectNet attention gets no selection signal). Raising the retriever's top-10 recall alone will not help.
- Implication: inject only high-confidence motives, weight by retriever probability, supervise motive selection, or have the model generate its own motive IDs as the reasoning step.

### 2026-10-29 — second dataset: Toys_and_Games (core scope: O0, O2, IntuRec)
Data copied from R3 (`data/Toys_and_Games/`, 5-core 2017-01 → 2018-11): train 53,898 / valid 6,737 / test 6,738 rows, 6,299 items; raw reviews (plain JSON lines) symlinked.

Motive pipeline for toys (same design as CDs, `--domain toys`):
- `extract_reviews.py`: plain `.json` input + toy motive keywords → informative rate 0.69 on all splits; all 65,429 (user, item) reviews matched.
- `gen_rationales.py`: toy prompt wording (no product/brand/franchise/character names; concrete triggers: gift, birthday, child's age, learning, family play, completing a set, replacing a broken piece, price) and 13 toy facets: gift_occasion, child_development_learning, age_stage_fit, interest_theme_affinity, franchise_brand_loyalty, collection_completion, creativity_building, social_family_play, hobby_enthusiast, active_outdoor_play, replacement_add_on, quality_safety, value_price. Music prompts verified byte-identical to before.
- `build_motive_corpus.py`: domain-specific naming prompt; `compare_clustering.py`: faiss imported lazily (encoding in racer, Louvain in rrec).
- `collab/submit_motive_corpus.sbatch`: vLLM Qwen3-8B on the job's GPU → v5 facet rationales (train/valid/test) → Qwen3-Embedding phrase vectors → Louvain (~1000) → merge / fold / name / drop complaints → `data/Toys_and_Games/collab/motive_corpus/`.

| Job | Run | Purpose |
|---|---|---|
| 342986 | `Toys_and_Games/CollabRec_oracle/…k0_beta0…` | O0: no supports |
| 342987 | `Toys_and_Games/IntuRec_original/IntuRecF1_micro8_job342987` | IntuRec stage 1 + intuition beams + stage-2 data |
| 342988 | `Toys_and_Games/IntuRec_original/IntuRecF2_micro8_job342988` | IntuRec stage 2 (after 342987) |
| 342991 | `MotiveCorpus` | toy motive corpus |
| 342992 | `Toys_and_Games/CollabRec_oracle/text-oracle_own_motive_k2_beta0…` | O2: own motive prototypes (after 342991) |

First submission (342778/79/80/84/85) all failed on node `lunar`: no CUDA toolkit (DeepSpeed import raises `CUDA_HOME does not exist` in the Trainer and in `save_pretrained`; O0 died at its first save after 67 min) and an old system libstdc++ (vLLM in racer: `GLIBCXX_3.4.30 not found`). Resubmitted with `--exclude=lunar`; the failed run folders (job342778, job342779) hold logs only. vLLM now runs in the `LLM` env (`module load CUDA/13.0.2`, vllm 0.31) in a subshell; the embedding / clustering / corpus steps stay in racer / rrec.

2026-10-29 evening:
- Motive corpus (342991, avenue, 4.3 h) done: 67,373 v5 rationales, 0 errors; Louvain 995 → merge 919 → fold 911 → 30 complaint clusters removed → **881 prototypes**; rows with no motive ≤ 0.6%; leak gap 0.0061 (CDs 0.0034). Largest prototype "complete collection" covers 7.5% of train rows (CDs max share smaller); next: family bonding through play, gift for special occasions, educational play. Facet mix: child_development_learning, creativity_building, social_family_play, collection_completion, gift_occasion lead; evidence = review for 82% of rows.
- Caveat: ~2.8k / 53.9k train rationales still name a franchise or brand (e.g. "Star Wars themed gift") despite the prompt; only 4 / 881 prototypes do (themes such as Disney / magic), so O2 inputs are essentially name-free. "complete collection" looks over-assigned (sometimes invented from the history).
- IntuRec F1 (342987, analog A100-40GB) finished stage 1: stage-1 test Recall@1/5/10 3.67 / 7.91 / 10.24, NDCG@10 6.59; now generating train intuition beams.
- O0 (342986) and O2 (342992) OOM'd within a minute on analog (A100 **40 GB**: Toys micro-batch 8 needs > 40 GB; it ran on the 48 GB L40S). Resubmitted as **343331 (O0)** and **343332 (O2)** with `--constraint="l40|l40s|a40"` (48 GB) + `--exclude=lunar`; IntuRec F2 342988 updated to the same constraint.

2026-10-30:
- 343331 / 343332 (O0 / O2) failed after ~1.5 h on lagoon at the first checkpoint save, and the requeued F1 342987 failed on lexicon: `CUDA_HOME does not exist`. Not lunar-specific: no node has a system CUDA anymore. Fix: all three training sbatch scripts now run `module purge; module load CUDA/12.4.0` (matches rrec torch cu124); vLLM keeps `CUDA/13.0.2` + LLM env.
- F1 342987 had been preempted on analog after 9 h (stage 1 done, train beams in progress); the requeue restarted from scratch. `submit_inturec_f1.sbatch` now skips stage-1 training when `model.safetensors` exists and skips beam files that exist, so requeues resume.
- Resubmitted: **343598** F1 (reuses run folder `IntuRecF1_micro8_job342987`, starts at beam generation), **343599** F2 (after 343598), **343600** O0, **343601** O2 (F2/O0/O2 on 48 GB GPUs).
- 15:35 O0 / O2 (343600 / 343601) were preempted on avenue in epoch 3 (epoch-2 valid NTP: O0 0.5549, O2 **0.4953**; epoch 1: 0.6372 / 0.6107); requeued, restart from scratch. 15:43 F1 343598 died at 59% of the train beams with no traceback: scratch was full (501 GB of ~500 GB; R3 holds 443 GB). Deleted the two HF Trainer intermediates of IntuRecF1_micro8_job342987 (checkpoint-844/1263, 17.4 GB; final model kept) → 481 GB. Resubmitted F1 **343927** (beams from scratch) and F2 **343928**; O0/O2 released.
- 16:13 user cleaned scratch: 163 GB used (~337 GB free); IntuRec checkpoints intact.

### 2026-10-31 — Toys_and_Games results
(both preempted twice more and restarted from scratch; final runs on lagoon / lotus)

| Run (job) | Support input | R@1 | R@5 | R@10 | N@5 | N@10 | valid NTP ep1/2/3 |
|---|---|---|---|---|---|---|---|
| O0 (343600) | none | 4.04 | 8.40 | 10.75 | 6.29 | 7.05 | 0.639 / 0.556 / 0.579 |
| **O2 oracle (343601)** | own 1–2 motive prototypes (k 2) | **5.22** | **11.77** | **16.22** | **8.58** | **10.00** | 0.616 / 0.502 / 0.517 |
| ref: IntuRec stage 1 (342987) | — | 3.67 | 7.91 | 10.24 | 5.83 | 6.59 | |

- O2 − O0 = **+5.47 R@10 / +2.96 N@10** (CDs: +1.29 / +0.70). Motive information helps far more on Toys.
- Caveat: O2 is an oracle (prototypes come from the target + review). Toys prototypes include item-descriptive themes (e.g. "unicorn-themed entertainment", "creative building", "outdoor active play") that partly name the target's category, so part of the gap may be target-attribute information rather than buyer motive. To check: split prototypes into buyer-motive vs item-descriptive and rerun O2 with each subset; and run the predicted-motive (non-oracle) version.
- IntuRec F1 343927 completed (beams train 16 / valid 15 / test 15 + stage-2 data); IntuRec stage 2 **343928** running on lagoon since 10:14.

### 2026-10-31 — Toys: CollabRec + predicted motives
- SASRec on Toys (user-trained, Verifiable-Rec `Toys_and_Games_5_2017-1-2018-11/models/SASRec/2026-10-31_12-00-01-480407.pkl`): test R@1/5/10 4.41 / 9.16 / 11.59, N@5/10 6.91 / 7.69 (valid R@10 10.84). Above O0 (10.75 / 7.05), far below O2 (16.22 / 10.00).
- `collab/submit_collab_supports.sbatch` (new, user's rrec + CUDA 12.4 recipe; finished steps skipped on requeue): tune_latent → build_stage2_support (`collab/stage2/`) → motive retriever (cls, 5-fold cross-fitted) → top-10 predicted motive supports (`motive_corpus/stage2/`, same format as CDs).

| Job | Run | Purpose |
|---|---|---|
| 345288 | `CollabSupports` | build the inputs above |
| 345289 | `CollabRec/text_k10_beta0.1…` | CollabRec text (peer rationales → InjectNet), after 345288 |
| 345290 | `CollabRec/emb_k10_beta0.1…` | CollabRec emb, after 345288 |
| 345291 | `CollabRec/text-motive_k10_beta0.1…` | top-10 **predicted** motives (non-oracle O2 counterpart; CDs 11.06), after 345288 |
- IntuRec F2 Toys (343928, lagoon, 5.9 h) test: R@1/5/10 3.13 / 7.42 / **10.34**, N@5/10 5.32 / **6.26** (valid NTP ep2/3 0.583 / 0.577). Below O0 (10.75 / 7.05) and SASRec (11.59 / 7.69); ≈ its own stage 1 (10.24 / 6.59).
- Supports job 345288 (20 min): latent predictor gte-large, τ 0.05, hidden 1024 (valid MRR 0.0235); alpha 0.5; test target_hit 0.092, rank MRR 0.0216 (CDs 0.0092). Motive retriever (cls) test R@1/5/10 9.8 / 23.6 / 31.8, MRR 0.290 (CDs 9.1 / 31.2 / 42.3, 0.323); a true motive is in the top-10 for 53.9 / 50.4 / 49.4% of rows (train OOF / valid / test; CDs ~60%).

### 2026-10-31 — Toys results: CollabRec emb + predicted motives
| Run (job) | Supports | R@1 | R@5 | R@10 | N@5 | N@10 | valid NTP ep1/2/3 |
|---|---|---|---|---|---|---|---|
| O2 oracle (343601) | own 1–2 motive prototypes | **5.22** | **11.77** | **16.22** | **8.58** | **10.00** | 0.616 / 0.502 / 0.517 |
| SASRec | — | 4.41 | 9.16 | 11.59 | 6.91 | 7.69 | |
| CollabRec emb (345290) | 10 peer rationales, mpnet + projection, β 0.1 | 3.96 | 8.40 | 11.31 | 6.20 | 7.15 | 0.645 / 0.559 / 0.583 |
| O0 (343600) | none | 4.04 | 8.40 | 10.75 | 6.29 | 7.05 | 0.639 / 0.556 / 0.579 |
| predicted motives (345291) | top-10 retriever motives, β 0.1 | 4.10 | 8.46 | 10.72 | 6.32 | 7.05 | 0.640 / 0.555 / 0.582 |
| IntuRec F2 (343928) | 15 intuition titles | 3.13 | 7.42 | 10.34 | 5.32 | 6.26 | |

- Same picture as CDs: every non-oracle design lands at O0 (10.3–11.3 R@10); only the oracle separates (+5.5 R@10). Predicted motives = O0 exactly (10.72 vs 10.75), as on CDs (11.06 vs 11.00) — with a true motive in the top-10 for only 49% of test rows and 8+ distractors, the model gets no usable signal (cf. the CDs dilution test).
- emb +0.56 R@10 over O0 is within the ±0.2–0.35 seed spread band's upper edge; single seed.
- CollabRec text (345289) preempted twice (18:49, 20:58); pending again, restarts from scratch.
- CollabRec text (345289, completed 11-01 on lexicon): R@1/5/10 4.01 / 8.15 / **10.75**, N@5/10 6.15 / **6.98**; valid NTP 0.636 / 0.556 / 0.585 — identical to O0.

### 2026-11-02 — IntuRec reproduction vs paper (arXiv 2606.27684, Table 2, IntuRec-B = standard decoding)
| Dataset | Run | R@5 | R@10 | N@5 | N@10 |
|---|---|---|---|---|---|
| Toys | paper IntuRec-B | 8.58 | 10.77 | 6.62 | 7.33 |
| Toys | ours 343928 (K 15, λ 1.0, N 1, micro 8) | 7.42 | 10.34 | 5.32 | 6.26 |
| Toys | paper LatentR3-B (≈ stage 1) | 7.73 | 10.60 | 5.73 | 6.65 |
| Toys | ours stage 1 | 7.91 | 10.24 | 5.83 | 6.59 |
| CDs | paper IntuRec-B | 9.86 | 12.21 | 7.95 | 8.71 |
| CDs | ours, 4 runs (12.08 / 11.55 / 11.55 / 11.19 R@10) | 9.26–9.52 | 11.64 mean | 7.60–7.75 | 8.38 mean |

- Data matches the paper (Toys 53,898 / 6,737 / 6,738 rows, 6,299 items; CDs 49,251 / 6,156 / 6,158); our Toys balance ratio 0.382 vs the authors' 0.40 (K 15, `preparedata2f2.ipynb`), computed the same way (test hit / train hit).
- Stage 1 reproduces (Toys N@10 6.59 vs 6.65). The gap is in stage 2: paper +0.68 N@10 over LatentR3-B on Toys, ours −0.33. On CDs our runs are −0.6 R@10 / −0.33 N@10 on average (best run 12.08 ≈ paper).
- Paper tunes K ∈ {5…30}, N ∈ {1…5}, λ ∈ {0.5, 1.0} per dataset (chosen values not reported) and does not report seeds or which epoch is decoded; repo defaults use micro-batch 2 / eval batch 2 (ours 8 / 8).
- Submitted **348597**: Toys IntuRec stage 2 exactly as the repo script (`run_inturec_f2.sh`): micro-batch 2, eval batch 2, batch 128, lr 3e-4, K 15, λ 1.0, N 1, 3 epochs, seed 0 → `IntuRec_original/IntuRecF2_micro2_job348597`. Same stage-2 data as 343928 (micro 8); isolates the batch-size effect.

### 2026-11-03 — CollabRec + collaborative InfoNCE (forked session, worktree `collab-refined-infonce`), Toys
| Run (job) | Supports | R@1 | R@5 | R@10 | N@5 | N@10 |
|---|---|---|---|---|---|---|
| predicted motives + InfoNCE (348439; cq 0.1, relations user/target/motive) | top-10 predicted motives, β 0.1 | 4.01 | 8.58 | **11.21** | 6.38 | **7.23** |
| predicted motives (345291) | same, no InfoNCE | 4.10 | 8.46 | 10.72 | 6.32 | 7.05 |
| O0 (343600) | none | 4.04 | 8.40 | 10.75 | 6.29 | 7.05 |
| O2 oracle + InfoNCE (348440) | own motives (k 2), β 0 | **5.77** | **13.22** | **18.61** | **9.59** | **11.33** |
| O2 oracle (343601) | own motives | 5.22 | 11.77 | 16.22 | 8.58 | 10.00 |

- InfoNCE adds +0.49 R@10 / +0.18 N@10 with predicted motives (single seed, inside the ±0.2–0.5 spread) and +2.39 R@10 / +1.33 N@10 on top of the oracle (larger than seed spread).
- Scratch quota (Ceph counter, not du: 534.6 / 536.9 GB) killed both CDs micro-2 repro runs at their first checkpoint save (348534 after 3.4 h, 348535 after 1.6 h). After the user's cleanup (221.7 GB used), resubmitted as **349506** and **349507**.
- Toys micro-2 repro 348597 preempted twice (4.9 h on analog, 0.6 h on lunar), restarting from scratch; attempt 3 running on alarm.

### 2026-11-03 — IntuRec CDs stage 2 at micro-batch 2 (repo-exact batch sizes), run 1 of 2 (values, not %)
| Run | micro | R@1 | R@5 | R@10 | N@5 | N@10 |
|---|---|---|---|---|---|---|
| paper IntuRec-B | 2 (script) | — | 0.0986 | 0.1221 | 0.0795 | 0.0871 |
| **349507 (lotus, nopreempt)** | 2 | 0.0594 | 0.0971 | **0.1179** | 0.0788 | **0.0854** |
| original run (Oct-15) | 2 | 0.0583 | 0.0952 | 0.1208 | 0.0775 | 0.0858 |
| original code, balance 0.295 | 2 | 0.0588 | 0.0950 | 0.1155 | 0.0771 | 0.0838 |
| rerun 335355 | 8 | 0.0572 | 0.0936 | 0.1155 | 0.0760 | 0.0831 |
| rerun 335690 | 8 | 0.0583 | 0.0926 | 0.1119 | 0.0761 | 0.0824 |

Micro-2 mean of 3 (0.1181 / 0.0850) vs micro-8 mean of 2 (0.1137 / 0.0828): +0.0044 R@10 / +0.0022 N@10; micro-2 vs paper: −0.0040 / −0.0021. Second micro-2 run 349506 pending.

### 2026-11-04 — IntuRec CDs micro-batch 2, run 2 (349506) + conclusion (values)
| Run | micro | stage-2 data | R@1 | R@5 | R@10 | N@5 | N@10 |
|---|---|---|---|---|---|---|---|
| paper IntuRec-B | 2 | — | — | 0.0986 | 0.1221 | 0.0795 | 0.0871 |
| **349506 (avenue)** | 2 | ours (ratio test/train) | 0.0580 | 0.0981 | **0.1202** | 0.0789 | **0.0860** |
| 349507 (lotus) | 2 | ours | 0.0594 | 0.0971 | 0.1179 | 0.0788 | 0.0854 |
| original run (Oct-15) | 2 | ours | 0.0583 | 0.0952 | 0.1208 | 0.0775 | 0.0858 |
| original code, balance 0.295 | 2 | authors' ratio | 0.0588 | 0.0950 | 0.1155 | 0.0771 | 0.0838 |
| rerun 335355 | 8 | ours | 0.0572 | 0.0936 | 0.1155 | 0.0760 | 0.0831 |
| rerun 335690 | 8 | ours | 0.0583 | 0.0926 | 0.1119 | 0.0761 | 0.0824 |

- Same data, micro 2 (3 runs): mean R@10 0.1196 (0.1179–0.1208), N@10 0.0857; micro 8 (2 runs): 0.1137 / 0.0828. **Micro-batch 2 is worth ≈ +0.006 R@10 / +0.003 N@10 on CDs**, larger than the micro-2 spread (0.003).
- Micro 2 vs paper: −0.0025 R@10 / −0.0014 N@10 → **CDs reproduces at micro 2** (within the paper's single-run noise).
- Consequence: all our CollabRec / O0 / O2 runs (micro 8) are comparable to each other, but IntuRec at micro 8 is understated by ~0.006 R@10. For CollabRec vs IntuRec claims, compare at the same micro-batch (re-run key CollabRec configs at micro 2, or report IntuRec at micro 8 explicitly).

### 2026-11-04 — IntuRec Toys stage 2 at micro-batch 2 (348597, sodag nopreempt), values
| Run | micro | R@1 | R@5 | R@10 | N@5 | N@10 |
|---|---|---|---|---|---|---|
| paper IntuRec-B | 2 | — | 0.0858 | 0.1077 | 0.0662 | 0.0733 |
| **348597** | 2 | 0.0420 | 0.0833 | **0.1112** | 0.0632 | **0.0721** |
| 343928 | 8 | 0.0313 | 0.0742 | 0.1034 | 0.0532 | 0.0626 |
| paper LatentR3-B (≈ stage 1) | — | — | 0.0773 | 0.1060 | 0.0573 | 0.0665 |

- Micro 2 vs 8 on Toys: **+0.0078 R@10 / +0.0095 N@10** (+0.0107 R@1, +0.0100 N@5): far larger than on CDs (+0.006 / +0.003).
- Micro 2 vs paper: R@10 +0.0035, N@10 −0.0012, R@5 −0.0025, N@5 −0.0030 → **Toys reproduces at micro-batch 2** (single run). Stage 2 now adds +0.0056 N@10 over the paper's LatentR3-B, as in the paper (+0.0068).
- So the IntuRec reproduction gap on both datasets was mainly the micro-batch size (per-micro-batch token-mean NTP weighting). All CollabRec comparisons need micro-batch 2 too (reruns 350999–351006 running).

### 2026-11-06 — Does IntuRec use its latent? Toys (355228, checkpoint IntuRecF2_micro2_job348597, epoch-2 best-valid), values
`collab/inturec_latent_diagnostics.py` (IntuRec files untouched; subclass adds a thought override). Full test (6,738 rows), Δ = variant − raw, paired bootstrap 95% CI. Target is in the row's own 15-title intuition list for 0.1213 of rows.

| condition | HR@10 | NDCG@10 | ΔHR@10 [CI] | top-10 overlap | top-1 same | copy rate | donor-target HR@10 | HR@10 target in / not in own list |
|---|---|---|---|---|---|---|---|---|
| raw | 0.1119 | 0.0701 | — | — | — | 0.418 | 0.0036 | 0.7271 / 0.0270 |
| mask | 0.1106 | 0.0695 | −0.0013 [−0.0031, +0.0001] | 0.922 | 0.916 | 0.419 | 0.0039 | 0.7222 / 0.0262 |
| mean | 0.1109 | 0.0695 | −0.0010 [−0.0025, +0.0003] | 0.936 | 0.926 | 0.419 | 0.0039 | 0.7283 / 0.0257 |
| shuffle | 0.1103 | 0.0692 | −0.0016 [−0.0030, −0.0003] | 0.933 | 0.926 | 0.418 | 0.0039 | 0.7222 / 0.0258 |
| swap_intuitions | 0.1103 | 0.0698 | −0.0016 [−0.0031, −0.0003] | 0.934 | 0.928 | 0.029 | 0.0036 | 0.7246 / 0.0255 |
| random_intuitions | 0.1107 | 0.0693 | −0.0012 [−0.0027, +0.0004] | 0.932 | 0.928 | 0.002 | 0.0037 | 0.7258 / 0.0258 |
| oracle_intuitions | 0.1106 | 0.0697 | −0.0013 [−0.0027, −0.0001] | 0.940 | 0.937 | 0.407 | 0.0037 | 0.7258 / 0.0257 |
| remove_target | 0.1109 | 0.0695 | −0.0010 [−0.0021, +0.0000] | 0.994 | 0.994 | 0.409 | 0.0036 | 0.7185 / 0.0270 |

Geometry of the test thoughts: effective rank 1.05, participation ratio 1.02, PC1 variance 0.992, random-pair cosine 0.9955, cosine to mean 0.9976 → **fully collapsed** (one direction; plain R3 v4: 21.3 / 0.885).

- IntuRec's prediction does **not** rely on its latent: every intervention moves HR@10 by ≤ 0.0016 (≈ 1.4% relative); shuffle / swap / oracle are nominally significant but tiny — the same size as R3's mean/shuffle effects (≤ 0.0015). Unlike R3, even mask is harmless.
- The intuition **content** is not used: putting the true target into the list does not help (−0.0013), removing it does not hurt (HR@10 on those rows 0.727 → 0.719), a donor's list is not followed (donor-target HR@10 at chance, copy rate 0.418 → 0.029). The 0.418 raw copy rate is not copying: stage 1 and stage 2 predict the same items from the same history.
- Consistent with IntuRec ≈ O0 at matched micro-batch: any IntuRec gain comes from training (micro-batch / loss weighting), not from latent reasoning at inference.

### 2026-11-07 — Does IntuRec use its latent? CDs (355229, checkpoint IntuRecF2_micro2_job349506, epoch-2 best-valid), values
Full test (6,157 rows); target in own intuition list for 0.1095 of rows.

| condition | HR@10 | NDCG@10 | ΔHR@10 [CI] | top-10 overlap | top-1 same | copy rate | donor-target HR@10 | HR@10 target in / not in own list |
|---|---|---|---|---|---|---|---|---|
| raw | 0.1191 | 0.0852 | — | — | — | 0.273 | 0.0107 | 0.7552 / 0.0409 |
| mask | 0.1140 | 0.0819 | **−0.0050 [−0.0075, −0.0026]** | 0.852 | 0.830 | 0.276 | 0.0107 | 0.7315 / 0.0381 |
| mean | 0.1161 | 0.0836 | **−0.0029 [−0.0050, −0.0006]** | 0.871 | 0.849 | 0.269 | 0.0109 | 0.7404 / 0.0394 |
| shuffle | 0.1174 | 0.0838 | −0.0016 [−0.0042, +0.0010] | 0.846 | 0.824 | 0.260 | 0.0106 | 0.7404 / 0.0409 |
| swap_intuitions | 0.1181 | 0.0843 | −0.0010 [−0.0037, +0.0016] | 0.844 | 0.826 | 0.059 | 0.0107 | 0.7433 / 0.0412 |
| random_intuitions | 0.1259 | 0.0893 | **+0.0068 [+0.0034, +0.0102]** | 0.713 | 0.681 | 0.011 | 0.0117 | 0.7671 / 0.0471 |
| oracle_intuitions | 0.1191 | 0.0854 | +0.0000 [−0.0016, +0.0015] | 0.895 | 0.877 | 0.258 | 0.0101 | 0.7537 / 0.0410 |
| remove_target | 0.1174 | 0.0848 | −0.0016 [−0.0036, +0.0003] | 0.926 | 0.912 | 0.267 | 0.0101 | 0.7463 / 0.0401 |

Geometry: effective rank 1.02, PC1 variance 0.998, random-pair cosine 0.953, cosine to mean 0.969 → collapsed, as on Toys.

**IntuRec conclusion (both datasets):** row-specific latent content is not used — oracle target in the list +0.0000 (CDs) / −0.0013 (Toys); removing it −0.0016 / −0.0010 (n.s.); a donor's thought or list is not followed (donor-target hits at chance). On CDs the slot matters as an activation (mask −0.0050, mean −0.0029, like R3's mask effect), and random intuitions *improve* HR@10 by +0.0068 (top-10 overlap 0.713): the stage-1 list seems to pull stage 2 toward stage 1's own predictions (single checkpoint, single random draw — to confirm). The latent is collapsed to one direction on both datasets (effective rank ≈ 1). IntuRec does not make prediction rely on latent reasoning; it behaves like R3.

### 2026-11-07 — CollabRec vs IntuRec at matched micro-batch 2 (reruns 350999–351006), values
CollabRec = top-10 predicted motives + collaborative InfoNCE (cq 0.1, relations user/target/motive), β 0.1. IntuRec CDs = mean of 3 micro-2 runs (349506, 349507, Oct-15).

| Dataset | Run (job) | R@1 | R@5 | R@10 | N@5 | N@10 |
|---|---|---|---|---|---|---|
| CDs | IntuRec F2 (mean of 3) | 0.0586 | 0.0968 | 0.1196 | 0.0784 | 0.0857 |
| CDs | O0, no supports (351003) | 0.0604 | 0.0968 | 0.1178 | 0.0796 | 0.0864 |
| CDs | CollabRec pred. motives + InfoNCE (351006) | 0.0604 | 0.0936 | 0.1171 | 0.0779 | 0.0855 |
| CDs | O2 oracle (351004) | 0.0622 | 0.1048 | 0.1252 | 0.0840 | 0.0906 |
| CDs | O2 oracle + InfoNCE (351005) | 0.0668 | 0.1078 | 0.1325 | 0.0877 | 0.0957 |
| Toys | IntuRec F2 (348597) | 0.0420 | 0.0833 | 0.1112 | 0.0632 | 0.0721 |
| Toys | O0, no supports (350999) | 0.0417 | 0.0867 | 0.1112 | 0.0650 | 0.0729 |
| Toys | CollabRec pred. motives + InfoNCE (351002) | 0.0417 | 0.0840 | 0.1103 | 0.0635 | 0.0719 |
| Toys | O2 oracle (351000) | 0.0487 | 0.1000 | 0.1358 | 0.0749 | 0.0865 |
| Toys | O2 oracle + InfoNCE (351001) | 0.0646 | 0.1402 | 0.1892 | 0.1032 | 0.1188 |

Micro-8 gate runs (motive gate 1.0 + InfoNCE): Toys 349555 R@10 0.1128 / N@10 0.0704; CDs 349556 0.1139 / 0.0816.

- At matched micro-batch, **non-oracle CollabRec = IntuRec = O0** on both datasets (CollabRec − IntuRec: CDs −0.0025 R@10 / −0.0002 N@10; Toys −0.0009 / −0.0002; inside the ~0.003 micro-2 seed spread). The micro-8 CollabRec "wins" over IntuRec on Toys came from IntuRec's micro-8 handicap.
- Oracle motives still separate clearly: O2 + InfoNCE − IntuRec = +0.0129 R@10 / +0.0100 N@10 (CDs), +0.0780 / +0.0467 (Toys). InfoNCE on top of O2 helps again at micro 2 (CDs +0.0073 R@10, Toys +0.0534).
- Toys O2 without InfoNCE is lower at micro 2 than at micro 8 (0.1358 vs 0.1622 R@10), unlike every other config; single seed each.

### 2026-11-07 — retrieval headroom: how predictable are motives at all? (`collab/retriever_analysis.py`, CPU, values)
User's plan: (1) improve motive retrieval, (2) make the motive readout robust to wrong motives. Before building either, measured the ceiling of a target-driven retriever. P@k = share of the top-k predicted motives that are true (what O2-style k 1–2 injection needs); rows with >= 1 motive.
Scorers: popularity; current MLP retriever (`retriever/cls`, train = out-of-fold); **target ceiling** = motive distribution of all train purchases of the row's TRUE target (leave-one-out on train) — the best any history → likely-item → motive retriever can do; SASRec candidate profile P(m|h) = sum_i softmax(SASRec/T)_i P(m|i) (T, K tuned on valid); log-linear MLP + candidate mix.

| test | Toys P@1 | Toys P@2 | Toys Hit@10 | Toys MRR | CDs P@1 | CDs P@2 | CDs Hit@10 | CDs MRR |
|---|---|---|---|---|---|---|---|---|
| popularity | 0.1580 | 0.1131 | 0.4358 | 0.2491 | 0.1198 | 0.1203 | 0.5800 | 0.2551 |
| current MLP (cls) | 0.1946 | 0.1480 | 0.4935 | 0.2896 | 0.1808 | 0.1657 | 0.6191 | 0.3233 |
| SASRec candidate profile | 0.1916 | 0.1448 | 0.4769 | 0.2819 | 0.1711 | 0.1552 | 0.6024 | 0.3113 |
| MLP + candidate | 0.2015 | 0.1515 | 0.4999 | 0.2958 | 0.1863 | 0.1676 | 0.6248 | 0.3284 |
| **target ceiling (true target known)** | **0.2889** | 0.2136 | 0.6172 | 0.3980 | **0.2264** | 0.1888 | 0.6034 | 0.3508 |
| O2 oracle input | 1.0 | 1.0 | — | — | 1.0 | 1.0 | — | — |

- **Motives are mostly row-specific, not item-specific:** even knowing the true next item, its motive profile gets the top-1 motive right for only 0.29 (Toys) / 0.23 (CDs) of rows. The O2 motives carry review-level information that neither the history nor the target's identity determines.
- The current retriever is already close to that ceiling (P@1 0.19 / 0.18); the missing target-candidate feature adds only +0.007 / +0.006 P@1. SASRec is heavily in-sample on train rows (target HR@10 0.83 train vs 0.09–0.12 test), a train/test mismatch for any SASRec-based feature.
- Consequence for both directions: with this motive vocabulary, a history-based retriever cannot exceed ~0.2–0.3 precision, so injected motives will be wrong 70–80% of the time; a robust readout can at best ignore them (the micro-8 predicted-motive model already ignores the slot: oracle motives at test ΔHR@10 +0.0001). A gain needs motives that are both predictable and informative.

### 2026-11-07 — motive utilization: gated / soft readouts + noisy-oracle training (user: focus on utilization first)
Motivation: the retrieval ceiling (above) means injected motives will be wrong 70–80% of the time, and the predicted-motive model learns to ignore the slot (oracle motives at test ΔHR@10 +0.0001). So the readout must learn which motives to use.
Code (worktree `collab-refined-infonce`; IntuRec files untouched, InjectNet's pooling / W_*_2 weights reused): `model/CollabRecF2.py` `--readout`
- `gated`: thought = base(query) + Σ_s sigmoid(r_s) · softmax(attn)_s · v_s; r_s = low-rank bilinear relevance (query, pooled motive); **selection BCE** vs "slot is a true motive" (`--sel_weight` 1.0). All gates closed → exactly the no-support thought (O0), so wrong motives can be switched off. Valid log prints mean gate on true vs wrong slots and gate AUC.
- `soft`: thought = base(query) + Σ_s softmax(attn_s + log π_s) · v_s, π = retriever probabilities (option C).
Data (`data/CollabRecDataset.py --motive_source`): `predicted` = retriever top-k (train OOF), `noisy` = each true motive kept with prob p else replaced by the top non-true prediction, rest filled with predictions, shuffled; train re-drawn every access, valid/test fixed per row. `--extra_test_predicted` decodes test a second time with the predicted top-k (`result/standard/predicted/`) = deployable number. CPU test `tests/test_motive_readout.py` (slot precision 0.245 / 0.494 / 0.986 at p 0.25 / 0.5 / 1.0; closed gates = base thought; logp reaches beam decoding).

Toys, micro-batch 2, β 0, no InfoNCE (compare O0 / IntuRec micro 2: 0.1112 R@10):

| Job | Run | Train supports | Readout | Question |
|---|---|---|---|---|
| 356474 | `CollabRec_oracle/text-micro2-noisy0.25-gated1.0_k2_beta0…` | noisy oracle p 0.25 | gated | can the gate exploit motives at ~ceiling precision? (+ predicted top-2 decode) |
| 356475 | `CollabRec_oracle/text-micro2-noisy0.5-gated1.0_k2_beta0…` | noisy oracle p 0.5 | gated | same at higher precision (+ predicted decode) |
| 356476 | `CollabRec_oracle/text-micro2-noisy0.5_k2_beta0…` | noisy oracle p 0.5 | inject | gate ablation (+ predicted decode) |
| 356477 | `CollabRec/text-micro2-predicted-gated1.0_k10_beta0…` | predicted top-10 | gated | deployable: gate selects among real predictions |
| 356478 | `CollabRec/text-micro2-predicted-soft_k20_beta0…` | predicted top-20 | soft | probability-weighted mixture (option C) |

### 2026-11-07 — verifying retrieved-motive quality: utility judge (user chose "utility judge first")
Why not label match: retriever confidence does select rows where its top-1 label is right (Toys test: top-10% most confident rows P@1 0.73 vs 0.195 overall; slot AUC over top-10 0.68–0.71), but 92% of those Toys rows predict the single generic motive "complete collection" (CDs: confidence barely helps, P@1 0.30–0.34 on the top 10%). Label precision rewards generic motives; what matters is whether a motive helps predict the item.

Utility judge `collab/motive_utility.py` (worktree `collab-refined-infonce`; sbatch `collab/submit_motive_utility.sbatch`): judge = trained O2 + InfoNCE model (Toys 351001, CDs 351005; best-valid epoch 2). Per test row, teacher-forced log-likelihood of the target title (+ `<|EndPredict|>`) with supports [m, m]:
u(m) = log p(target | history, m) − log p(target | history, "No specific motive.").
Candidates: null, true pair (O2's exact input), each true motive alone, retriever top-10, one random motive. Summary: share of the oracle gain recovered (u(top-1 pred), mean u(top-10), u(best of top-10) = perfect-verifier bound, all / u(true pair)); label audit (u of true vs wrong predicted motives, AUC, mean u and label precision of the most frequent top-1 motives). Caveat: single motives as [m, m] are off-distribution for O2 (98% of its rows had 2 distinct motives); 3-row CPU smoke: u(true pair) +1.07 vs u(true single) −0.07.

| Job | Dataset | Judge |
|---|---|---|
| 356488 | Toys test | 351001 |
| 356489 | CDs test | 351005 |

### 2026-11-07 — is history–motive agreement an indicator of a good motive? (`collab/motive_agreement.py`, CPU, label = "is a true motive")
Retriever top-10 candidates per test row. Agreement features from train only: hist_p = mean P(m | item) over history items; hist_lift = hist_p / P(m); user_p / user_lift = same from the user's own train-row motives; pop = P(m); logp = retriever.

| feature | Toys pooled AUC | Toys within-row AUC | Toys P@1 | CDs pooled AUC | CDs within-row AUC | CDs P@1 |
|---|---|---|---|---|---|---|
| retriever logp | 0.705 | 0.701 | 0.195 | 0.682 | 0.679 | 0.181 |
| hist_p | 0.700 | 0.686 | 0.190 | 0.671 | 0.663 | 0.172 |
| pop (genericness) | 0.647 | 0.664 | 0.182 | 0.600 | 0.603 | 0.120 |
| hist_lift | 0.614 | 0.569 | 0.067 | 0.605 | 0.589 | 0.099 |
| user_p | 0.528 | 0.546 | 0.184 | 0.535 | 0.543 | 0.174 |
| user_lift | 0.522 | 0.533 | 0.146 | 0.533 | 0.534 | 0.152 |
| LR(all + logp), fit valid | 0.725 | 0.712 | 0.201 | 0.691 | 0.686 | 0.184 |

- Raw history agreement looks as good as the retriever, but mostly because it encodes genericness (pop alone AUC 0.65 / 0.60). Corrected for genericness (lift), history agreement is weak (within-row AUC 0.57 / 0.59; P@1 below popularity); the user's past motives are near chance (0.53–0.55). Adding all agreement features to logp: +0.02 AUC, +0.006 P@1 — the retriever already uses them.
- Lift does pick far more specific motives (top-1 distinct motives Toys 311 vs 140, max share 0.06 vs 0.28 "complete collection"; CDs 105 vs 32). Specific motives are wrong by the labels but may be MORE useful — to re-test with utility labels from the judge (356488 / 356489).

### 2026-11-07 — do wrong motives harm prediction? (submitted)
`collab/latent_diagnostics.py --oracle_input --skip_geometry` (new: raw = O2's own motives; conditions null = "No specific motive." ×2, pred_top = retriever top-2, wrong_top = top-2 non-true predictions, half_wrong = 1 true + 1 wrong; plus swap_motives / random_motives; deltas vs raw AND vs null, paired bootstrap). Full test, constrained 10-beam decoding; outputs `<run>/wrong_motive/`. Harm = HR@10(wrong) − HR@10(null). 4-row CPU smoke: top-10 overlap with raw 0.15–0.45 under wrong motives (O2 follows its motives).

| Job | Model |
|---|---|
| 356547 | Toys O2 + InfoNCE (351001) |
| 356548 | CDs O2 + InfoNCE (351005) |

### 2026-11-07 — using the correctness labels: curriculum (submitted) and utility labels (prepared; user's choice)
- Early gate check (epoch 1, step 80): running selection BCE 0.488 (p 0.25) / 0.575 (p 0.5) vs base-rate entropy 0.562 / 0.693 → the gate learns beyond the prior. Possible shortcut in noisy-oracle runs: wrong slots are always the retriever's top non-true (often generic) motives, so the gate may learn "popular ⇒ wrong"; the predicted-motive decode and 356477 (trained on real predictions) will expose it.
- **Curriculum clean → noisy** (`--noisy_precision_start`, `NOISY_P_START`): per-epoch linear train precision 1.0 → 0.625 → 0.25 (valid/test at 0.25); model first learns to use motives (as O2), then to filter them. **356575** `CollabRec_oracle/text-micro2-noisy1.0to0.25-gated1.0_k2_beta0…` (Toys, gated, k 2, + predicted decode); compare with constant p 0.25 (356474).
- **Utility labels** (gate trained on "does m raise p(target)" instead of "is a true motive"): train-row utilities from the existing O2 judge would be in-sample (it trained on those rows with their true motives). User chose: first read the test-split judge (356488/9); only if utility differs from correctness, train two cross-fitted O2 judges on user halves and score each held-out half. Prepared, not submitted: `--train_fold i/n` (`TRAIN_FOLD`; stable crc32 user folds, Toys 27,384 / 26,514 train rows) and `collab/motive_utility.py --fold i/n` (scores only fold-i users).
- From now on jobs are submitted / monitored via the JobManager session (user request); it is monitoring 356474–356478, 356488/9, 356547/8, 356575.

### 2026-11-07 — utility judge result, Toys test (356488; judge 351001 O2 + InfoNCE), values (log-likelihood of the target title, nats)
u(m) = log p(target | history, m) − log p(target | history, null); null mean log p = −12.61.

| input | mean u | share of oracle gain |
|---|---|---|
| true pair (O2's input) | +1.752 | 1.00 |
| one true motive ([m, m]) | +1.025 | 0.59 |
| retriever top-1 | −0.695 | −0.40 |
| retriever top-10, mean | −0.761 | −0.43 |
| random motive | −1.540 | −0.88 |
| **best of retriever top-10** (perfect verifier bound) | **+1.729** | **0.99** |

| best of top-k | 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| rows with a useful candidate (u > 0) | 0.404 | 0.561 | 0.672 | 0.790 | 0.900 |
| mean max(u, 0) (verifier may abstain) | +0.686 | +0.932 | +1.134 | +1.405 | +1.786 |

Label audit: predicted motives that are true u +0.97, wrong −0.88; u > 0 for 0.70 of true vs **0.36 of wrong** predictions; AUC(u: true vs wrong) 0.73. Frequent top-1: "complete collection" (0.28 of rows, label precision 0.44): u +1.76 when true, −1.04 on average — useful when right, harmful when wrong (not merely generic); "Enhance Family Game Nights" u −3.43 on average.

- **Wrong motives harm** a model that reads motives: wrong predictions −0.88 nats, random −1.54; injecting the retriever's top-1 is net harmful (−0.69). (HR@10 version: 356547 / 356548.)
- **Utility ≠ correctness:** a third of "wrong" predictions help and 30% of "true" ones do not (AUC 0.73) → the user's condition for utility labels is met; launching 2-fold judges (Toys).
- **Large verifier headroom:** 90% of rows have a useful motive among the top-10, and a perfect verifier choosing the best of top-10 recovers 0.99 of the oracle gain (upper bound; selection over 10 candidates). Picking among existing candidates is the bottleneck, not the candidate pool.
- [m, m] format check: one true motive gives +1.03 (0.59 of the pair), so single-motive scoring is usable (the 3-row smoke test was misleading).

### 2026-11-07 — utility judge result, CDs test (356489; judge 351005), values (nats)
| input | Toys u | CDs u | CDs share of oracle gain |
|---|---|---|---|
| true pair | +1.752 | +0.650 | 1.00 |
| one true motive | +1.025 | +0.445 | 0.68 |
| retriever top-1 | −0.695 | +0.044 | 0.07 |
| top-10 mean | −0.761 | −0.092 | −0.14 |
| random | −1.540 | −0.184 | −0.28 |
| best of top-10 | +1.729 | +1.096 | 1.69 |

Label audit CDs: u > 0 for 0.64 of true vs 0.44 of wrong predictions; AUC(u: true vs wrong) 0.64 (Toys 0.73).
- CDs: smaller oracle gain (as O2 − O0 is smaller on CDs) and much milder harm from wrong motives (−0.15 vs −0.88); usefulness is even less tied to the labels than on Toys.
- Best of top-10 exceeds the true pair (1.10 vs 0.65): partly real (some predicted motives help more than the labelled ones), partly selection bias of a max over 10 candidates — treat best-of-k as a loose upper bound.

### 2026-11-07 — noisy-oracle runs, epoch-1 validation (Toys; valid motives at the run's own p)
| run | valid NTP | gate mean, true slots | gate mean, wrong slots | gate AUC |
|---|---|---|---|---|
| 356474 noisy 0.25, gated | 0.6826 | 0.565 | 0.185 | 0.813 |
| 356475 noisy 0.5, gated | 0.6805 | 0.717 | 0.352 | 0.830 |
| 356476 noisy 0.5, inject (no gate) | 0.6755 | — | — | — |
- Gate separates true from wrong slots (AUC 0.81–0.83) better than the retriever's own score (~0.70), but wrong slots here are the retriever's top non-true motives → possible "popular ⇒ wrong" shortcut; the predicted-motive decode decides.
- After epoch 1 the ungated run has slightly lower valid NTP (0.6755 vs 0.6805): gate not yet paying off in the loss; test HR@10 decides.
- **356477 predicted top-10, gated** (real retriever predictions, deployable): valid NTP 0.6811, selection 0.2174; gates true 0.136 (n 4289) vs wrong 0.065 (n 63,081), **AUC 0.679**: below the retriever's own ~0.70 and far below the noisy-oracle 0.81–0.83 → on real predictions the gate adds no verification beyond the retriever score; the noisy-oracle AUC was likely the "popular ⇒ wrong" shortcut. Gates mostly closed (precision@10 ~0.06), so the model largely falls back to the no-support thought. Preempted 22:38 (3h52m), requeued.
- All three preempted after ~5h15m (22:27), requeued, resume from `resume/state.pt`. Fold judges 356691/356692 moved to hadylauwresearch A40 (user OK); CDs fold judges wait for the Toys utility-label result (user).

### 2026-11-07 — utility-label gate training: code ready (worktree collab-refined-infonce)
- `--utility_files a.npz,b.npz --utility_margin 0` (`UTILITY_FILES`, `UTILITY_MARGIN`; run tag `-util0`): train-split gate label of a slot = u(m) > margin, u from the cross-fitted judges' `utility_train_fold{0,1}of2.npz` (each judge scores the users it did not train on). Uses single-motive [m, m] scores (true singles + retriever top-10), which cover every slot of `predicted` (top-k) and `noisy` (true + top-(k+2)) sources; any uncovered slot keeps the correctness label. Valid/test keep correctness labels (valid gate AUC stays comparable).
- Verified: synthetic utility file with inverted labels → 0 mismatches on 200 train rows; existing tests pass.
- Planned once 356693/356694 finish (Toys, gated, micro 2): (a) predicted top-10 + utility labels vs 356477 (predicted k 10, correctness labels); (b) noisy 0.25 + utility labels vs 356474.

### 2026-11-08 — scratch quota failure; first test results of the utilization runs (Toys, micro 2), values
- ~05:21 the scratch quota (~504 GB) filled (resume states ~70 GB); every running/requeued job failed (356474–356477, 356575, 356691, 356692, 356547); 356693/356694 dependency never satisfied. Resume states have since been deleted (no resume possible). Code change: `--resume_every` default 0 (`RESUME_EVERY`), existing states still loaded.

| run (job) | test motives | R@1 | R@5 | R@10 | N@5 | N@10 |
|---|---|---|---|---|---|---|
| O0, no supports (350999) | — | 0.0417 | 0.0867 | 0.1112 | 0.0650 | 0.0729 |
| O2 oracle (351000) | own true | 0.0487 | 0.1000 | 0.1358 | 0.0749 | 0.0865 |
| predicted top-20, soft readout (356478, 3 epochs) | retriever top-20 | 0.0423 | 0.0867 | 0.1118 | 0.0653 | 0.0734 |
| curriculum noisy 1.0→0.25, gated (356575, died before predicted decode) | noisy oracle p 0.25 | 0.0432 | 0.0874 | 0.1119 | 0.0664 | 0.0743 |
- Soft readout with the retriever prior = O0 (+0.0006 R@10): no gain from real predictions.
- Curriculum model at test precision 0.25 (≈ the retriever's, but with true motives at slot level) = O0 (+0.0007 R@10, +0.0014 N@10). At p 0.25 a gate that filters well should still beat O0 (≈ half the rows have a true motive in a slot); it does not → the gate/readout recovers little of the O2 gain (0.1358) once 3 of 4 slots are wrong.
- CDs wrong-motive harm (356548, O2 + InfoNCE judge 351005, HR@10 vs raw 0.1174): swap_motives −0.0286, random −0.0166, wrong_top −0.0161, pred_top −0.0148, half_wrong −0.0102, null −0.0084, all significant → a model that reads motives is harmed by wrong ones (as utility judge showed); predicted top motive is worse than no motive.

### 2026-11-08 — wrong-motive harm, Toys (357952; model 351001 O2 + InfoNCE, own-motive inputs), values
Full test (6,738 rows); HR@10 deltas with paired-bootstrap 95% CI. (Diagnostic decode raw 0.1775 vs the run's own test R@10 0.1892: different decode setup; compare within the table.) Reference: O0 (no-support model) 0.1112.

| input to the O2 model | HR@10 | NDCG@10 | Δ vs raw | Δ vs null |
|---|---|---|---|---|
| raw (own true motives) | 0.1775 | 0.1096 | — | — |
| half_wrong (one true + one wrong) | 0.1270 | 0.0814 | −0.0505 | **+0.0508** |
| pred_top (retriever top motives) | 0.1015 | 0.0664 | −0.0760 | **+0.0252** |
| wrong_top (top non-true predictions) | 0.0932 | 0.0619 | −0.0843 | +0.0169 |
| null ("No specific motive.") | 0.0763 | 0.0506 | −0.1012 | — |
| random_motives | 0.0548 | 0.0353 | −0.1227 | −0.0215 |
| swap_motives (another row's true) | 0.0525 | 0.0339 | −0.1250 | −0.0237 |
- **The O2 model is motive-dependent on Toys:** without a motive it drops to 0.0763, far below O0 0.1112; with the retriever's motives 0.1015, still below O0 → an O2-trained model cannot be deployed with predicted motives (consistent with predicted-motive runs ≈ O0: they learn to ignore motives instead).
- Unlike CDs (pred_top −0.0064 vs null), on Toys even wrong-but-plausible top predictions beat null (+0.0169): the retriever's wrong motives are near-misses that still point to the right region; random / other users' motives are harmful (−0.02 vs null; swap's donor-target HR@10 0.0677 = the model follows the motive to the donor's target).
- One true motive next to a wrong one keeps ~half of the oracle gain over null (0.1270): a true motive is not cancelled by a wrong one on Toys (on CDs half_wrong was −0.0102 vs raw, only −0.0018 vs null).
- The likelihood-based utility (top-1 u −0.70 vs null) and HR@10 (pred_top +0.025 vs null) disagree in sign on Toys: wrong motives lower p(exact title) but still help ranking → utility labels from log-likelihood may be too strict; consider labelling by rank/HR of the target instead.
- Fixed `collab/submit_latent_diag.sbatch`: the slurm tail now shows `diag_${OUT_NAME}.log` (it printed the stale `diag.log`).

### 2026-11-08 — fold-0 utility judge (357943, O2 + InfoNCE trained on users outside fold 0 of 2, ~half the train rows)
- Valid NTP 0.9278 / **0.6743** / 0.6880 (best epoch 2). Test (own true motives): R@1/5/10 0.0485 / 0.1113 / **0.1572**, N@5/10 0.0803 / 0.0951; full-data judge 351001: R@10 0.1892. Half the data costs ~0.03 R@10 but the judge still reads motives strongly (≫ O0 0.1112) → usable as a cross-fitted judge. Scorer 357945 (fold-0 users of train) running; fold-1 judge 357944 still training.
- Fold-1 judge (357944): valid NTP 0.9222 / **0.6828** / 0.7041; test R@1/5/10 0.0430 / 0.0988 / **0.1330**, N@10 0.0827 — weaker than fold 0 (0.1572) though trained on a similar number of rows (single-seed spread + fold difference); utilities of fold-1 users come from this weaker judge, so their magnitudes may be smaller (labels use only the sign, u > 0). Scorer 357946 queued.

### 2026-11-09 — train-split utilities, fold 1 (scorer 357946, judge 357944, 26,514 rows), values (nats)
u: true pair +1.265, single true +0.794, top-1 −0.248, top-10 mean −0.228, random −0.503, best of top-10 +1.645. Label audit: top-1 label precision 0.203; predicted true u +0.971, wrong −0.317; u > 0 for 0.693 of true vs 0.433 of wrong predictions; AUC 0.683 (Toys test with the full judge: 0.70 / 0.36 / 0.73).
- **Gate label balance changes completely:** among the retriever's top-10 slots, correctness positives 0.069 vs utility positives (u > 0) **0.451**; top-2 slots 0.455, and 0.661 of rows have a useful motive in the top-2.
- Caveat: u is measured against null for an O2-trained judge, which is crippled without a motive (Toys HR@10 null 0.0763 vs O0 0.1112), so "better than null" is a low bar: many plausible-but-wrong motives clear it. If the utility-gated runs open the gate on almost everything, try a stricter margin (`UTILITY_MARGIN`, e.g. the median u of true motives or ~0.5 nats).
- Fold 0 (scorer 357945, judge 357943, 27,384 rows): true pair +1.114, single true +0.655, top-1 −0.271, top-10 mean −0.436, random −0.957, best of top-10 +1.373; u > 0 for 0.672 of true vs 0.378 of wrong predictions; AUC 0.699. Consistent with fold 1 (wrong-but-useful 0.38 / 0.43).
- Utility-gated runs started 2026-11-09 ~04:10: **358873** predicted top-10 gated util0 (lexicon), **358874** noisy 0.25 gated util0 (avenue).
- Fold-0 scorer 357945 restarted after a preemption (2h06m lost; scorer now saves partial results every 200 batches, resumes from `partial_<tag>.npz`); ETA ~04:05. Utility-gated runs 358873 (predicted k10) / 358874 (noisy 0.25) wait on it.

### 2026-11-09 — noisy-oracle p 0.25, gated, correctness labels (357938, rerun of 356474), values
Valid NTP 0.6801 / **0.5896** / 0.6134; gates true vs wrong 0.569/0.184 (AUC 0.813) → 0.569/0.145 (0.826) → 0.599/0.162 (0.830).

| test motives | R@1 | R@5 | R@10 | N@5 | N@10 |
|---|---|---|---|---|---|
| O0 reference (350999) | 0.0417 | 0.0867 | 0.1112 | 0.0650 | 0.0729 |
| noisy oracle p 0.25 | 0.0435 | 0.0844 | 0.1097 | 0.0646 | 0.0727 |
| retriever top-2 | 0.0432 | 0.0844 | 0.1104 | 0.0646 | 0.0730 |
- **= O0** under both test inputs, although the gate separates true from wrong slots well on the noisy input (AUC 0.83, true gates 3.7× wrong). Selection works; **use** does not: trained where 3 of 4 slots are wrong, the model learns a motive-independent predictor and the gated motive barely moves the output. Same for the curriculum (0.1119): the last noisy epoch undoes the clean start.
- Contrast: the O2 model keeps half its oracle gain with one true + one wrong motive (half_wrong 0.1270 vs null 0.0763) → a model that learned to use motives on clean input can exploit a true motive among wrong ones; the failure is in training under noise, not in the information.
- **p 0.5 (357939, rerun of 356475)** confirms it: valid NTP 0.6878 / **0.5883** / 0.6119, gates true/wrong 0.735/0.307 (AUC 0.846); test noisy oracle p 0.5 R@10 0.1110 / N@10 0.0736, retriever top-2 0.1121 / 0.0736 → = O0 even when half of all slots are true. The O2 model given exactly this kind of input (one true + one wrong, half_wrong) reaches 0.1270. Training under noise, not selection, is the bottleneck.
- **Gate ablation, p 0.5 without gate (357940, inject):** valid NTP 0.6770 / **0.5886** / 0.6210; test noisy p 0.5 R@10 0.1103 / N@10 0.0732, retriever top-2 0.1088 / 0.0725 → same as gated p 0.5 (0.1110 / 0.1121) and O0: the gate neither helps nor hurts; all noisy-trained models ≈ O0.
- Idea (not run): **decouple use from selection** — freeze/keep an O2-trained model (learns to use motives on clean input) and train only a selector/gate on noisy candidates (utility labels), with fallback to an O0 model when the selector abstains (O2 alone collapses without a motive: null 0.0763 < O0 0.1112). Upper bound from the utility judge: best-of-top-10 recovers ≥ the oracle gain on train folds.

### 2026-11-09 — utility-label gates, epoch 1 (valid gates still scored against correctness labels)
| run | valid NTP | gate true | gate wrong | AUC vs correctness | correctness-label twin (epoch 1) |
|---|---|---|---|---|---|
| 358873 predicted top-10, util0 | 0.6821 | 0.461 | 0.407 | 0.567 | 356477: 0.6811, 0.136 / 0.065, 0.679 |
| 358874 noisy 0.25, util0 | 0.6836 | 0.535 | 0.433 | 0.712 | 357938: 0.6801, 0.569 / 0.184, 0.813 |
- As designed, utility gates open on ~40–45% of slots (train utility positive rate 0.45) instead of ~6–18%; separation against correctness drops (a perfect utility predictor would score only ~0.70 AUC vs correctness). No valid-split utility labels, so gate quality vs utility is not measured; NTP unchanged. Test HR decides.

### 2026-11-09 — utility-label gate, predicted top-10 (358873), test, values
Best epoch 2 (valid NTP 0.5896). Test (retriever top-10): R@1/5/10 0.0414 / 0.0840 / **0.1095**, N@5/10 0.0635 / **0.0717** → = O0 (0.1112 / 0.0729). Utility labels do not change the outcome: like every noisy/predicted-trained model, it ignores motives. Correctness-label twin **357941** (predicted top-10, gated; epochs NTP 0.6769 / **0.5883** / 0.6099, gates true/wrong 0.148/0.066 AUC 0.683 at best epoch): R@1/5/10 0.0410 / 0.0830 / **0.1069**, N@5/10 0.0624 / **0.0701**. Utility labels +0.0026 R@10 / +0.0016 N@10 over correctness labels, within seed spread (~0.003), and both ≤ O0 (0.1112 / 0.0729).

- **358874** noisy 0.25 gated + utility labels (epoch-2 model, test-only job 359526): noisy-oracle p 0.25 test R@1/5/10 0.0404 / 0.0839 / **0.1080**, N@5/10 0.0629 / **0.0708** vs correctness-label twin 357938 0.1097 / 0.0727 → no gain. Predicted decode skipped (job preempted mid-decode, cancelled: conclusion unchanged).
- **357942** curriculum 1.0→0.25 gated (rerun of 356575): valid NTP 0.6808 / **0.5842** (best, epoch 2, trained at p 0.625) / 0.6074; gates AUC 0.775 / 0.813 / 0.813. Test noisy p 0.25 R@10 **0.1115** / N@10 0.0738; retriever top-2 R@10 **0.1121** / N@10 0.0740 (first attempt 0.1119 / 0.0743) → = O0, reproducible. Note the best checkpoint is the epoch-2 (p 0.625) model, so even a mostly-clean-trained gated model gains nothing with predicted motives.
- **Conclusion (8 runs):** gates (correctness or utility labels), soft prior, curriculum: all noisy/predicted-trained models = O0 (R@10 0.107–0.112). Motive *selection* is not the bottleneck; models trained on noisy motives never learn to *use* motives. Next: use a clean-trained O2 reader with candidate motives (marginalize over top-k at inference; then RAG-style marginal-likelihood training).

### 2026-11-09 — retest of the 8 motive-utilization checkpoints with a better retriever (handed over from the CollabRec_GRPO session; user request)
Each saved best-valid (epoch-2) checkpoint decoded on Toys test twice with top-10 motives from (1) `cls` (MLP retriever; motive HR@1 0.1946 / HR@10 0.4935) and (2) `mlp_cand` = log_softmax(MLP) + 0.5 · log cand(m|h), cand from SASRec top-50 items (T 4) × smoothed P(m|item) (HR@1 0.2015 / HR@10 0.4999; top-10 overlap with cls 8.45/10, same top-1 0.747). Code (worktree, commit da13562): `collab/export_mix_scores.py`, `collab/decode_with_scores.py`, `collab/submit_decode_scores.sbatch`; outputs `<run>/result/standard/retest_<set>_top10/`. cls at the same checkpoint is the like-for-like reference (earlier test numbers mostly came from the in-memory epoch-3 model). 5 of 8 models trained with k 2 → top-10 at test is off-distribution.
Jobs: 357938→360210, 357939→360211, 357940→360212, 357941→360213, 356478→360214, 356575→360215, 358873→360216, 358874→360217. Retest decodes resumable since 2026-11-10 (`partial.json` every 25 batches; `evaluate(..., partial_every)`).

| model (checkpoint) | top-10 from | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| 357941 predicted gated k10 | cls | 0.0378 | 0.0810 | 0.1060 | 0.0378 | 0.0605 | 0.0686 |
| | mlp_cand | 0.0380 | 0.0813 | 0.1067 | 0.0380 | 0.0607 | 0.0689 |
| 358873 predicted gated util0 k10 | cls | 0.0411 | 0.0846 | 0.1116 | 0.0411 | 0.0635 | 0.0722 |
| | mlp_cand | 0.0413 | 0.0846 | 0.1115 | 0.0413 | 0.0636 | 0.0722 |
| 356575 curriculum 1.0→0.25 gated k2 | cls | 0.0421 | 0.0852 | 0.1126 | 0.0421 | 0.0643 | 0.0732 |
| | mlp_cand | 0.0421 | 0.0843 | 0.1119 | 0.0421 | 0.0640 | 0.0730 |
| 357938 noisy 0.25 gated k2 | cls | 0.0421 | 0.0846 | 0.1092 | 0.0421 | 0.0640 | 0.0720 |
| | mlp_cand | 0.0416 | 0.0837 | 0.1077 | 0.0416 | 0.0634 | 0.0712 |
| 357940 noisy 0.5 inject k2 | cls | 0.0401 | 0.0839 | 0.1100 | 0.0401 | 0.0627 | 0.0711 |
| | mlp_cand | 0.0392 | 0.0831 | 0.1100 | 0.0392 | 0.0620 | 0.0707 |
| 357939 noisy 0.5 gated k2 | cls | 0.0404 | 0.0833 | 0.1104 | 0.0404 | 0.0629 | 0.0718 |
| | mlp_cand | 0.0402 | 0.0837 | 0.1101 | 0.0402 | 0.0631 | 0.0717 |
| 356478 predicted soft k20 (top-10 at test) | cls | 0.0441 | 0.0844 | 0.1103 | 0.0441 | 0.0650 | 0.0734 |
| | mlp_cand | 0.0441 | 0.0839 | 0.1094 | 0.0441 | 0.0647 | 0.0730 |
| 358874 noisy 0.25 gated util0 k2 | cls | 0.0405 | 0.0836 | 0.1083 | 0.0405 | 0.0630 | 0.0710 |
| | mlp_cand | 0.0410 | 0.0839 | 0.1080 | 0.0410 | 0.0632 | 0.0711 |

- **Better retriever, no gain:** mlp_cand − cls HR@10 across the 8 models: +0.0007, −0.0001, −0.0007, −0.0015, 0.0000, −0.0003, −0.0009, −0.0003 (mean −0.0004); NDCG@10 within ±0.0008. All 16 decodes 0.106–0.113 HR@10 = O0 (0.1112). Expected: the retriever gain is small (HR@1 +0.007) and these models ignore motives, so retrieval quality cannot show through. Retrieval improvements only matter for a model that uses motives (O2-style reader; see the decoupling / marginalization proposal).


### 2026-11-12 — code recovery + predicted top-k sweep (k 2 / 3 / 5)
- The `collab-refined-infonce` worktree (all CollabRec code: run_collabrec.py, model/CollabRecF2.py, data/CollabRecDataset.py, submit_collabrec.sbatch, collab/{latent_diagnostics,motive_utility,decode_with_scores,export_mix_scores}.py, tests) was deleted, and the repo was re-initialized (old branches/commits gone). Rebuilt by replaying every recorded Write/Edit/patch from the session transcripts onto the main-checkout base: run_collabrec.py / CollabRecF2.py / CollabRecDataset.py / submit_collabrec.sbatch match the md5s logged on 2026-11-09 exactly (93dbc537 / 81be516a / 093d72d0 / f5b83805), then the later edits (resume_every 0, partial decode, utility labels) were replayed. CPU tests pass (motive_readout, resume, collab_queue, thought_override). Location: `/common/scratch/users/m/mk.ngohuu.2024/IntuRec/code/collab-refined-infonce-recovered` (data symlinked to the main checkout); submit with `CODE_DIR=<that dir>`. Same day (user): worktree restored at `.claude/worktrees/collab-refined-infonce` (new branch `worktree-collab-refined-infonce` from the re-init commit f3af79a; CollabRec files untracked as before, IntuRec files = the pre-reinit originals; identical to the scratch copy, which the k-sweep jobs 364238–40 still use).
- User: measure the effect of the number of retrieved motives. Toys, `MOTIVE_SOURCE=predicted` (cls retriever top-k; train OOF), inject readout, β 0, no InfoNCE, micro 2, k ∈ {2, 3, 5} (same recipe as O0 350999 / O2 351000). Runs `text-micro2-predicted_k{2,3,5}_beta0_rs1_lr0.0003_job<ID>`: k2 364238, k3 364239, k5 364240 (researchshort, l40|l40s|a40, excl. album/alarm/avenue).

## Next steps

- Represent reasoning as **discrete shared motive units**, not free text (O2 vs O1).
- Dilution test shows **precision/selection** is the bottleneck: probability-weighted motive injection; auxiliary loss supervising selection of the true motive among candidates; scheduled sampling from oracle to predicted motives.
- Generate motive IDs as the reasoning step (motive codewords).
- Seeds (2–3) for O0, O2 and IntuRec before claiming the gaps.
