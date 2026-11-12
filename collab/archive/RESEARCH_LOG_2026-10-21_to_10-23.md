# Research log — Collaborative (latent) reasoning for recommendation

Dataset: Amazon CDs_and_Vinyl, 5-core, 2015-10 → 2018-11, **time split** (train 49,251 rows / 6,865 users; valid 6,156; test 6,157; 5,841 items; only 49.4% of test users appear in train).
Base LLM: Qwen2.5-1.5B (full fine-tune). Teacher LLM: Qwen3-8B (vLLM). Python env `~/.conda/envs/rrec` (Qwen3 models need `racer`: transformers ≥ 4.51).
Last updated: 2026-10-22 21:30.

---

## 0. Reference baselines (test, top-10 generation / ranking over items)

| Model | R@1 | R@5 | R@10 | N@5 | N@10 | Source |
|---|---|---|---|---|---|---|
| SASRec | 5.33 | 9.85 | 12.19 | 7.73 | 8.48 | `Verifiable-Rec/code/baselines/results/sasrec/.../metrics.json` |
| IntuRec stage 2 (original run, epoch 3) | 5.83 | 9.52 | 12.08 | 7.75 | 8.58 | `log/CDs_and_Vinyl/inturec-f2-lr_3e-4-intunum_15-alpha_1.0-train-standard-rs_1.log` |
| IntuRec stage 2 (original code, balance ratio 0.295) | 5.88 | 9.50 | 11.55 | 7.71 | 8.38 | `..._balance_ratio0295.log` |
| IntuRec stage 2 rerun (micro 8, A40) | 5.72 | 9.36 | 11.55 | 7.60 | 8.31 | `inturec-f2-micro8_job335355-...log` |
| **IntuRec, mean of 3 runs (use as reference)** | 5.81 | 9.46 | **11.73** | 7.69 | **8.42** | 12.08 is the outlier (spread ≈ 0.5 R@10 between near-identical runs) |

---

## 1. Original idea and hypothesis

**Idea:** replace IntuRec's intuition latent (pooled stage-1 candidate titles) with *collaboratively retrieved reasoning*: rationales of similar users / items.
**Hypothesis:** reasoning has collaborative signal — if two users have similar interests, their motivations for buying the same item are similar.

Early design decisions (with the user):
- Retrieve support rationales from **train** rows only (never valid/test), keyed on history only, so retrieval is identical at train and test.
- Target-aware balancing of supports (as IntuRec's candidate balancing): train supports whose target equals the query target are thinned to the test rate.
- Rationale source: **LLM rationale grounded in the user's own review** (option "B grounded in A"); not latent-only (circular with history).

---

## 2. Reviews and rationale generation (`collab/extract_reviews.py`, `collab/gen_rationales.py`)

- Joined Amazon 2018 reviews: 60,286/60,286 interactions matched. Median review 9 words; 28% < 5 words.
- `informative` flag: ≥ 12 words and not a bare star phrase (33% train); later + motive keywords (gift, soundtrack, vinyl, live, price, replace…) → 37.6%.

### Facet / prompt iterations (held-out 504 train rows)

| Version | Change | Key metrics |
|---|---|---|
| open pilot | free-form motives, no list | 877 unique motives; seed list missed emotional (~12%), artistry (~9%), value, holiday |
| v1 | 10 seed facets | taxonomy coverage 95.1%, assignment recall 42.9% |
| v2 | 13 facets + definitions, motives-then-facets in one call | same-call recall 79%, coverage 88%, largest facet 53% |
| v3 | widened definitions, 1–2 facets | same-call coverage 96.1%, recall 85.3%, largest facet 49.8% (genre filler) |
| v4 | narrower genre def, review priority, banned phrases | recall 85.1%, largest facet 35.3%, reason pairwise cos 0.551 |
| **v5 (final)** | **first-person reason ("I …")** | recall 83.1%, coverage 93.9%, largest 36.1%, **reason cos 0.479**, banned phrases 12.7% |

Findings: the model invents a default "emotional" story when evidence is thin; self-reported `evidence` field unreliable (use the deterministic `informative` flag); v5 rationales still sometimes contain artist names.
Full v5 generation: train 49,251 / valid 6,156 / test 6,157, 0 errors (`data/.../collab/rationales_facet_{split}_v5.jsonl`). Valid/test rationales are for evaluation / loss targets only, never model inputs.

---

## 3. First support retrieval (`collab/build_support.py`)

Pool = train rows; exclude self and same user's later rows; ≤ 1 support per user; calibrated (bisection) target-aware balancing.

| Encoder | Side | Test target-in-support | Train after balancing |
|---|---|---|---|
| SVD | user | 6.5% | 6.6% |
| SVD | item (co-occurrence) | 13.9% | 14.1% |
| SASRec (Verifiable-Rec ckpt) | user | 11.0% | 11.1% |
| SASRec | item (top next-item) | 11.5% | 11.3% |

SASRec memorises train: 82% train vs 11.5% test target hit before balancing.

---

## 4. Hypothesis test (`collab/hypothesis_test.py`) — **hypothesis not supported as stated**

300,825 same-target pairs (different users) vs 300,825 different-target pairs; within-item standardized OLS outcome ~ cf_sim (SASRec) + hist_sim (history-text control), item-bootstrap CI.

- **Target effect is strong:** same-target reason cos 0.567 vs 0.495; facet Jaccard ≈ 0.36 vs 0.185.
- **User similarity adds ~nothing given the same target:** β_cf (reason) = +0.019 [0.011, 0.027] (all pairs); **−0.002 [−0.017, 0.012]** (informative reviews); raw review text +0.006 (n.s.). CF-quintile curve flat (0.565 → 0.568).
- CF similarity predicts reasoning *more* for different-target pairs (β 0.03–0.06) → user-level taste, not target-conditioned reasoning.
- **Conclusion:** reasoning is item-driven; "similar users + same item → similar reasons" is not supported beyond the item effect.

---

## 5. Better support retrieval

### 5a. Text HyDE (LLM predicts motive from history) — failed
Predicted rationales collapse to generic text (pairwise cos 0.71 vs 0.49 real); own-vs-random true rationale cos 0.52 vs 0.49; test target hit 0.7% (≈ random).

### 5b. Evaluation harness fix (`collab/eval_support.py`)
`reason_mean` was broken: a constant "centroid" support set (same 10 generic rationales for everyone) scored best (0.649). Fixed with mean-centred embeddings + discriminative **rank MRR / R@10** (pooled supports rank the query's own true rationale among all split rationales). Centroid → chance (0.0018), oracle 0.75.

### 5c. Latent HyDE (`collab/latent_hyde.py`, `collab/tune_latent.py`)
Contrastive MLP history → rationale embedding.

| Test (rank MRR, chance 0.0015) | value |
|---|---|
| SASRec user supports | 0.0056 |
| SVD item supports | 0.0050 |
| text HyDE | 0.0047 |
| latent HyDE (mpnet text) | 0.0093 |
| latent hybrid α=0.5 (mpnet) | 0.0078 (target hit 11.0%) |
| **tuned latent hybrid (gte-large text, τ 0.05, hidden 1024, α 0.5)** | **0.0092** (target hit 11.3%, R@10 1.62%) |
| direct prediction (no retrieval) | 0.0111 |

### 5d. Stage-2 supports (`collab/build_stage2_support.py` → `collab/stage2/`)
Cross-fitted predictor (5 folds by user) for train rows; full predictor for valid/test; calibrated balancing (ratio 0.315). Train target hit 11.25% vs test 11.26%. Harness: train OOF MRR 0.0100 vs test 0.0092.
For comparison, IntuRec's stage-1 intuition list (15) contains the test target for 10.95% of rows.

---

## 6. Model design evolution

1. **IntuRec-integrated** (supports added to IntuRecF2 + rationale aux loss) — abandoned: user wants collaborative reasoning as a separate branch; IntuRec code kept as reference (later restored).
2. **CollabRecF2, `emb` mode** (`model/CollabRecF2.py`, `data/CollabRecDataset.py`, `run_collabrec.py`): thought = W_r·his_query + attention(his_query → projected mpnet support vectors); loss = NTP + β·InfoNCE(rationale_head(thought), own rationale) vs 1,024 bank negatives.
3. **`text` mode** (user's idea: avoid mpnet→LLM projection domain gap): support rationale *tokens* → LLM `embed_tokens` → IntuRec's `InjectNet` (imported unmodified); aux = cosine InfoNCE(thought, convertseq2emb(own rationale)) vs 63 random train rationales.
4. **Motive supports** (`text/motive`): top-10 predicted motive prototypes (section 8) as support texts.
5. **Item-grounded supports** (`text/item`, `collab/build_item_supports.py`): `"<title the peer bought>": <peer rationale>` (same rows as text mode) — user agreed this is closer to a CF signal.

Rationale-loss reference (emb mode, 1,024 negatives, τ 0.05, valid): chance 6.93; constant vector 14.89; item-oracle (mean rationale of the true target's buyers) 9.67; standalone history predictor 6.66; **CollabRec epoch 1: 6.51**. InfoNCE at τ 0.05 heavily penalises hedged predictions; 22.6% of rows have a near-duplicate (cos > 0.8) among random negatives.

---

## 7. Evaluation protocol issues and decisions

- **IntuRec validation bug:** `run_inturec_f2.py` validation read `minibatch` (last *train* micro-batch) instead of `batch`. Its logged "validation NTP" (0.835 / 0.995 / 0.572) is a single train-batch loss. Same CollabRec checkpoint: train rows NTP 0.672 vs real valid 0.971. **Fixed on 2026-10-22 at the user's request** (7 lines); IntuRec rerun queued (section 11).
- Second IntuRec quirk (unfixed): test mode loads `.../standard/rs_*` but train saves `.../rs_*` (train-mode runs decode in memory, unaffected).
- CollabRec mirrors IntuRec's protocol: best-by-valid-NTP + every-epoch checkpoints; **test decoded with the last-epoch model**; logs both a real validation line and an "IntuRec-style (last train batch)" line.
- **Valid NTP is a poor checkpoint criterion:** text run, epoch 2 (best valid NTP 0.982) vs epoch 3 (1.104): R@10 10.30% vs **11.00%**, N@10 7.55 vs 8.08. Emb run: epoch 2 10.36% vs epoch 3 11.48%. Val NTP rises in epoch 3 but ranking still improves.
- Training budget limit: **3 epochs** (user constraint).
- Micro-batch: IntuRec original run 2, CollabRec Slurm runs 8 (effective batch 128 both). IntuRec rerun uses 8 for parity.
- Train-NTP epoch averages are equal or lower for CollabRec than IntuRec (e.g. epoch 3: 0.609 text vs 0.642), so CollabRec does not fit worse; the apparent "epoch-3 lag" in the IntuRec-style line is single-batch noise (8-row batches range 0.33–0.97).

---

## 8. Motive prototype corpus (user's idea: one shared rationale corpus for all splits → no leakage, no train/test mismatch)

### 8a. Clustering whole rationales (`collab/build_prototypes.py`, `collab/check_prototypes.py`)
Spherical k-means on mean-centred mpnet rationale embeddings, LLM-written canonical text per cluster.

| P | median size | <5 members | test coverage | leak gap (train − test own-prototype-text cos) |
|---|---|---|---|---|
| 128 | 368 | 0 | 0.691 | +0.005 |
| 512 | 83 | few | 0.752 | +0.008 |
| 1,000 | 42 | 4% | 0.774 | +0.008 |
| 2,000 | 18 | 12% | 0.790 | +0.013 |
| 4,000 | 8 | 31% | 0.804 | +0.024 |

P = 1,000 consistency: coherence median 0.54 (random 0.002), facet purity 0.92 (3.2× lift), 31% of rows within 0.05 of 2nd centroid, seed stability ARI 0.54 / NMI 0.82.
Problems found: 17.4% of train rationales are exact duplicates (generic templates across users, plus a data artifact); **29 items share a broken HTML title** (`<span class="a-size-medium…`) — target of 1.41% of test rows, affects IntuRec and CollabRec alike.

### 8b. Clustering method comparison (`collab/compare_clustering.py`, ~1,000 clusters; dedup per user, broken-title rows removed)

| space | method | noise | coherence | ambiguous | near-dup | ARI | cov test | leak gap |
|---|---|---|---|---|---|---|---|---|
| rationale/mpnet | k-means | 0 | 0.54 | 31% | 1% | 0.44 | 0.767 | 0.018 |
| rationale/mpnet | HDBSCAN | 81% | 0.80 | 42% | 1% | 0.73 | 0.693 | 0.000 |
| rationale/mpnet | Louvain | 0 | 0.58 | 36% | 10% | 0.63 | 0.775 | 0.013 |
| phrase/mpnet | k-means | 0 | 0.55 | 17% | 0% | 0.45 | 0.848 | 0.003 |
| phrase/mpnet | Louvain | 0 | 0.57 | 18% | 2% | 0.70 | 0.866 | 0.004 |
| phrase/qwen3 | k-means | 0 | 0.60 | 20% | 1% | 0.50 | 0.887 | 0.004 |
| **phrase/qwen3** | **Louvain** | 0 | 0.64 | 23% | 10% | **0.70** | 0.903 | **0.003** |

Conclusions: motive *phrases* (1–2 per rationale) fix the mixed-motive geometry (ambiguity halves, leak gap ÷4–5); kNN-graph Louvain is far more stable than k-means; HDBSCAN unusable as a full corpus; Qwen3-Embedding groups by meaning, mpnet by shared words.

### 8c. Final motive corpus (`collab/build_motive_corpus.py` → `collab/motive_corpus/`)
996 Louvain clusters → 911 after merging near-duplicates (cos > 0.9) → 893 after folding < 5 rows → **859 prototypes** after removing 34 post-purchase complaint clusters (user decision; "replace damaged copy" clusters kept via `--force-keep`). 1.2% of train phrase occurrences removed. Test rows: 97.4% have 2 motives, 0.4% none. Own-phrase vs prototype text cos 0.827 / 0.823 (train / test), **leak gap 0.0034**. Largest prototype 5.2% (generic nostalgia / curated collection head).

### 8d. Motive retriever (`collab/train_motive_retriever.py`) — history → 1–2 of 859 prototypes

Features: SASRec emb + gte-large history text + item motive profile (why people buy the history items) + user's earlier motives.

| Test | R@1 | R@5 | R@10 | R@20 | MRR | Soft@5 |
|---|---|---|---|---|---|---|
| popularity | 6.1 | 24.8 | 38.4 | 47.6 | 0.255 | 0.492 |
| user past only | 6.8 | 24.0 | 37.2 | 47.4 | 0.259 | 0.493 |
| item profile only | 8.3 | 27.7 | 36.2 | 45.0 | 0.294 | 0.520 |
| **MLP classifier (cls)** | **9.1** | **31.2** | **42.3** | **52.0** | **0.323** | **0.547** |
| InfoNCE text heads (frozen / proj / proj+id, ± prior bias) | — | 28.5–30.8 | — | 42.8–51.5 | 0.308–0.323 | 0.507–0.546 |

All variants plateau at valid MRR ≈ 0.30–0.31 (feature-limited). Text (InfoNCE) heads need a log-prior bias to recover popularity and still do not beat the classifier. Motive is item-driven (item profile strongest; user past ≈ popularity).
Cross-fitted (5 folds by user) train scores: OOF MRR 0.316 vs valid 0.310 / test 0.323; top-10 contains a true motive for 59.8 / 58.7 / 61.9% (train / valid / test). Supports in `collab/motive_corpus/stage2/`.

---

## 9. CollabRec results (3 epochs, β 0.1, k 10, test decoded with last epoch unless noted)

| Run (job) | Supports | R@1 | R@5 | R@10 | N@5 | N@10 |
|---|---|---|---|---|---|---|
| emb, **epoch 2** (best valid NTP; RTX 3090, micro 4) | peer rationales, mpnet + projection | 5.20 | 8.38 | 10.36 | 6.87 | 7.51 |
| emb (335192) | peer rationales, mpnet + projection | 5.62 | 9.16 | **11.48** | 7.46 | **8.20** |
| text (335212) | peer rationale text, InjectNet | 5.73 | **9.19** | 11.00 | **7.50** | 8.08 |
| text, epoch 2 decode (335282) | same checkpoint, epoch 2 | 5.36 | 8.38 | 10.30 | 6.93 | 7.55 |
| motive (335237) | top-10 predicted motive prototypes | 5.57 | 9.00 | 11.06 | 7.35 | 8.02 |
| item-grounded (335328) | `"title": peer rationale` | **5.78** | 9.00 | 11.13 | 7.47 | 8.15 |
| *IntuRec original* | *15 intuition titles* | *5.83* | *9.52* | *12.08* | *7.75* | *8.58* |

Observations:
- Four very different support designs all land at **11.0–11.5% R@10** → the single injected thought token appears to be largely ignored; predictions come mostly from the history prompt.
- Exposing peer purchase titles (11% contain the true target) did not help (11.00 → 11.13).
- Motive supports: the rationale aux loss stayed exactly at chance for all 3 epochs (4.158 vs ln 64 = 4.159) — generic prototype texts give the alignment nothing to latch on; peer-rationale supports reach valid 3.79.
- Valid NTP per epoch (real): ~1.09 / 0.98 / 1.10 for all variants; train NTP 1.49 / 0.88 / 0.61.

---

## 10. Oracle experiment (does reasoning help at all?) — β = 0 for all (own rationale as input would make the aux loss trivial)

| ID | Thought input | Job | Status |
|---|---|---|---|
| O0 | none (history only, `SUPPORT_K=0`) | 335336 → **335415** | **done** (first attempt failed: disk quota) |
| O1 | row's own rationale text (k=1) | 335337 → **335636** | preempted after epoch 1 (valid NTP 1.065 vs O0 1.093); resubmitted |
| O2 | row's own 1–2 motive prototype texts (k=2) | 335338 → 335416 → **335637** | disk quota, then preempted; resubmitted |

**O0 result (2026-10-22):** R@1 5.78, R@5 9.13, **R@10 11.00**, N@5 7.49, N@10 8.09; valid NTP 1.093 / 0.984 / 1.088.
→ **No-reasoning CollabRec matches every support variant (11.0–11.5% R@10).** The retrieved supports, in any form tried so far, add nothing measurable over the history alone.

Interpretation plan: O1 ≫ O0 but O2 ≈ O0 → gain is target leakage, not motive; O2 > O0 → motive information helps (O2 − O0 = ceiling for motive prediction); CollabRec runs vs O0 → value of retrieved supports. Planned (not run): O3 own facets, O4 same-target peer rationale (collaborative ceiling), C1 random rationale control; cheap non-LLM probe of all conditions.

### 10b. IntuRec stage 1 (F1) as a no-reasoning reference (decoded on lucas L40S, 2026-10-22)
Saved F1 checkpoint decoded with the same constrained top-10 beam search: R@1 4.92, R@5 8.45, **R@10 10.28**, N@10 7.33 (reproduces the original log, 10.26). `run_inturec_f1.py` uses HF Trainer with `load_best_model_at_end=True` + early stopping → the saved model is the **best-valid-loss epoch (≈ epoch 2)**. It matches every CollabRec epoch-2 decode (10.30 / 10.36), so F1 ≈ O0 once the epoch is matched. Ladder: plain LLM rec ≈ 10.3 (epoch 2) / 11.0 (epoch 3) → IntuRec 12.08 (epoch 3, rerun pending).

Run settings (all: Qwen2.5-1.5B full FT bf16, lr 3e-4, warmup 20 + linear decay, 3 epochs, effective batch 128, constrained top-10 beam):

| Run | Train GPU / framework | Micro | Decoded epoch | Decode GPU | R@10 |
|---|---|---|---|---|---|
| IntuRec F1 | not logged / HF Trainer (dynamic padding) | 12 | 2 (best valid loss 0.990; 1.171/0.990/1.216) | L40S | 10.28 |
| CollabRec text, ep-2 decode | A100-40GB / custom | 8 | 2 | L40S | 10.30 |
| CollabRec emb, ep-2 decode | RTX 3090 / custom | 4 | 2 | RTX 3090 | 10.36 |
| CollabRec O0 | L40S / custom | 8 | 3 | L40S | 11.00 |
| CollabRec text | A100-40GB / custom | 8 | 3 | A100 | 11.00 |
| CollabRec emb | A40 / custom | 8 | 3 | A40 | 11.48 |
| IntuRec F2 original | not logged / custom | 2 | 3 | same | 12.08 |
| IntuRec F2 rerun (335355) | A40 / custom | 8 | 3 | A40 | **11.55** |

Effects: epoch 2 → 3 = +0.7 to +1.1 R@10 (dominant); GPU / micro-batch / framework: epoch-2 runs on three setups all 10.28–10.36 (negligible); epoch-3 run-to-run spread ±0.25.

**IntuRec rerun result (335355, A40, micro 8, fixed validation; 2026-10-23):** R@1 5.72, R@5 9.36, **R@10 11.55**, N@5 7.60, **N@10 8.31**. The original 12.08 does not reproduce (−0.53). Under matched settings IntuRec's gain over O0 is only +0.55 R@10 / +0.22 N@10, and CollabRec emb (11.48 / 8.20) is level with it. Single seeds; seed spread (±0.25–0.5 R@10) is as large as these gaps → key configs (O0, IntuRec, best CollabRec) need 2–3 seeds before any claim.

**Overwrite incident (2026-10-23):** the rerun wrote its test predictions to the same `result_data/CDs_and_Vinyl/IntuRecF2/.../rs_1/meta.json` as the original Oct-15 run, so the original per-row predictions are lost (no filesystem snapshots); the original metrics survive in its log. No original IntuRec F2 checkpoint existed in the repo (it had gone to the default `your output ckpt path`, now absent). Fix: `submit_inturec_f2.sbatch` now writes every run to `checkpoints/runs/$RUN_ID/` and `result_data/runs/$RUN_ID/` (default `RUN_ID=micro${MICRO}_job${SLURM_JOB_ID}`); the rerun's files were moved to `runs/micro8_job335355/` and its log renamed `inturec-f2-micro8_job335355-...log`.

### 10c. Contrastive support loss (IntuRec intuition-loss analog), user's two anchors
`softplus(thought_rand·anchor − thought_ret·anchor)`, γ = 1.0, on item-grounded supports (β 0.1), random supports in the same format; anchor (training only) = target title (pooled like IntuRec's tgt_emb) or the row's ground-truth motive prototypes (mean of 2 pooled texts). Smoke test: starts at ln 2 = 0.693 like IntuRec. Baseline: item-grounded run without the loss (R@10 11.13).
Jobs: **335655** (anchor = title), **335656** (anchor = motive).

---

## 11. Pending jobs (as of 2026-10-22 evening)

| Job | What | Log |
|---|---|---|
| 335636 | O1 own rationale (resubmitted after preemption) | `log/CDs_and_Vinyl/collabrec-text-oracle_own-k_1-beta_0-rs_1.log` |
| 335355 | IntuRec rerun: **done**, R@10 11.55 / N@10 8.31 | `log/CDs_and_Vinyl/inturec-f2-micro8_job335355-lr_3e-4-intunum_15-alpha_1.0-train-standard-rs_1.log` |
| 335637 | O2 own motives (resubmitted after preemption) | `log/CDs_and_Vinyl/collabrec-text-oracle_own_motive-k_2-beta_0-rs_1.log` |

**IntuRec rerun, real validation (first time measured):** NTP 1.095 / 0.984 / 1.098 (epochs 1/2/3), intuition loss 0.024 / 0.016 / 0.018; train NTP 1.516 / 0.890 / 0.618 — **identical to CollabRec** (valid ≈ 1.09 / 0.98 / 1.10). IntuRec also "overfits" in epoch 3 by valid NTP; the old log hid it. Test metrics pending.

---

**2026-10-23: all running CollabRec jobs cancelled at the user's request** — O1 (335636, after epoch 1), O2 (335637, during epoch 1), support-loss title (335655) and motive (335656) runs (both during epoch 1, no checkpoint). O1/O2/O0-style oracle results and the support-loss comparison are therefore still open. Partial O1 epoch-1 checkpoint kept as `CollabRec_oracle/text-oracle_own_k1_beta0.0_rs1_lr0.0003_job335636-cancelled-epoch1`.

**2026-10-23: all checkpoints deleted at the user's request except the original IntuRec stage-1 model** (`checkpoints/CDs_and_Vinyl/IntuRec_original/IntuRecF1` on scratch, symlinked from the old F1 path). All CollabRec / CollabRec_oracle checkpoints and the IntuRec stage-2 rerun checkpoint are gone; test predictions in `result_data/` and all logs are kept. Re-decoding any past run requires retraining it.

## 12. Open ideas / next steps

1. **Make the model use its supports** (if O0 ≈ 11%): contrastive support loss (IntuRec intuition-loss analog: thought from retrieved vs random supports) with **channel-matched anchors** — item-grounded channel ↔ target title; motive channel ↔ own motive prototypes (user: a title anchor does not fit motive supports).
2. **Two thought tokens** (reason_step = 2): item-grounded peer channel (what) + motive channel (why); depends on idea 1's outcome.
3. Supports as prompt tokens instead of one pooled vector.
4. Checkpoint selection by valid Recall@10 on a 1k subset (NTP is a poor proxy); keep last-epoch reporting for IntuRec parity.
5. Learnable temperature / false-negative filtering / multi-positive targets for the rationale InfoNCE.
6. Item-aware motive prototypes (keep top items per prototype) if pure motives lose the CF item signal.
7. Regenerate rationales with a more specific prompt (era / format / occasion / trigger) if finer motives are needed.
8. Forked session: decompose textual rationales into **codewords** and train the model to generate them as reasoning (paper arXiv 2606.29712) — explored in a separate session.

---

## 13. Operational notes

- **Best-only checkpoints (since 2026-10-23, user request):** `run_collabrec.py` and `run_inturec_f2.py` now save only the best checkpoint (lowest average valid NTP), overwriting the previous best in the run folder; no `epoch_N/` copies (~3.1 GB per run instead of 9–12 GB). Test decoding after training still uses the in-memory **last-epoch** model (IntuRec protocol), so the saved checkpoint (best valid NTP, usually epoch 2) is not the model that produced `result/standard/meta.json`.

- **Per-run folders (since 2026-10-23, user's choice):** everything about a run lives in `/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints/<dataset>/<experiment>/<run>/` — `epoch_N/` checkpoints (+ best as symlinks), `result/<decode>/meta.json`, `train.log` (decode-only jobs: `test_<ckpt>_job<id>.log`), `slurm-<job>.out`. The repo's `checkpoints/` is a symlink to `/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints`; the repo's `log/`, `run_logs/`, `result_data/` are now empty (Slurm still writes `run_logs/` first; the sbatch scripts copy it into the run folder at the end). All earlier logs/results were moved there by `collab/migrate_logs_results.py` (74 items), including failed/preempted/cancelled attempts (`..._job<id>-failed-quota|-preempted|-cancelled`), the two original IntuRec F2 runs (`IntuRec_original/IntuRecF2_original_oct15[_balance0295]/train.log`) and pipeline logs (`collab_pipeline/logs/`). **Log paths quoted elsewhere in this file are pre-move.**

- **Storage layout (since 2026-10-23):** checkpoints in `/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints/<dataset>/<experiment>/<run_name>/` (epoch_N/ folders; the best-by-valid checkpoint is a symlink to its epoch folder, not a second copy); predictions in `result_data/<dataset>/<experiment>/<run_name>/standard/`; new logs in `log/<dataset>/<experiment>/<run_name>.log`. Experiments: `IntuRec_original` (IntuRecF1, IntuRecF2_<run_id>), `CollabRec`, `CollabRec_oracle`. Run names are built from the config + `_job<SLURM id>` (`run_collabrec.py --ckpt_root/--experiment/--run_name/--run_suffix`; IntuRec scripts got optional `--ckpt_dir/--result_dir`, defaults unchanged). `checkpoints/CDs_and_Vinyl/IntuRecF1/lr_0.0003` is a symlink to the scratch copy (used by `genf1midres.sh`). Finished runs that still used old paths are moved by `collab/migrate_run.sh` (copy, verify, delete, dedup).
- Scratch has its own per-user quota (~500 GB); it filled during migration on 2026-10-23 (no data lost: `mv` keeps sources it cannot copy); the user then freed space.

- Both sbatch scripts use `#SBATCH --requeue` (since 2026-10-22: O1/O2 were preempted once); a requeued job restarts from scratch and overwrites its log.
- Slurm: `submit_collabrec.sbatch` (vars `SUPPORT_MODE, SUPPORT_K, BETA, SUPPORT_DIR, RUN_TAG, TRAIN_OR_TEST, EVAL_CKPT, ORACLE, MICRO`), `submit_inturec_f2.sbatch`. QOS allows ~4 running jobs.
- Each run saves 4 checkpoints (~12 GB). **Home quota filled on 2026-10-22** (O0/O2 crashed). Finished-run checkpoints moved to `/common/scratch/users/m/mk.ngohuu.2024/IntuRec_checkpoints/` with symlinks back.
- vLLM Qwen3-8B: was at holiday:8000 → lexicon:8000; later started locally on the interactive node (`racer` env `vllm serve Qwen/Qwen3-8B`), stopped when not needed.
- IntuRec code is the reference: only CollabRec code is edited unless the user explicitly asks (exception: validation fix).
