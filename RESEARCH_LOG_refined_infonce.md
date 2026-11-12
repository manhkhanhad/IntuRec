# Collaborative InfoNCE on the motive-refined latent

To be merged into `collab/RESEARCH_LOG.md` (main checkout) once this worktree's results are in.

### 2026-11-01 — idea (user's idea)
Motivation, from R3's collaborative-reasoning log: R3 Attempt 10 put user/target InfoNCE and a correctness-gated queue on R3's latent. That latent is collapsed (RL bank random-pair cos 0.99, effective rank 2.6; mean or shuffled latents ≈ raw), so the loss reorganised the latent without any ranking gain. CollabRec's refined thought (InjectNet over motive prototypes) is content-sensitive (O2 +5.5 R@10 on Toys) and diversified by the motives, so the same collaborative loss may now act on a latent the decoder actually uses.

Design (branch `worktree-collab-refined-infonce`; `model/CollabRecF2.py`, `data/CollabRecDataset.py`, `run_collabrec.py --collab_*`, `submit_collabrec.sbatch COLLAB_*`; CPU test `tests/test_collab_queue.py`):
- anchor and queue entries = the refined thought (the InjectNet output written into `<|Thought|>`);
- 10k FIFO queue, training-only, not saved; admission gate: ≥ 50% of target tokens correct (R3's best Toys gate);
- three projection heads (LayerNorm + Linear → 256): same user, same target, shared motive prototype (train `row_motives`);
- multi-positive InfoNCE, τ 0.1, weight 0.1 each, ≤ 16 positives, 256 negatives; negatives share no user, target or motive with the anchor;
- valid NTP excludes the collaborative loss, so it stays comparable to the baselines.

| Job | Run | Compare with |
|---|---|---|
| 348439 | `Toys_and_Games/CollabRec/text-motive-cq0.1_user+target+motive_k10_beta0.1…` | predicted top-10 motives, no InfoNCE (345291: R@10 10.72 / N@10 7.05) |
| 348440 | `Toys_and_Games/CollabRec_oracle/text-oracle_own_motive-cq0.1_user+target+motive_k2_beta0…` | O2 (343601: 16.22 / 10.00) |

Submitted via the JobManager session (48 GB GPUs, lunar excluded).

### 2026-11-01 — IntuRec reproducibility at micro-batch 2 (CDs)
Question: does the original IntuRec stage-2 result (12.08 R@10, micro-batch 2, buggy validation log) reproduce, or was it a lucky run? The validation bug cannot affect test metrics: test is decoded with the in-memory last-epoch model, and the buggy loop runs under eval/no_grad. Micro-8 reruns: 11.55 (335355), 11.19 (335690).
Setup: main checkout's `submit_inturec_f2.sbatch` (fixed validation, our protocol), `MICRO=2`, two identical runs (script has no seed option, so seed 0 like the original; the pair measures GPU nondeterminism). Caveat: eval batch = micro = 2 (original default 4).

| Job | Run |
|---|---|
| 348534 | `CDs_and_Vinyl/IntuRec_original/IntuRecF2_micro2_job348534` |
| 348535 | `CDs_and_Vinyl/IntuRec_original/IntuRecF2_micro2_job348535` |

Reading: both ≈ 12.0 → micro-batch 2 helps; both 11.2–11.6 → 12.08 was run-to-run luck.
