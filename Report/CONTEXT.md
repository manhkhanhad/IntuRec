# MotiveRec — project context for a new session

Read this first, then [main.md](main.md) (the research plan) and the reports it links. Last updated 2026-11-13.

## What the project is
- **MotiveRec** grounds the latent `<|Thought|>` vector of an LLM recommender (Qwen2.5-1.5B) in **purchase
  motives**: motive mining → motive retrieval → motive integration.
- It has two integration variants:
  - **E** (embedding): `--support_mode emb`;
  - **T** (text): `--support_mode text`, which uses IntuRec's InjectNet as a "motive reader".
- **Code:** `run_collabrec.py` + `model/CollabRecF2.py` + `data/CollabRecDataset.py`. This is "CollabRec" with
  motive-prototype supports, **without** the collaborative InfoNCE (cq) of the `collab-refined-infonce` worktree.
  The β rationale-alignment InfoNCE (β = 0.1) **is** part of MotiveRec.
- **Motivation:** latent collapse of existing latent recommenders (LatentR³, IntuRec), measured by geometry.

## Where things are
- **Worktree:** `/common/home/users/m/mk.ngohuu.2024/research/IntuRec/.claude/worktrees/motiverec`, branch `motiverec`.
  - `data/CDs_and_Vinyl`, `data/Toys_and_Games` and `checkpoints` are symlinks to the main repo data and to
    `/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints` (`<ckpt>`).
- **Plan:** `Report/main.md`. Results: `Report/experiments/<NN_name>/report_*.md`. Running log:
  `collab/RESEARCH_LOG_motiverec.md`.
- **Pilot code:** `pilot/`.
  - `common.py`: paths, run ids, metrics, `collapse_stats`.
  - `pilot_a_motive_breakdown.py`
  - `pilot_bc.py probe|collapse`
  - `pilot_c_r3.py` + `submit_pilot_c.sbatch`: R³ interventions via R³'s own `experiment.py`.
  - Outputs go to `<ckpt>/<ds>/MotiveRec_pilot/`.
- **Motive data:** `data/<ds>/collab/`.
  - `rationales_facet_<split>_v5.jsonl`: Qwen3-8B rationales, phrases, facets.
  - `motive_corpus/`: `prototypes.json` (859 CDs / 881 Toys), `row_motives_<split>.json`, `retriever/cls/`.
  - `motive_corpus/stage2/`: predicted top-10 supports for the T variant.
  - `motive_corpus/stage2_emb/`: the E variant, built by `collab/build_motive_emb_support.py`.
- **Thought diagnostics for CollabRec / MotiveRec runs:**
  `.claude/worktrees/collab-refined-infonce/collab/latent_diagnostics.py` + `submit_latent_diag.sbatch`. They write
  `<run>/diagnostics/{thoughts_test.npz,meta_<cond>.json,summary.json}`.
- **Baselines:**
  - O0: `<ckpt>/<ds>/CollabRec_oracle/text-micro2_k0_beta0_rs1_lr0.0003_job351003|350999`
  - IntuRec: `<ckpt>/<ds>/IntuRec_original/IntuRecF2_micro2_job349506|348597`
  - LatentR³ original: `/common/home/users/m/mk.ngohuu.2024/research/R3_ori/output_dir/<ds>/{latent,nomul_attn_lr_5e-5}`
  - R³ code: `/common/home/users/m/mk.ngohuu.2024/research/R3/src/prototype_grounding/experiment.py`

## Rules (from the author; follow them)
- **Environment:** `eval "$(conda shell.bash hook)"; conda activate rrec` (torch 2.5.1). The login node has no GPU.
- **GPU jobs:** submit through the **JobManager** peer session (SendMessage), never `sbatch` directly. Don't
  resubmit failed jobs (other than preemption) without asking. Toys needs ≥ 48 GB GPUs.
- **Code edits:** only CollabRec / MotiveRec code. Never modify IntuRec's own files, nor R³ code; wrappers import them.
- **Metrics:** report as values (0.1094), not %. Always give **HR@1, HR@5, HR@10, NDCG@1, NDCG@5 and NDCG@10**,
  never HR@10 alone.
- **Oracle motives:** development ceilings only, never paper evidence.
- **Paper scope:**
  - The motivation uses geometry only (effective rank, PC1, pair cosine). Do not use intervention results ("the
    decoder ignores the thought"); the author will address them in a separate paper.
  - Claim "motive grounding prevents collapse", not "the prediction relies on motives".
- **Novelty:** claim motive grounding. Describe the T-variant reader generically (two-level attention pooling) and
  cite IntuRec once; no hiding of attribution.
- **Writing:** the author does not want a paper yet; keep the research plan + experiment reports. `paper/main.tex`
  is an earlier LaTeX draft and is superseded by `Report/main.md`.
- **New experiments:** add a folder or report under `Report/experiments/`, link it from the table in `main.md`, and
  append to `collab/RESEARCH_LOG_motiverec.md`.

## Current state (2026-11-13)
- **Done:**
  - E1 collapse geometry, plus interventions (dev only);
  - E2 facet breakdown and thought probe;
  - E3 MotiveRec-T geometry (no collapse: pair cosine 0.54–0.69).
- **Goal (author):** make MotiveRec beat **IntuRec**. Below-O0 is IntuRec's problem too, so not the target. See E5.
- **Main open problem:** MotiveRec-T accuracy is below IntuRec and O0.
  - CDs HR@10: 0.1174 vs 0.1202 / 0.1178.
  - Toys HR@10: 0.1072 vs 0.1112 / 0.1112.
- **Running, via JobManager:**
  - 364572 / 364573: Toys R³ shuffle interventions. When done, run `python pilot/pilot_bc.py collapse Toys_and_Games`
    and update E1b.
  - 364681: CDs MotiveRec-T diagnostics decodes. Update E1b notes.
  - 364727 / 364728: MotiveRec-E with β 0.1 — deleted 2026-11-14 (wrong loss); NTP-only reruns submitted. Fill in E4 from those.
- **Next ideas:**
  - β = 0 run (motive input vs alignment);
  - k sweep;
  - per-facet accuracy of MotiveRec;
  - probe on MotiveRec thoughts;
  - history-query geometry of R³ / IntuRec themselves.
