# MotiveRec — research plan

*Grounding latent reasoning in purchase motives for LLM-based sequential recommendation.*
Status: research plan (not a paper draft). Last updated 2026-11-13. Results live in [experiments/](experiments/); this
file only links to them. Project context for new sessions: [CONTEXT.md](CONTEXT.md).

---

## Abstract
Latent reasoning lets an LLM recommender "think" in a continuous vector before it names the next item, avoiding the
cost and noise of generated text rationales. Trained only through the recommendation loss, this latent thought
**collapses**: across two Amazon domains the thoughts of different users have an effective rank of 1–9, the first
principal direction carries up to 99.8% of their variance, and two random users' thoughts have cosine 0.94–1.00
([E1](experiments/01_pilot_latent_collapse/report_geometry.md)). A thought that is nearly identical for every user
cannot carry user-specific reasoning. We hypothesise that the cause is that nothing tells the latent *what* to reason
about. We propose **MotiveRec**, which grounds the latent thought in the user's **purchase motive**, the reason a
person buys. MotiveRec:
1. mines an interpretable vocabulary of motive prototypes from review-grounded purchase rationales;
2. retrieves the motives likely to drive the next purchase from the interaction history alone;
3. builds the latent thought from the retrieved motives, using either their sentence embeddings or their text read
   in the LLM's own token space, and aligns it with the user's mined rationale.

Motive grounding removes the collapse: random-pair cosine 0.54–0.69, effective rank 9–16
([E3](experiments/03_motiverec_text/report_geometry.md)). Whether it also improves accuracy is still open
([E3 accuracy](experiments/03_motiverec_text/report_accuracy.md), [E4](experiments/04_motiverec_emb/report_setup.md)).

## 1. Introduction
**Context.** LLM-based sequential recommenders cast next-item prediction as generating the title of the next item
from a textual history (TALLRec, BIGRec). Explicit chain-of-thought helps but is slow and unreliable. Latent
reasoning writes one continuous vector, the *thought*, into the prompt before the answer:
- **LatentR³** pools the LLM's hidden states into a thought and refines it with RL.
- **IntuRec** builds it from retrieved candidate items ("intuitions").

**Problem: latent collapse.** The thought is supervised only by the next-item loss. For both methods on both
datasets, the test thoughts lie almost on one direction. On Toys, R³ gives every user the same vector. RL makes the
collapse worse on CDs (effective rank 9.01 → 1.55). Meanwhile the backbone's hidden state at the thought position
is diverse (rank 132–138), so the user information is available but is not kept in the latent. A latent that is the
same for everyone acts as a learned prompt, not as reasoning about the user.

**Hypothesis.** The latent has no *content* to organise around. A content-free latent (R³) collapses. So does a
latent built from item-level content, i.e. *what* to buy (IntuRec). We give the latent a different kind of content:
**why** the user buys. Purchase motives, such as completing a collection, buying a gift, or loyalty to an artist or
franchise, are interpretable, shared across users and items, and differ between users with similar histories.

**Positioning (novelty claim).**
- R³: no content in the latent.
- IntuRec: item candidates (*what*).
- MotiveRec: mined, named, interpretable motives (*why*).

The novelty is motive grounding: mining, retrieval, and integration plus alignment. The text-variant reader adopts a
generic two-level attention pooling. It is described in our own terms and cited once to IntuRec as prior design.
It is not claimed as a contribution.

**Planned contributions.**
1. Identify and measure latent collapse in latent-reasoning LLM recommenders (geometry).
2. MotiveRec, a target-free pipeline: motive mining → motive retrieval → motive integration (embedding and text
   variants) with motive-aligned training.
3. Motive vocabularies for two Amazon domains (859 / 881 named prototypes).
4. Experiments showing that collapse is removed, plus the accuracy effect (pending).

**Scope decisions (author, 2026-11-13).**
- The motivation uses **geometry only**. Intervention results ("does the decoder use the thought?") are out of
  scope here; they are kept as development notes in
  [E1 interventions](experiments/01_pilot_latent_collapse/report_interventions.md). Write claims as "motive
  grounding prevents collapse", never "the prediction relies on motives".
- Oracle-motive experiments are development ceilings only, never evidence.
- Scoring below the original R³ numbers is acceptable to the author. IntuRec, which uses the same backbone and slot,
  is the baseline reviewers will compare against.

## 2. Method

### 2.0 Backbone and latent slot
- **Input:** a user's chronological history H_u = (i_1 … i_n) of item titles, written into a prompt followed by a
  reserved `<|Thought|>` token. The task is to generate the title of i_{n+1}.
- **History query:** a first LLM pass gives the hidden state at the thought position, h_u ∈ R^d (Qwen2.5-1.5B,
  d = 1536).
- **Thought:** the vector t_u replaces the thought token's input embedding. The LLM then decodes the next title
  with catalogue-constrained beam search (10 beams).
- MotiveRec only changes how t_u is formed.

### 2.1 Motive mining (offline, training data only)
1. **Rationales.** For every training interaction, Qwen3-8B reads the history, the purchased item and the user's
   review. It writes 1–2 motive phrases (2–6 words) and a first-person reason ("I bought this to …"). Reviews ground
   the rationale in what the buyer said. Output: `rationales_facet_<split>_v5.jsonl`.
2. **Clustering.**
   - Distinct phrases are embedded with Qwen3-Embedding, using a purchase-motive instruction.
   - A similarity graph over the phrases is clustered with Louvain (about 1,000 communities).
   - Near-duplicate communities are merged, and communities with too few training rows are folded into their
     nearest neighbour.
3. **Naming.** An LLM names each community from its top phrases: a canonical motive, a first-person reason and a
   label. Communities that are post-purchase complaints are removed (30 on Toys).
   - Result: vocabulary M = {m_1 … m_K}, K = 859 (CDs) / 881 (Toys).
   - Prototype text: "*motive*: *reason*".
4. **Labels.** Every phrase is assigned to its nearest prototype, so each interaction gets 0–2 motives (99.5% have
   at least one). Labels are used only as retriever targets, never as model inputs.

Code: `collab/gen_rationales.py`, `collab/compare_clustering.py`, `collab/build_motive_corpus.py`.

### 2.2 Motive retrieval (target-free)
- **Features** (history only):
  - SASRec embedding of the item sequence;
  - gte-large embedding of the history titles;
  - item motive profile: the mean motive distribution of all training purchases of the history items;
  - the user's own earlier motive distribution.
- **Model:** an MLP that scores all K prototypes, trained with multi-positive softmax cross-entropy and early-stopped
  on validation MRR.
- **Leakage control:** recommender training rows get motives from 5-fold cross-fitting grouped by user.
- **Output:** the top-k prototypes R_u, k = 10 (`motive_corpus/stage2/`).
- **Quality** (test, Recall@10 / MRR): CDs 0.423 / 0.323 against a popularity prior of 0.384 / 0.255; Toys
  0.318 / 0.290 against 0.257 / 0.249.

Code: `collab/train_motive_retriever.py`, step 4 of `collab/submit_collab_supports.sbatch`.

### 2.3 Motive integration
The history decides which retrieved motives matter. There are two variants.

**MotiveRec-E (embedding).**
- Each motive is embedded once by a frozen encoder in the rationale space: e_k = Enc(x_k), using
  all-mpnet-base-v2, centred on the train-rationale mean.
- Motives are projected into the LLM space: s_j = W_p e_{r_j}.
- Thought: t_u = W_r h_u + Σ_j α_j W_v s_j, with α = softmax_j((W_q h_u)ᵀ W_k s_j / √d).
- The residual path keeps the history signal, and α is an interpretable distribution over motives.

Code: `SupportAttention` in `model/CollabRecF2.py`, `--support_mode emb`. Data: `motive_corpus/stage2_emb/`, built by
`collab/build_motive_emb_support.py`.

**MotiveRec-T (text, "motive reader").**
- Motive text is tokenised and embedded with the LLM's own input embedding table, giving X_j ∈ R^{L×d}.
- **Level 1:** a rotary-position attention pool, whose query is the last token, summarises each motive into z_j.
- **Level 2:** the history query attends over {z_j}: t_u = softmax_j((W2_q h_u)ᵀ W2_k z_j / √d) W2_v z_j.
- Motives live in the token space the decoder reads. The only new parameters are the reader's projections.
- Cite IntuRec once for the two-level pooling.

Code: `InjectNet` used by `CollabRecF2`, `--support_mode text`, supports `motive_corpus/stage2/`.

### 2.4 Training objective (decided 2026-11-14)
- **Objective: L = L_NTP only.** MotiveRec uses the next-item loss alone.
- **Motives per row: 3** (decided 2026-11-15, from E9: 3 ≥ 10 for both v5 and v6).
- CollabRec's other terms are all off:
  - **L_align** (rationale alignment, β): the thought must pick the row's own reason sentence among 63 random ones.
    CollabRec default β = 0.1 → MotiveRec β = 0.
  - **L_support** (IntuRec-style contrastive support loss, γ): γ = 0.
  - **L_collab** (collaborative reasoning loss of `collab-refined-infonce`: thought aligned with same-user /
    same-target / shared-motive rows from a queue): removed.
- Note: every MotiveRec-T run before 2026-11-14 (345291, 365120, 365345, 365351, 365470, 365477, …) used
  β = 0.1 (L_align on). They are not the final MotiveRec objective.

## 3. Experiment plan
Datasets: Amazon CDs & Vinyl (train / test 49,251 / 6,157) and Toys & Games (53,898 / 6,738). Splits are identical
for all models. Metrics: **always report HR@{1,5,10} and NDCG@{1,5,10}** on the full test set, as values (0.1094), not %.

| ID | Question | Status | Report |
|---|---|---|---|
| E1 | Do latent-reasoning recommenders collapse? (motivation) | done | [geometry](experiments/01_pilot_latent_collapse/report_geometry.md) · [interventions, dev only](experiments/01_pilot_latent_collapse/report_interventions.md) |
| E2 | Is the motive a meaningful axis? (optional support) | done | [facet breakdown](experiments/02_pilot_motive_analysis/report_facet_breakdown.md) · [thought probe](experiments/02_pilot_motive_analysis/report_thought_probe.md) |
| E3 | MotiveRec-T: accuracy and geometry | geometry done; accuracy below IntuRec | [accuracy](experiments/03_motiverec_text/report_accuracy.md) · [geometry](experiments/03_motiverec_text/report_geometry.md) |
| E4 | MotiveRec-E: accuracy and geometry | done (v5 motives): CDs below T, Toys ≈ T; both below IntuRec | [setup and results](experiments/04_motiverec_emb/report_setup.md) |
| E5 | Improve MotiveRec to beat IntuRec (goal) | in progress | [gap + rerank](experiments/05_improve_vs_inturec/report_gap_and_rerank.md) · [curriculum](experiments/05_improve_vs_inturec/report_curriculum.md) · [joint training](experiments/05_improve_vs_inturec/report_joint_training.md) |
| E8 | Improve module 1, motive mining | diagnostics done; v6 mined: predictable + item-linked; aligner + IRLLRec dropped (cls retriever best); β 0.1 MotiveRec-T/E runs invalidated (wrong loss, deleted); NTP-only (β 0) reruns submitted 2026-11-14 | [diagnostics](experiments/06_motive_mining/report_mining_diagnostics.md) · [aligner on v5](experiments/06_motive_mining/report_aligner_v5.md) · [relabel on v5](experiments/06_motive_mining/report_relabel_v5.md) · [by history length](experiments/06_motive_mining/report_history_length.md) · [v6 mining](experiments/06_motive_mining/report_mining_v6.md) · [aligner v2](experiments/06_motive_mining/report_aligner2_v6.md) · [IRLLRec](experiments/06_motive_mining/report_irllrec_v6.md) |
| E9 | MotiveRec with NTP-only objective (8 reruns) | done: v5 = v6; 3 ≥ 10 motives; best (v6/v5, 3, text) ≈ O0, marginally > IntuRec on Toys; CDs E < IntuRec | [results](experiments/07_ntp_only/report_ntp_only.md) |
| E5b | Main table vs. baselines | planned | — |
| E6 | Ablations | planned | — |
| E7 | Analysis and case studies | planned | — |

**E5, main table.**
- Baselines:
  - SASRec, GRU4Rec;
  - BIGRec / D3;
  - O0, the backbone with an empty thought slot and the same training;
  - LatentR³ (SFT, RL), from the original code;
  - IntuRec.
- Models: MotiveRec-E and MotiveRec-T.
- Target: beat IntuRec and O0 on both datasets.

**E6, ablations.**
1. k ∈ {1, 3, 5, 10, 20} motives.
2. β = 0, which separates the effect of the motive input from that of alignment on the geometry.
3. Random motives instead of retrieved ones.
4. E vs. T.
5. Retriever variants: popularity, item profile, user past, full MLP. This links retriever quality to accuracy.

**E7, analysis.**
- Geometry table for all models (E1 measures applied to MotiveRec).
- Thought neighbourhoods: same user, same motive.
- Per-motive-type accuracy, using the facet strata from E2: do the hard motive types improve?
- Case studies with attention weights over the retrieved motives.

**Goal of the improvement track (author, 2026-11-13): MotiveRec must beat IntuRec.** Being below O0 is shared with IntuRec, so O0 is not the target.

**Improvement track (parallel).** MotiveRec-T is below IntuRec, see [E3](experiments/03_motiverec_text/report_accuracy.md).
Candidates:
- the embedding variant (E4, running);
- tuning k and β;
- a better retriever.

Record each attempt as a new report under `experiments/`, and add a row to the table above.
