# E9 — MotiveRec with the NTP-only objective (8 reruns, 2026-11-14/15)

**Why these runs exist.** On 2026-11-14 the user fixed the MotiveRec objective to **L = L_NTP only**. That turns off
L_align (rationale alignment, β → 0) and L_support (γ = 0), and removes L_collab. Every earlier MotiveRec-T/E run had
used β = 0.1. Those runs were deleted, and the runs below replace them.

**Common setup.**
- Qwen2.5-1.5B, 3 epochs, lr 3e-4, effective batch 128 (micro 8; f uses micro 4).
- BETA = 0. CODE_DIR = motiverec worktree.
- Test decode uses the **epoch-3 in-memory model**. Validation NTP was best at epoch 2 in every run.
- Single seed per run. The binomial standard error of HR@10 is about 0.004 on Toys (6,738 rows) and CDs (6,157 rows).

## Toys & Games (test)

| Run | Job | Motives | Count | Integration | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| b | 365542 | v6, cls retriever | 3 | text, InjectNet | 0.0421 | 0.0837 | **0.1128** | 0.0421 | 0.0637 | **0.0730** |
| e | 365543 | v5 | 3 | text, InjectNet | **0.0423** | 0.0844 | 0.1122 | **0.0423** | 0.0639 | 0.0728 |
| d | 365545 | v5 | 10 | text, InjectNet | 0.0404 | 0.0844 | 0.1101 | 0.0404 | 0.0636 | 0.0719 |
| a | 365544 | v6, cls retriever | 10 | text, InjectNet | 0.0417 | 0.0819 | 0.1097 | 0.0417 | 0.0624 | 0.0713 |
| f | 365549 | v6, cls retriever | 10 | text, LLM mean-pool + LN | 0.0398 | 0.0839 | 0.1095 | 0.0398 | 0.0625 | 0.0708 |
| g | 365547 | v5 | 10 | emb (MotiveRec-E) | 0.0402 | 0.0839 | 0.1094 | 0.0402 | 0.0628 | 0.0710 |
| c | 365546 | v6, aligner | 3 | text, InjectNet | 0.0404 | 0.0810 | 0.1082 | 0.0404 | 0.0616 | 0.0703 |
| IntuRec | — | — | — | — | 0.0420 | 0.0833 | 0.1112 | 0.0420 | 0.0632 | 0.0721 |
| O0 (no motives) | 350999 | — | 0 | — | 0.0417 | **0.0867** | 0.1112 | — | **0.0650** | 0.0729 |

Validation NTP by epoch:

| Run | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---|---|---|
| b | 0.6347 | 0.5554 | 0.5824 |
| e | 0.6373 | 0.5522 | 0.5783 |
| d | 0.6394 | 0.5542 | 0.5802 |
| a | 0.6347 | 0.5547 | 0.5796 |
| f | 0.6425 | 0.5628 | 0.5971 |
| g | 0.6425 | 0.5568 | 0.5828 |
| c | 0.6403 | 0.5555 | 0.5823 |

## CDs & Vinyl (test)

| Run | Job | Setup | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|---|
| h | 365548 | MotiveRec-E, v5, 10 motives | 0.0557 | 0.0901 | 0.1135 | 0.0557 | 0.0736 | 0.0810 |
| IntuRec | — | — | 0.0580 | **0.0981** | **0.1202** | 0.0580 | 0.0789 | 0.0860 |
| O0 | 351003 | no motives | **0.0604** | 0.0968 | 0.1178 | **0.0604** | **0.0796** | **0.0864** |

Validation NTP for h: 1.0959, 0.9837, 1.0997. Epoch 3 overfits strongly on CDs.

## Findings
1. **v5 = v6.**
   - At 3 motives, v6 − v5 = +0.0006 HR@10 and +0.0002 NDCG@10.
   - At 10 motives, v6 − v5 = −0.0004 HR@10 and −0.0006 NDCG@10.
   - E8b showed v6 mining is much better: history predictability is 8.2× popularity vs 1.03×, and the oracle
     next-item HR@10 is 0.181 vs 0.138. None of that reaches the recommender.
2. **3 motives ≥ 10 motives, for both versions.**
   - HR@10: +0.0021 (v5) and +0.0031 (v6). NDCG@10: +0.0009 and +0.0017.
   - The direction is consistent, but each gap is below one SE.
   - This matches the earlier β-0 k-sweep (`collab/RESEARCH_LOG.md`, 2026-11-12), which found no trend in k.
3. **Every motive configuration ≈ O0 (no motives).** The best runs (b, e) are marginally above IntuRec on all six
   metrics, but they are level with O0. On CDs, MotiveRec-E is clearly below IntuRec and O0: HR@10 −0.0067 and −0.0043.
4. **Integration choices do not help:**
   - aligner supports (c) < cls supports (b) on every metric (HR@10 −0.0046). The same held with β = 0.1.
   - Embedding mode (g) < text mode (d).
   - LLM mean pooling with LayerNorm (f) < InjectNet (a). Without the normalisation, f diverged; see E8 / log.
5. **Removing L_align changes almost nothing.** Compare b with the deleted β 0.1 run of the same setup: HR@10
   0.1128 vs 0.1097, NDCG@10 0.0730 vs 0.0722. That is within noise.

## Implication
- With retrieved motives, the model learns to ignore them, whatever the motive version, count, integration or
  objective. This is the same conclusion the CollabRec k-sweep reached.
- The bottleneck is the **use** of motives (reader and decoding), not their mining.
- Open, cheap checks:
  - an epoch-2 checkpoint decode, since every run's validation is best at epoch 2;
  - a CDs MotiveRec-T NTP-only run, which does not exist yet.

## E9b — L_align back on, with a v6 target (job 365889, 2026-11-15)
- **Setup.** Same as run b (v6 cls retriever, 3 motives), plus L_align with β = 0.1. The target is the row's own v6
  extraction, "need -> item kind; …", in `rationales_facet_{train,valid}_v6.jsonl`. Rows without GEARec
  reasoning fall back to the v5 reason (1.4% of train, 6.8% of valid). The code default stays β = 0.

| Run | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| b, NTP only (365542) | 0.0421 | **0.0837** | **0.1128** | 0.0421 | **0.0637** | **0.0730** |
| β 0.1, v6 target (365889) | **0.0426** | 0.0831 | 0.1103 | **0.0426** | 0.0635 | 0.0723 |
| β 0.1, v5 target (deleted 365345) | 0.0424 | 0.0847 | 0.1097 | 0.0424 | 0.0642 | 0.0722 |

| Validation | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---|---|---|
| 365889 NTP | 0.6404 | 0.5566 | 0.5828 |
| 365889 L_align | 2.7453 | 2.6604 | 2.6647 |

- **Reading:**
  - L_align is much easier with the v6 target: 2.66 vs about 3.65 for the v5 target, against chance ln 64 = 4.16.
    The likely reason is that the v6 target shares wording with the 3 v6 motive texts the thought is built from.
  - The easier alignment does not help recommendation. Against NTP-only b, HR@1 is +0.0005, but HR@10 is −0.0025
    and NDCG@10 −0.0007. The v6-target and v5-target versions match each other (HR@10 0.1103 vs 0.1097).
  - All gaps are within one SE. Keep β = 0.

### g3 (365891) — MotiveRec-E Toys, emb, K=3, β 0
Test HR@1/5/10 = 0.0390 / 0.0831 / 0.1091 | NDCG@1/5/10 = 0.0390 / 0.0616 / 0.0700.
Worse than g (K10: 0.0402 / 0.0839 / 0.1094 | 0.0402 / 0.0628 / 0.0710) and below b (text InjectNet K3: 0.0421 / 0.0837 / 0.1128 | 0.0421 / 0.0637 / 0.0730). K=3 does not help emb mode on Toys (unlike CDs h3).

### b-micro2 (365893) — run b (v6, K=3, β 0, text InjectNet) with MICRO=2
Test HR@1/5/10 = 0.0430 / 0.0879 / 0.1129 | NDCG@1/5/10 = 0.0430 / 0.0662 / 0.0743.
vs b (MICRO 8): 0.0421 / 0.0837 / 0.1128 | 0.0421 / 0.0637 / 0.0730 → better on all six.
vs IntuRec Toys: 0.0420 / 0.0833 / 0.1112 | 0.0420 / 0.0632 / 0.0721 → better on all six (first MotiveRec run to do so).
vs O0 Toys: 0.0417 / 0.0867 / 0.1112 | – / 0.0650 / 0.0729 → better.
Caveat: single seed; the gap is small (HR@10 +0.0017 over IntuRec). Validation NTP (0.5839 best, epoch 2) is not comparable with MICRO 8 runs.

## E9c — Reader B on v6 (insertion 0.8 + motive head 0.1), Toys K=3 micro 2 (2026-11-16/17)
Train-only true-motive insertion p 0.8 (rows whose top-3 misses every true motive), motive-head CE weight 0.1;
non-true slots filled with the retriever's next picks (B6) or the true motive's nearest prototypes (B6sim).
Test = epoch-3 in-memory model (validation was best at epoch 2 in both).

| Run | Job | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| b-micro2 (no insertion, no head) | 365893 | **0.0430** | **0.0879** | **0.1129** | **0.0430** | **0.0662** | **0.0743** |
| B6sim (fill similar) | 366599 | 0.0392 | 0.0840 | 0.1121 | 0.0392 | 0.0621 | 0.0712 |
| B6 (fill pred) | 366598 | 0.0386 | 0.0824 | 0.1118 | 0.0386 | 0.0609 | 0.0705 |
| IntuRec | — | 0.0420 | 0.0833 | 0.1112 | 0.0420 | 0.0632 | 0.0721 |
| O0 | 350999 | 0.0417 | 0.0867 | 0.1112 | — | 0.0650 | 0.0729 |

| Valid (NTP / motive loss / head top-1) | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---|---|---|
| B6 | 0.6854 / 5.6545 / 0.1195 | 0.6061 / 5.9224 / 0.1159 | 0.6415 / 6.2796 / 0.1149 |
| B6sim | 0.6862 / 5.7815 / 0.1130 | 0.6002 / 5.7504 / 0.1281 | 0.6306 / 6.0711 / 0.1236 |

Reading: both below b-micro2 on all six metrics (HR@1 −0.004, NDCG@10 −0.003/−0.004); similar fill ≥ pred fill.
Same pattern as v5 reader B in the GRPO session (top3 0.1069 < O0): a reader that trusts motives is hurt by wrong
retrieved ones. Pending: epoch-2 checkpoint decodes (test top; valid top/true/null/random) to test motive use.

### E9c diagnostics — epoch-2 checkpoint decodes (2026-11-17)
| Run | Decode | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| B6 | test / top | 0.0399 | 0.0840 | 0.1067 | 0.0399 | 0.0629 | 0.0704 |
| B6 | valid / true | 0.0604 | 0.1159 | 0.1510 | 0.0604 | 0.0887 | 0.1000 |
| B6 | valid / top | 0.0525 | 0.0946 | 0.1170 | 0.0525 | 0.0739 | 0.0811 |
| B6 | valid / null | 0.0512 | 0.0883 | 0.1115 | 0.0512 | 0.0702 | 0.0777 |
| B6 | valid / random | 0.0447 | 0.0763 | 0.0928 | 0.0447 | 0.0604 | 0.0657 |
| B6sim | test / top | 0.0398 | 0.0841 | 0.1072 | 0.0398 | 0.0622 | 0.0696 |
| B6sim | valid / true | 0.0610 | 0.1185 | 0.1511 | 0.0610 | 0.0903 | 0.1008 |
| B6sim | valid / top | 0.0536 | 0.0983 | 0.1208 | 0.0536 | 0.0764 | 0.0836 |
| B6sim | valid / null | 0.0399 | 0.0652 | 0.0810 | 0.0399 | 0.0529 | 0.0581 |
| B6sim | valid / random | 0.0396 | 0.0652 | 0.0818 | 0.0396 | 0.0528 | 0.0581 |

- Both readers use the motive input: valid HR@10 true > top > null ≥ random. B6sim depends on it most
  (null/random 0.081 vs top 0.121 vs true 0.151); B6 less (null 0.112 ≈ top 0.117).
- Deployable test (top) is lower than the epoch-3 end-of-train decode (B6 0.1067 vs 0.1118) and below b-micro2 0.1129.
  The gain from true motives (+0.03 HR@10) does not survive retrieval noise.

## E9d — curriculum reader C, v5 vs v6 (C5 366678, C6 366679; Toys K3 micro 2, insert 1.0→0.5→0, head 0.1, save last)
| Run | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| C5 (v5) | 0.0430 | 0.0849 | 0.1109 | 0.0430 | 0.0647 | 0.0731 |
| C6 (v6) | 0.0430 | 0.0843 | 0.1109 | 0.0430 | 0.0644 | 0.0729 |
| b-micro2 | 0.0430 | 0.0879 | 0.1129 | 0.0430 | 0.0662 | 0.0743 |

| Valid (NTP / motive loss / head top-1) | Epoch 1 (p 1.0) | Epoch 2 (p 0.5) | Epoch 3 (p 0) |
|---|---|---|---|
| C5 | 0.6864 / 5.6859 / 0.1683 | 0.5913 / 5.3785 / 0.1808 | 0.5996 / 5.4050 / 0.1697 |
| C6 | 0.6960 / 5.9869 / 0.1115 | 0.5946 / 5.5639 / 0.1348 | 0.6109 / 5.5927 / 0.1318 |

- v5 = v6 again (identical HR@1/10). Both are ≈ IntuRec/O0, below b-micro2. Valid diagnostics (top/true/null/random) submitted.

### E9d valid diagnostics (epoch-3 model, valid split)
| Run / motives | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| C5 true | 0.0567 | 0.0980 | 0.1272 | 0.0567 | 0.0783 | 0.0877 |
| C5 top | 0.0571 | 0.0977 | 0.1266 | 0.0571 | 0.0782 | 0.0875 |
| C5 null | 0.0520 | 0.0947 | 0.1208 | 0.0520 | 0.0741 | 0.0825 |
| C5 random | 0.0567 | 0.0971 | 0.1268 | 0.0567 | 0.0775 | 0.0871 |
| C6 true | 0.0595 | 0.1075 | 0.1383 | 0.0595 | 0.0842 | 0.0942 |
| C6 top | 0.0557 | 0.0951 | 0.1185 | 0.0557 | 0.0761 | 0.0837 |
| C6 null | 0.0521 | 0.0870 | 0.1122 | 0.0521 | 0.0702 | 0.0783 |
| C6 random | 0.0463 | 0.0750 | 0.0923 | 0.0463 | 0.0609 | 0.0666 |

- C5 ignores motive content: true ≈ top ≈ random (0.1272/0.1266/0.1268); the +0.006 over null is a slot-format effect.
- C6 reads motive content: true > top > null > random (0.138 > 0.119 > 0.112 > 0.092 HR@10). Retrieved v6 motives help (+0.006 over null), but the gap to true (+0.020) is lost to retrieval error.
- Reading motives costs the no-motive baseline (C6 null 0.1122 vs C5 null 0.1208), so with current retrieval C5 ≥ C6. Bottleneck = retrieval accuracy.
