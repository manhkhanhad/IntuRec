# E8e — Are motives reliable for short histories? (Toys, by history length)

Back to [main plan](../../main.md). Date: 2026-11-13. CPU, inline analysis (no new script).
Concern (author): GEARec uses a minimum history length of 3; can we get a reliable motive for short histories?
History length in our Toys data is 1–10 (capped at 10).

## 1. GEARec v2 reasoning (source of v6 motives): target reachability by history length
Best-of-8 candidate, mxbai rank of the target (history masked); final = rank ≤ 100 and judge ≥ 4.

| split | hist | rows | share | has reasoning | rank ≤ 20 | rank ≤ 100 | median rank | pass all |
|---|---|---|---|---|---|---|---|---|
| train | 1 | 10,167 | 0.189 | 0.933 | 0.259 | 0.416 | 155 | 0.391 |
| train | 2 | 9,823 | 0.182 | 1.000 | 0.248 | 0.437 | 159 | 0.420 |
| train | 3 | 9,332 | 0.173 | 1.000 | 0.242 | 0.427 | 176 | 0.414 |
| train | 4–5 | 13,507 | 0.251 | 1.000 | 0.236 | 0.413 | 196 | 0.400 |
| train | 6–10 | 11,069 | 0.205 | 0.995 | 0.241 | 0.428 | 172 | 0.416 |
| test | 1 | 441 | 0.065 | **0.379** | 0.134 | 0.200 | 81 | 0.184 |
| test | 2 | 587 | 0.087 | 1.000 | 0.298 | 0.508 | 96 | 0.487 |
| test | 3 | 890 | 0.132 | 1.000 | 0.246 | 0.429 | 162 | 0.415 |
| test | 4–5 | 2,402 | 0.356 | 0.999 | 0.225 | 0.393 | 223 | 0.383 |
| test | 6–10 | 2,418 | 0.359 | 0.999 | 0.215 | 0.377 | 245 | 0.365 |

(Rank columns are over all rows of the bucket; rows without reasoning count as misses.)

## 2. v5 motive retrieval by history length (test; cls retriever vs popularity)
| hist | rows | cls MRR | cls R@10 | pop MRR | pop R@10 | MRR lift |
|---|---|---|---|---|---|---|
| 1 | 441 | 0.4077 | 0.4567 | 0.2924 | 0.3599 | +0.115 |
| 2 | 587 | 0.3460 | 0.3988 | 0.2898 | 0.3310 | +0.056 |
| 3 | 890 | 0.3084 | 0.3410 | 0.2690 | 0.2858 | +0.039 |
| 4–5 | 2,402 | 0.2736 | 0.2900 | 0.2321 | 0.2330 | +0.042 |
| 6–10 | 2,418 | 0.2633 | 0.2924 | 0.2408 | 0.2334 | +0.023 |

## 3. Recommendation HR@10 by history length (test, single seed)
| model | 1 | 2 | 3 | 4–5 | 6–10 | all |
|---|---|---|---|---|---|---|
| O0 | 0.1769 | **0.1618** | 0.1225 | **0.1003** | 0.0935 | 0.1112 |
| IntuRec | **0.1837** | 0.1567 | **0.1281** | 0.0991 | 0.0926 | 0.1112 |
| MotiveRec-T | **0.1837** | 0.1465 | 0.1225 | 0.0974 | 0.0877 | 0.1072 |
| Curriculum (356575) | 0.1769 | **0.1618** | 0.1213 | 0.0974 | **0.0988** | 0.1119 |

Bucket sizes are small for short histories (441 rows: 95% CI about ±0.036 on HR@10), so only large gaps count.

## Reading
1. **Short histories do not give less reliable motives here.** GEARec reachability is flat over history length on
   train (rank ≤ 20: 0.24–0.26) and highest for length 2 on test. v5 retrieval beats popularity most for short
   histories (MRR +0.115 at length 1) and least for long ones (+0.023 at 6–10).
2. **The hard case is long histories (6–10, 36% of test).** Several interests are mixed, and which one drives the next
   purchase is less determined: reasoning reachability is lowest there (0.215 test), retrieval lift is smallest, and
   MotiveRec-T loses most to IntuRec there (0.0877 vs 0.0926). The curriculum model is best there (0.0988).
3. **Exception: length-1 test rows have GEARec reasoning for only 38%** (most candidates failed the format check).
   This affects only v6 labels for evaluating the retriever; at inference MotiveRec retrieves motives from the history
   and needs no reasoning. Train length-1 has 93% coverage.
4. **A minimum history of 3 is not advisable for MotiveRec training data.** It would drop 37% of train rows (lengths
   1–2), which are the rows with the most retrievable motives, while the real weakness is at length 6–10.
5. Implication for v6 and the retriever: for long histories the retriever must choose *which* interest is active;
   recency-aware features or more retrieved motives (k) for long histories are the levers to test.
