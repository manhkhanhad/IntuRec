# E2a — Accuracy by purchase-motive facet (optional support for "why motives")

Back to [main plan](../../main.md). Date: 2026-11-13. Code: `pilot/pilot_a_motive_breakdown.py`. Output:
`<ckpt>/<ds>/MotiveRec_pilot/pilot_a.json`.

## Setup
- **Strata:** test rows are grouped by the coarse motive facets of the target purchase: 13 types per dataset, from
  `rationales_facet_<split>_v5.jsonl`, excluding `other`. A row with 2 facets counts in both groups.
- **Models:** O0, IntuRec, R³-SFT, R³-RL, using their existing test predictions.
- **Spread test:** the across-facet std of HR@10 against a 2000-permutation null.
- **Controlled test:** logistic regression hit@10 ~ title Jaccard + log target popularity + log history length, with
  vs without the facet indicators; likelihood-ratio test.

## Results (HR@10; O0 / IntuRec / R³-SFT / R³-RL)
**CDs** (n 6157)

| facet (rows) | O0 | IntuRec | R³-SFT | R³-RL |
|---|---|---|---|---|
| occasion_gift_seasonal (223) | 0.2915 | 0.2780 | 0.2825 | 0.2870 |
| artist_loyalty (179) | 0.1955 | 0.2123 | 0.1676 | 0.2123 |
| collection_completion (928) | 0.1994 | 0.2004 | 0.1929 | 0.2015 |
| discovery_novelty (804) | 0.0759 | 0.0796 | 0.0672 | 0.0784 |
| live_performance (346) | 0.0925 | 0.0751 | 0.0665 | 0.0751 |
| compilation_best_of (2204) | 0.1003 | 0.0998 | 0.0649 | 0.0789 |
| overall | 0.1178 | 0.1202 | 0.0989 | 0.1095 |

**Toys** (n 6738)

| facet (rows) | O0 | IntuRec | R³-SFT | R³-RL |
|---|---|---|---|---|
| franchise_brand_loyalty (402) | 0.3134 | 0.2910 | 0.2935 | 0.2886 |
| collection_completion (1898) | 0.2445 | 0.2455 | 0.2276 | 0.2313 |
| interest_theme_affinity (526) | 0.1692 | 0.1692 | 0.1654 | 0.1711 |
| active_outdoor_play (605) | 0.0397 | 0.0397 | 0.0281 | 0.0331 |
| quality_safety (1042) | 0.0518 | 0.0585 | 0.0451 | 0.0461 |
| age_stage_fit (144) | 0.0625 | 0.0486 | 0.0417 | 0.0417 |
| value_price (874) | 0.0595 | 0.0641 | 0.0538 | 0.0549 |
| overall | 0.1112 | 0.1112 | 0.1023 | 0.1055 |

**Tests**
- **Spread:** across-facet std is 0.057–0.061 (CDs) and 0.076–0.079 (Toys), against a null of 0.012–0.016;
  p < 0.0005 for every model. The best/worst facet ratio is 3.7–4.4 (CDs) and 7.3–10.5 (Toys).
- **Controlled LR test:** p ≤ 1e-11 for every model and dataset (CDs LR 131–221, Toys 77–87, df 13).
- **Motive continuation vs shift:** not significant; not usable. Users without train rows have a higher HR
  (0.12–0.18), which is unexplained.

## Reading and caveat
- Accuracy depends strongly on the motive type, beyond lexical similarity, popularity and history length.
- All models share the same profile, and latent reasoning does not close the gap on the hard facets.
- **Caveat:** the facets come from an LLM that saw the target and its review, so they partly describe the target's
  purchase type. Use them as analysis strata only.
- **Planned use:** E7 per-motive-type analysis of MotiveRec.
