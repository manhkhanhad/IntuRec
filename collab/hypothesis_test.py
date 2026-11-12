#!/usr/bin/env python3
"""Test whether purchase reasoning carries collaborative signal.

Hypothesis: when two users buy the SAME item, the more similar they are
collaboratively (SASRec session embeddings of their histories), the more
similar their reasons for buying it.

Pairs are train rows of different users with the same target (at most
``--max-per-item`` buyers sampled per item). A control set of the same size
pairs rows with DIFFERENT targets. For each pair we compute:
  cf_sim       cosine of SASRec session embeddings
  hist_sim     cosine of history-title text embeddings (control: rationales are
               partly paraphrased history, so this path must be partialled out)
  reason_sim   cosine of the LLM rationale ``reason`` embeddings
  facet_jacc   Jaccard of LLM facets; ``facet_core`` drops genre_style_affinity
               and collection_completion (history-derivable / over-assigned)
  review_sim   cosine of raw review embeddings (no LLM; informative pairs only)

For each outcome we fit, within target item (item fixed effects),
standardised OLS: outcome ~ cf_sim + hist_sim, with 95% CIs from a bootstrap
over items. The effect is reported for all pairs and for pairs where both
reviews are informative, and compared with the different-target control.
"""

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import pandas as pd

from build_support import SasrecEncoder, find_csv, load_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


CORE_EXCLUDE = {"genre_style_affinity", "collection_completion"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CDs_and_Vinyl")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--rationale-tag", default="v5")
    parser.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--sasrec-ckpt", type=Path, default=Path(
        "/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/baselines/results/sasrec/"
        "CDs_and_Vinyl_5_2015-10-2018-11/models/SASRec/2026-09-09_00-49-25-209512.pkl"))
    parser.add_argument("--max-per-item", type=int, default=40)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def embed_cached(texts: list, path: Path, model_name: str) -> np.ndarray:
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(model_name, device=DEVICE).encode(
        texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    np.save(path, emb)
    return emb


def build_pairs(train: pd.DataFrame, max_per_item: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    same, groups = [], []
    for item, rows in train.groupby("item_id").groups.items():
        by_user = {}
        for j in rows:
            by_user.setdefault(train.user_id.iat[j], j)
        rows = list(by_user.values())
        if len(rows) > max_per_item:
            rows = rng.sample(rows, max_per_item)
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                same.append((rows[a], rows[b]))
                groups.append(item)
    same = np.array(same)
    targets, users = train.item_id.to_numpy(), train.user_id.to_numpy()
    diff = []
    while len(diff) < len(same):
        i, j = rng.randrange(len(train)), rng.randrange(len(train))
        if targets[i] != targets[j] and users[i] != users[j]:
            diff.append((i, j))
    return same, np.array(groups), np.array(diff)


def within_ols(y, x, groups, clusters, n_boot, rng) -> dict:
    """Standardised OLS of y on x (columns) after demeaning within groups; cluster-bootstrap CI."""
    frame = pd.DataFrame(np.column_stack([y, x]))
    demeaned = (frame - frame.groupby(groups).transform("mean")).to_numpy()
    demeaned /= demeaned.std(axis=0) + 1e-12
    yd, xd = demeaned[:, 0], demeaned[:, 1:]
    codes, uniq = pd.factorize(clusters)
    k = xd.shape[1]
    xtx = np.zeros((len(uniq), k, k))
    xty = np.zeros((len(uniq), k))
    np.add.at(xtx, codes, xd[:, :, None] * xd[:, None, :])
    np.add.at(xty, codes, xd * yd[:, None])
    beta = np.linalg.solve(xtx.sum(0), xty.sum(0))
    np_rng = np.random.default_rng(rng.randrange(2**32))
    boots = []
    for _ in range(n_boot):
        w = np.bincount(np_rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        boots.append(np.linalg.solve(np.tensordot(w, xtx, 1), w @ xty))
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    return {"beta": beta.round(4).tolist(), "ci_low": lo.round(4).tolist(), "ci_high": hi.round(4).tolist(),
            "n_pairs": int(len(y)), "n_clusters": int(len(uniq))}


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    collab = args.data_root / args.dataset / "collab"
    cache = collab / "hyp_cache"
    cache.mkdir(exist_ok=True)

    train = load_split(find_csv(args.data_root, args.dataset, "train"))
    with (collab / "reviews_train.json").open(encoding="utf-8") as fin:
        reviews = json.load(fin)
    rationales = {}
    with (collab / f"rationales_facet_train_{args.rationale_tag}.jsonl").open(encoding="utf-8") as fin:
        for line in fin:
            rec = json.loads(line)
            rationales[rec["row"]] = rec
    assert len(rationales) == len(train), "rationales must cover every train row"

    n_items = sum(1 for _ in next((args.data_root / args.dataset / "info").glob("*.txt")).open(encoding="utf-8"))
    encoder = SasrecEncoder({"train": train}, n_items, SimpleNamespace(sasrec_ckpt=args.sasrec_ckpt))
    cf = encoder.history_vectors(train)

    hist = embed_cached(["; ".join(eval(t)) for t in train.history_item_title], cache / "hist.npy", args.embedder)
    reason = embed_cached([rationales[i]["reason"] for i in range(len(train))],
                          cache / f"reason_{args.rationale_tag}.npy", args.embedder)
    review = embed_cached([f"{r['summary']}. {r['review']}" for r in reviews], cache / "review.npy", args.embedder)
    facets = [set(rationales[i]["facets"]) for i in range(len(train))]
    informative = np.array([r["informative"] for r in reviews])

    same, groups, diff = build_pairs(train, args.max_per_item, rng)
    print(f"{len(same)} same-target pairs over {len(set(groups))} items; {len(diff)} different-target pairs")

    def pair_features(pairs):
        a, b = pairs[:, 0], pairs[:, 1]
        cos = lambda e: np.einsum("ij,ij->i", e[a], e[b])
        jac = lambda drop: np.array([len((facets[i] - drop) & (facets[j] - drop)) /
                                     max(len((facets[i] - drop) | (facets[j] - drop)), 1) for i, j in pairs])
        return pd.DataFrame({
            "cf_sim": cos(cf), "hist_sim": cos(hist), "reason_sim": cos(reason),
            "facet_jacc": jac(set()), "facet_core": jac(CORE_EXCLUDE), "review_sim": cos(review),
            "both_informative": informative[a] & informative[b],
        })

    feats = {"same": pair_features(same), "diff": pair_features(diff)}
    pair_groups = {"same": groups, "diff": train.item_id.to_numpy()[diff[:, 0]]}

    # Binned view: outcome means by CF-similarity quintile (same-target pairs).
    s = feats["same"]
    s["cf_quintile"] = pd.qcut(s.cf_sim, 5, labels=["Q1 (least similar)", "Q2", "Q3", "Q4", "Q5 (most similar)"])
    print("\nSame-target pairs, outcome mean by CF-similarity quintile:")
    print(s.groupby("cf_quintile", observed=True)[["reason_sim", "facet_jacc", "facet_core", "hist_sim"]]
          .mean().round(4).to_string())
    d = feats["diff"]
    print("Different-target pairs (all):", d[["reason_sim", "facet_jacc", "facet_core", "hist_sim"]].mean().round(4).to_dict())

    results = {}
    print("\nStandardised within-item OLS: outcome ~ cf_sim + hist_sim   [beta_cf (95% CI) | beta_hist]")
    for subset in ("all", "both_informative"):
        for outcome in ("reason_sim", "facet_jacc", "facet_core", "review_sim"):
            if outcome == "review_sim" and subset == "all":
                continue
            for kind in ("same", "diff"):
                f = feats[kind]
                mask = f.both_informative.to_numpy() if subset == "both_informative" else np.ones(len(f), bool)
                # Different-target pairs share no item, so they are demeaned globally;
                # both kinds are bootstrapped over (anchor) items.
                clusters = pair_groups[kind][mask]
                grp = clusters if kind == "same" else np.zeros(mask.sum(), int)
                res = within_ols(f[outcome].to_numpy()[mask], f[["cf_sim", "hist_sim"]].to_numpy()[mask],
                                 grp, clusters, args.bootstrap, rng)
                results[f"{subset}/{outcome}/{kind}"] = res
                print(f"  {subset:16s} {outcome:11s} {kind:4s}  n={res['n_pairs']:7d}  "
                      f"beta_cf={res['beta'][0]:+.4f} ({res['ci_low'][0]:+.4f}, {res['ci_high'][0]:+.4f})  "
                      f"beta_hist={res['beta'][1]:+.4f}")

    with (collab / f"hypothesis_{args.rationale_tag}.json").open("w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=2)


if __name__ == "__main__":
    main()
