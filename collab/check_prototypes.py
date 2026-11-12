#!/usr/bin/env python3
"""Consistency checks for a prototype corpus (collab/build_prototypes.py output).

  coherence   mean pairwise cosine between members of a cluster vs. random pairs
  margin      sim(own centroid) - sim(2nd nearest centroid) per rationale (ambiguity)
  purity      share of members carrying the cluster's most common facet vs. its base rate
  text fit    cosine of the prototype text to its members vs. to random rationales
  stability   ARI / NMI between the corpus clustering and re-runs with other seeds
Prints a high / median / low coherence cluster with members for inspection.
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from build_prototypes import spherical_kmeans
from eval_support import centre, load_rationales, reason_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CDs_and_Vinyl")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--rationale-tag", default="v5")
    parser.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--P", type=int, default=1000)
    parser.add_argument("--stability-seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--kmeans-iters", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    collab = args.data_root / args.dataset / "collab"
    out = collab / "prototypes" / f"P{args.P}"
    n = len(pd.read_csv(next((args.data_root / args.dataset / "train").glob("*.csv"))))
    recs = load_rationales(collab / f"rationales_facet_train_{args.rationale_tag}.jsonl", n)
    raw = reason_embeddings(collab, "train", recs, args.rationale_tag, args.embedder)
    mean = raw.mean(0)
    x = centre(raw, mean)
    c = np.load(out / "centroids.npy")
    protos = json.loads((out / "prototypes.json").read_text())
    text = centre(np.load(out / "prototype_text_emb.npy"), mean)
    sims = torch.from_numpy(x).to(device) @ torch.from_numpy(c).to(device).T
    top2 = sims.topk(2, dim=1)
    assign = top2.indices[:, 0].cpu().numpy()
    margin = (top2.values[:, 0] - top2.values[:, 1]).cpu().numpy()
    rng = np.random.default_rng(0)

    # Coherence and facet purity per cluster.
    facet_base = collections.Counter(f for r in recs for f in r["facets"])
    rows = []
    for k in range(args.P):
        idx = np.where(assign == k)[0]
        if len(idx) < 2:
            rows.append({"id": k, "size": len(idx), "coherence": np.nan, "purity": np.nan, "purity_lift": np.nan})
            continue
        s = x[idx] @ x[idx].T
        coherence = (s.sum() - len(idx)) / (len(idx) * (len(idx) - 1))
        facets = collections.Counter(f for i in idx for f in recs[i]["facets"])
        top_facet, cnt = facets.most_common(1)[0]
        purity = cnt / len(idx)
        rows.append({"id": k, "size": len(idx), "coherence": coherence, "purity": purity,
                     "purity_lift": purity / (facet_base[top_facet] / len(recs))})
    df = pd.DataFrame(rows)
    pairs = rng.integers(0, len(x), (200000, 2))
    random_cos = float((x[pairs[:, 0]] * x[pairs[:, 1]]).sum(1).mean())
    member_fit = np.array([(x[assign == k] @ text[k]).mean() if (assign == k).any() else np.nan for k in range(args.P)])
    random_fit = float((x[rng.integers(0, len(x), 5000)] @ text.T).mean())

    print(f"P={args.P}  rows={len(x)}")
    print(f"coherence (mean pairwise cos within cluster): median {df.coherence.median():.3f}, "
          f"p10 {df.coherence.quantile(.1):.3f}, p90 {df.coherence.quantile(.9):.3f}  | random pairs {random_cos:.3f}")
    print(f"  clusters with coherence < 0.3: {(df.coherence < 0.3).sum()}  (< 0.2: {(df.coherence < 0.2).sum()})")
    print(f"  size-weighted coherence: {np.nansum(df.coherence * df['size']) / df['size'].sum():.3f}")
    print(f"assignment margin: median {np.median(margin):.3f}; rows with margin < 0.02: {(margin < 0.02).mean():.1%}, "
          f"< 0.05: {(margin < 0.05).mean():.1%}")
    print(f"facet purity (share with cluster's top facet): median {df.purity.median():.2f}, "
          f"lift over base rate median {df.purity_lift.median():.2f}x")
    print(f"prototype text fit: members {np.nanmean(member_fit):.3f} vs random rationales {random_fit:.3f}; "
          f"clusters where text fits members < 0.3: {(member_fit < 0.3).sum()}")
    print("coherence by cluster size:")
    df["size_bin"] = pd.cut(df["size"], [0, 4, 20, 50, 100, 1000], labels=["<5", "5-20", "21-50", "51-100", ">100"])
    print(df.groupby("size_bin", observed=True).agg(clusters=("id", "count"), coherence=("coherence", "median"),
                                                     purity=("purity", "median")).round(3).to_string())

    for seed in args.stability_seeds:
        c2 = spherical_kmeans(torch.from_numpy(x).float().to(device), args.P, args.kmeans_iters, seed)
        a2 = (torch.from_numpy(x).float().to(device) @ c2.T).argmax(1).cpu().numpy()
        print(f"stability vs seed {seed}: ARI {adjusted_rand_score(assign, a2):.3f}, NMI {normalized_mutual_info_score(assign, a2):.3f}")

    ranked = df.dropna().query("size >= 10").sort_values("coherence")
    for name, k in (("LOW", ranked.iloc[0].id), ("MEDIAN", ranked.iloc[len(ranked) // 2].id), ("HIGH", ranked.iloc[-1].id)):
        k = int(k)
        idx = np.where(assign == k)[0]
        print(f"\n--- {name} coherence cluster {k}: size {len(idx)}, coherence {df.coherence[k]:.3f}, "
              f"top facets {protos[k]['top_facets']}")
        print(f"    PROTOTYPE: {protos[k]['label']} | {protos[k]['reason']}")
        for i in rng.choice(idx, min(5, len(idx)), replace=False):
            print(f"    member: {recs[i]['reason']}")


if __name__ == "__main__":
    main()
