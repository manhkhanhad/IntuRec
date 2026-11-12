#!/usr/bin/env python3
"""Offline usefulness of support sets, scored against each query's TRUE rationale.

Valid/test rationales are written with the target known, so they are never
model inputs; here they only serve as ground truth for how well a support set
anticipates the query's purchase reasoning. For every ``*_support`` list in
each ``<support_dir>/support_<split>.json`` (first --k entries):
  target_hit    the query target is among the supports' targets
  reason_max    max cosine of support rationales to the true rationale
  reason_mean   mean cosine of support rationales to the true rationale
  facet_recall  share of the true facets covered by the supports' facets
  n_items       distinct targets among the supports (diversity)
  rank_mrr/r10  discriminative: the mean of the support rationales ranks the
                query's own true rationale among all the split's true rationales
Embeddings are mean-centred (train mean) and renormalised, so the shared
"generic music lover" direction does not count as a match. Reference rows:
``random`` (k random train rows), ``centroid`` (the same k most generic train
rationales for everyone; should score chance on rank_*) and ``oracle`` (the k
train rationales nearest the true rationale, i.e. the ceiling).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_support import find_csv
from hypothesis_test import embed_cached


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CDs_and_Vinyl")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--support-dirs", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--rationale-tag", default="v5")
    parser.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_rationales(path: Path, n_rows: int) -> list:
    recs = [None] * n_rows
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            rec = json.loads(line)
            recs[rec["row"]] = rec
    assert all(r is not None for r in recs), f"{path} does not cover every row"
    return recs


def reason_embeddings(collab: Path, split: str, recs: list, tag: str, embedder: str) -> np.ndarray:
    return embed_cached([r["reason"] for r in recs], collab / "hyp_cache" / f"reason_{tag}_{split}.npy"
                        if split != "train" else collab / "hyp_cache" / f"reason_{tag}.npy", embedder)


def centre(emb: np.ndarray, mean: np.ndarray) -> np.ndarray:
    out = emb - mean
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-8)


def rank_metrics(pooled: np.ndarray, truth_emb: np.ndarray) -> dict:
    """Rank of each query's own true rationale when the pooled vector scores all true rationales."""
    pooled = pooled / (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-8)
    ranks = []
    for start in range(0, len(pooled), 2048):
        scores = pooled[start:start + 2048] @ truth_emb.T
        own = scores[np.arange(len(scores)), np.arange(start, start + len(scores))]
        ranks.append((scores > own[:, None]).sum(1) + 1)
    ranks = np.concatenate(ranks)
    return {"rank_mrr": float(np.mean(1 / ranks)), "rank_r10": float(np.mean(ranks <= 10))}


def score(rows_per_query: list, truth_emb, truth_facets, truth_items, train_emb, train_facets, train_items) -> dict:
    hit, rmax, rmean, frec, nitems = [], [], [], [], []
    pooled = np.stack([train_emb[np.asarray(rows, dtype=int)].mean(0) for rows in rows_per_query])
    for q, rows in enumerate(rows_per_query):
        rows = np.asarray(rows, dtype=int)
        sims = train_emb[rows] @ truth_emb[q]
        hit.append(truth_items[q] in set(train_items[rows]))
        rmax.append(sims.max())
        rmean.append(sims.mean())
        true_f = truth_facets[q] - {"other"}
        covered = set().union(*(train_facets[j] for j in rows))
        frec.append(len(true_f & covered) / max(len(true_f), 1))
        nitems.append(len(set(train_items[rows])))
    return {"target_hit": np.mean(hit), "reason_max": np.mean(rmax), "reason_mean": np.mean(rmean),
            "facet_recall": np.mean(frec), "n_items": np.mean(nitems), **rank_metrics(pooled, truth_emb)}


def main() -> None:
    args = parse_args()
    collab = args.data_root / args.dataset / "collab"
    train = pd.read_csv(find_csv(args.data_root, args.dataset, "train"))
    train_recs = load_rationales(collab / f"rationales_facet_train_{args.rationale_tag}.jsonl", len(train))
    raw_train = reason_embeddings(collab, "train", train_recs, args.rationale_tag, args.embedder)
    train_mean = raw_train.mean(0)
    train_emb = centre(raw_train, train_mean)
    train_facets = [set(r["facets"]) for r in train_recs]
    train_items = train.item_id.to_numpy()
    rng = np.random.default_rng(args.seed)

    table = []
    for split in args.splits:
        frame = pd.read_csv(find_csv(args.data_root, args.dataset, split))
        recs = load_rationales(collab / f"rationales_facet_{split}_{args.rationale_tag}.jsonl", len(frame))
        emb = centre(reason_embeddings(collab, split, recs, args.rationale_tag, args.embedder), train_mean)
        facets = [set(r["facets"]) for r in recs]
        items = frame.item_id.to_numpy()
        common = (emb, facets, items, train_emb, train_facets, train_items)

        central = np.argsort(-(raw_train @ (train_mean / np.linalg.norm(train_mean))))[:args.k]
        refs = {"random": [rng.choice(len(train), args.k, replace=False) for _ in range(len(frame))],
                "centroid": [central] * len(frame),
                "oracle": [np.argpartition(-(train_emb @ e), args.k)[:args.k] for e in emb]}
        for name, rows in refs.items():
            table.append({"split": split, "retriever": name, **score(rows, *common)})
        for directory in args.support_dirs:
            with (Path(directory) / f"support_{split}.json").open(encoding="utf-8") as fin:
                supports = json.load(fin)
            for key in [k for k in supports[0] if k.endswith("_support")]:
                rows = [[s["row"] for s in entry[key][:args.k]] for entry in supports]
                table.append({"split": split, "retriever": f"{Path(directory).name}/{key}", **score(rows, *common)})

    out = pd.DataFrame(table)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
