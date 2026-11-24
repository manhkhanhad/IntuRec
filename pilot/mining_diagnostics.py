#!/usr/bin/env python3
"""Diagnostics of a motive vocabulary (E8: motive mining quality). CPU only.

usage: python pilot/mining_diagnostics.py <dataset> [corpus_dir]

Measures, from train labels, evaluated on test rows:
  redundancy     prototype pairs with centroid cosine above 0.7 / 0.8 / 0.9;
  concentration  number of prototypes covering 50% / 80% of label occurrences;
  who decides    Recall@1/@10 of the true motive when predicted from
                   the popularity prior, the user's past motives, the target item's motive profile (oracle side info),
                   and the history items' motive profile (what the retriever sees);
  usefulness     HR@10 / NDCG@10 of ranking the next item by sum_m p(m | true row) p(item | m), against popularity:
                   how much the (oracle) motive tells about the next item.
"""
import ast
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load(ds, corpus):
    root = Path("data") / ds
    df = {s: pd.read_csv(glob.glob(str(root / s / "*.csv"))[0]) for s in ("train", "test")}
    rm = {s: json.loads((corpus / f"row_motives_{s}.json").read_text()) for s in ("train", "test")}
    return df, rm


def rank_metrics(scores_rows, targets, ks=(1, 10)):
    out = {f"R@{k}": 0.0 for k in ks}
    for s, t in zip(scores_rows, targets):
        if not t:
            continue
        order = np.argsort(-s)[: max(ks)]
        for k in ks:
            out[f"R@{k}"] += len(set(order[:k]) & set(t)) / len(t)
    n = sum(1 for t in targets if t)
    return {k: round(v / n, 4) for k, v in out.items()}


def main():
    ds = sys.argv[1]
    corpus = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data") / ds / "collab" / "motive_corpus"
    df, rm = load(ds, corpus)
    K = len(json.loads((corpus / "prototypes.json").read_text()))
    c = np.load(corpus / "centroids.npy")
    res = evaluate(df, rm, K, c)
    res["dataset"] = ds
    print(json.dumps(res, indent=1))
    out = Path("/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints") / ds / "MotiveRec_pilot"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"mining_diag_{corpus.name}.json").write_text(json.dumps(res, indent=1))


def evaluate(df, rm, K, c=None):
    res = {"K": K}
    if c is not None:
        sim = np.triu(c @ c.T, 1)
        res["redundant_pairs"] = {str(t): int((sim > t).sum()) for t in (0.7, 0.8, 0.9)}
    cnt = np.bincount([m for ms in rm["train"] for m in ms], minlength=K).astype(float)
    cum = np.cumsum(np.sort(cnt)[::-1]) / cnt.sum()
    res["protos_for_50_80pct"] = [int(np.searchsorted(cum, 0.5) + 1), int(np.searchsorted(cum, 0.8) + 1)]

    tr, te = df["train"], df["test"]
    n_items = int(max(tr.item_id.max(), te.item_id.max(),
                      max(max(ast.literal_eval(h)) for h in tr.history_item_id)) + 1)
    # item -> motive profile (train target rows), user -> past motives (train rows)
    item_m = np.zeros((n_items, K))
    user_m = collections.defaultdict(lambda: np.zeros(K))
    for it, u, ms in zip(tr.item_id, tr.user_id, rm["train"]):
        for m in ms:
            item_m[it, m] += 1
            user_m[u][m] += 1
    prior = cnt / cnt.sum()
    targets = rm["test"]

    def norm(v):
        s = v.sum()
        return v / s if s > 0 else v

    pred = {
        "popularity": [prior for _ in targets],
        "user_past": [norm(user_m[u]) + 1e-6 * prior for u in te.user_id],
        "history_items_profile": [norm(sum(norm(item_m[i]) for i in ast.literal_eval(h))) + 1e-6 * prior
                                  for h in te.history_item_id],
        "target_item_profile(oracle)": [norm(item_m[i]) + 1e-6 * prior for i in te.item_id],
    }
    res["motive_prediction_test"] = {k: rank_metrics(v, targets) for k, v in pred.items()}
    res["test_target_items_seen_in_train"] = round(float(np.mean([item_m[i].sum() > 0 for i in te.item_id])), 4)

    # usefulness: rank next item by p(item | true motives), smoothed with item popularity
    pop = np.bincount(tr.item_id, minlength=n_items).astype(float)
    pop_p = (pop + 1e-3) / (pop + 1e-3).sum()
    p_item_m = (item_m + 0.1 * pop_p[:, None] * 1) / (item_m.sum(0, keepdims=True) + 0.1)  # n_items x K

    def item_hits(score_fn):
        hr = nd = 0.0
        for ms, t in zip(targets, te.item_id):
            s = score_fn(ms)
            rank = int((s > s[t]).sum())
            if rank < 10:
                hr += 1
                nd += 1 / np.log2(rank + 2)
        return round(hr / len(targets), 4), round(nd / len(targets), 4)

    res["next_item_HR10_NDCG10"] = {
        "popularity": item_hits(lambda ms: pop_p),
        "true_motives(oracle)": item_hits(lambda ms: p_item_m[:, ms].mean(1) if ms else pop_p),
    }
    return res


if __name__ == "__main__":
    main()
