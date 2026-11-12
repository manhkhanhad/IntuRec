#!/usr/bin/env python3
"""Offline motive-retrieval analysis: how far can a target-driven retriever go?

Motives are mostly determined by the target item (hypothesis test), so score motives through
likely next items:  P(m | h) = sum_i P(i | h) P(m | i),  P(m | i) = motive distribution of all
train purchases of item i (leave-one-out for train rows), P(i | h) = softmax(SASRec logits / T).

Reports, per split, for rows with >= 1 motive:
  P@k      share of the top-k motives that are true (what O2-style k=1/2 injection needs)
  Hit@k    a true motive is in the top k
  R@k      share of true motives in the top k (train_motive_retriever.py's "recall")
  MRR      of the first true motive
Scorers: popularity, current MLP retriever (retriever/cls, train = out-of-fold),
target ceiling (P(m | true target), LOO), SASRec candidate profile (top-K, T tuned on valid),
and a log-linear mix of MLP and candidate profile (weight tuned on valid).
"""

import argparse
import collections
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from build_support import SasrecEncoder, find_csv, load_split

SPLITS = ("train", "valid", "test")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="Toys_and_Games")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--sasrec-ckpt", type=Path, required=True)
    p.add_argument("--smooth", type=float, default=1.0, help="Dirichlet prior mass (x popularity) per item profile.")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def evaluate(scores, labels, ks=(1, 2, 5, 10)):
    keep = [i for i, l in enumerate(labels) if l]
    order = np.argsort(-scores[keep], axis=1)
    out = collections.defaultdict(list)
    for row, i in zip(order, keep):
        true = set(labels[i])
        isin = np.isin(row, list(true))
        for k in ks:
            out[f"P@{k}"].append(isin[:k].mean())
            out[f"Hit@{k}"].append(isin[:k].any())
            out[f"R@{k}"].append(isin[:k].sum() / len(true))
        out["MRR"].append(1.0 / (np.argmax(isin) + 1) if isin.any() else 0.0)
    return {k: round(float(np.mean(v)), 4) for k, v in out.items()}


def main():
    args = parse_args()
    corpus = args.data_root / args.dataset / "collab" / "motive_corpus"
    frames = {s: load_split(find_csv(args.data_root, args.dataset, s)) for s in SPLITS}
    labels = {s: json.loads((corpus / f"row_motives_{s}.json").read_text()) for s in SPLITS}
    n_proto = len(json.loads((corpus / "prototypes.json").read_text()))
    n_items = sum(1 for _ in next((args.data_root / args.dataset / "info").glob("*.txt")).open(encoding="utf-8"))

    def multi_hot(ls):
        m = np.zeros((len(ls), n_proto), np.float32)
        for i, l in enumerate(ls):
            m[i, l] = 1.0
        return m
    y = {s: multi_hot(labels[s]) for s in SPLITS}
    pop = y["train"].sum(0) + 1.0
    pop_p = pop / pop.sum()

    # Item -> motive counts over train purchases.
    counts = np.zeros((n_items, n_proto), np.float32)
    np.add.at(counts, frames["train"].item_id.to_numpy(), y["train"])

    def profile(c):  # counts [.., n_proto] -> smoothed distribution
        return (c + args.smooth * pop_p) / (c.sum(-1, keepdims=True) + args.smooth)
    item_prof = profile(counts)                                  # [n_items, n_proto]

    results = {}
    # Target ceiling: the true target's motive profile (train rows: leave own row out).
    for s in SPLITS:
        c = counts[frames[s].item_id.to_numpy()]
        if s == "train":
            c = c - y["train"]
        results[f"{s}/target_ceiling"] = evaluate(profile(c), labels[s])
        results[f"{s}/popularity"] = evaluate(np.tile(pop, (len(frames[s]), 1)), labels[s])
        mlp = corpus / "retriever" / "cls" / f"scores_{s}.npy"
        if mlp.exists():
            results[f"{s}/mlp"] = evaluate(np.load(mlp).astype(np.float32), labels[s])

    # SASRec candidate profile.
    sasrec = SasrecEncoder(frames, n_items, SimpleNamespace(sasrec_ckpt=args.sasrec_ckpt))
    logits, target_rank = {}, {}
    for s in SPLITS:
        sasrec.history_vectors(frames[s])
        logits[s] = sasrec.item_scores(frames[s], range(0, len(frames[s])), None).astype(np.float32)
        logits[s][~np.isfinite(logits[s])] = -1e9
        t = frames[s].item_id.to_numpy()
        target_rank[s] = (logits[s] > logits[s][np.arange(len(t)), t][:, None]).sum(1)
        print(f"{s}: SASRec target HR@10 {np.mean(target_rank[s] < 10):.4f}", flush=True)

    def cand_profile(s, temp, topk, chunk=2000):
        out = np.zeros((len(frames[s]), n_proto), np.float32)
        targets = frames[s].item_id.to_numpy()
        for a in range(0, len(out), chunk):
            lg = logits[s][a:a + chunk]
            idx = np.argpartition(-lg, topk, axis=1)[:, :topk]
            sel = np.take_along_axis(lg, idx, 1) / temp
            w = np.exp(sel - sel.max(1, keepdims=True))
            w /= w.sum(1, keepdims=True)
            prof = item_prof[idx]                                # [chunk, topk, n_proto]
            if s == "train":  # leave-one-out where the candidate is the row's own target
                r, c = np.nonzero(idx == targets[a:a + chunk, None])
                if len(r):
                    prof[r, c] = profile(counts[idx[r, c]] - y["train"][a + r])
            out[a:a + chunk] = np.einsum("nk,nkp->np", w, prof)
        return out

    best = None
    for temp in (0.25, 0.5, 1.0, 2.0, 4.0):
        for topk in (10, 50, 200):
            m = evaluate(cand_profile("valid", temp, topk), labels["valid"])
            print(f"cand T={temp} K={topk}: valid {m}", flush=True)
            if best is None or m["MRR"] > best[0]["MRR"]:
                best = (m, temp, topk)
    _, temp, topk = best
    cand = {s: cand_profile(s, temp, topk) for s in SPLITS}
    for s in SPLITS:
        results[f"{s}/sasrec_cand(T={temp},K={topk})"] = evaluate(cand[s], labels[s])

    # Log-linear mix with the MLP retriever.
    if all((corpus / "retriever" / "cls" / f"scores_{s}.npy").exists() for s in SPLITS):
        def lsm(x):
            x = x - x.max(1, keepdims=True)
            return x - np.log(np.exp(x).sum(1, keepdims=True))
        mlp = {s: lsm(np.load(corpus / "retriever" / "cls" / f"scores_{s}.npy").astype(np.float32)) for s in SPLITS}
        best = None
        for a in (0.25, 0.5, 1.0, 2.0, 4.0):
            m = evaluate(mlp["valid"] + a * np.log(cand["valid"]), labels["valid"])
            if best is None or m["MRR"] > best[0]["MRR"]:
                best = (m, a)
        a = best[1]
        for s in SPLITS:
            results[f"{s}/mlp+cand(a={a})"] = evaluate(mlp[s] + a * np.log(cand[s]), labels[s])

    cols = ["P@1", "P@2", "Hit@1", "Hit@2", "Hit@10", "R@10", "MRR"]
    print(f"\n{'scorer':45s} " + " ".join(f"{c:>7s}" for c in cols))
    for k, v in sorted(results.items(), key=lambda kv: (SPLITS.index(kv[0].split('/')[0]), kv[0])):
        print(f"{k:45s} " + " ".join(f"{v[c]:7.4f}" for c in cols))
    out = args.out or corpus / "retriever" / "analysis.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
