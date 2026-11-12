#!/usr/bin/env python3
"""Is history-motive agreement an indicator of a good retrieved motive?

For each row's retriever top-K candidates (valid/test), agreement features computed from train data only:
  hist_p     mean over history items of P(m | item)   (why people buy the items in this history)
  hist_lift  hist_p / P(m)                             (same, corrected for how generic m is)
  user_p     share of m among the user's own train-row motives (valid/test: all of the user's train rows)
  user_lift  user_p / P(m)
  pop        P(m) (genericness, reference)
  logp       retriever log-probability (reference)
Label: candidate is one of the row's true motives (utility labels: see collab/motive_utility.py).
Reports pooled AUC, within-row AUC (ranking candidates of the same row), and whether a logistic
combination fitted on valid improves on the retriever score on test (pooled AUC, P@1 after re-ranking).
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from build_support import find_csv, load_split


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="Toys_and_Games")
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    corpus = args.data_root / args.dataset / "collab" / "motive_corpus"
    splits = ("train", "valid", "test")
    frames = {s: load_split(find_csv(args.data_root, args.dataset, s)) for s in splits}
    truth = {s: json.loads((corpus / f"row_motives_{s}.json").read_text()) for s in splits}
    n_proto = len(json.loads((corpus / "prototypes.json").read_text()))
    protos = json.loads((corpus / "prototypes.json").read_text())

    y_train = np.zeros((len(frames["train"]), n_proto), np.float32)
    for i, l in enumerate(truth["train"]):
        y_train[i, l] = 1.0
    pop = (y_train.sum(0) + 1) / (y_train.sum() + n_proto)
    item_cnt = collections.defaultdict(lambda: np.zeros(n_proto, np.float32))
    for item, row in zip(frames["train"].item_id, y_train):
        item_cnt[int(item)] += row
    item_prof = {k: v / v.sum() for k, v in item_cnt.items() if v.sum() > 0}
    user_rows = collections.defaultdict(list)
    for j, u in enumerate(frames["train"].user_id):
        user_rows[u].append(j)

    feats, labels, groups = {}, {}, {}
    for s in ("valid", "test"):
        f = frames[s]
        scores = np.load(corpus / "retriever" / "cls" / f"scores_{s}.npy").astype(np.float32)
        scores -= scores.max(1, keepdims=True)
        logp = scores - np.log(np.exp(scores).sum(1, keepdims=True))
        cand = np.argsort(-logp, 1)[:, :args.topk]
        X, Y, G = [], [], []
        for r, (hist, user) in enumerate(zip(f.history, f.user_id)):
            if not truth[s][r]:
                continue
            vecs = [item_prof[h] for h in hist if h in item_prof]
            hp = np.mean(vecs, 0) if vecs else np.zeros(n_proto, np.float32)
            rows = user_rows.get(user, [])
            up = y_train[rows].sum(0) if rows else np.zeros(n_proto, np.float32)
            up = up / up.sum() if up.sum() > 0 else up
            for m in cand[r]:
                X.append([hp[m], hp[m] / pop[m], up[m], up[m] / pop[m], pop[m], logp[r, m]])
                Y.append(m in set(truth[s][r]))
                G.append(r)
        feats[s], labels[s], groups[s] = np.array(X), np.array(Y), np.array(G)
    names = ["hist_p", "hist_lift", "user_p", "user_lift", "pop", "logp"]

    def within_auc(score, y, g):
        vals = []
        for r in np.unique(g):
            m = g == r
            if 0 < y[m].sum() < m.sum():
                vals.append(roc_auc_score(y[m], score[m]))
        return float(np.mean(vals))

    def p_at1(score, y, g):
        hits = []
        for r in np.unique(g):
            m = np.where(g == r)[0]
            hits.append(y[m[np.argmax(score[m])]])
        return float(np.mean(hits))

    s = "test"
    X, y, g = feats[s], labels[s], groups[s]
    print(f"{args.dataset} {s}: {len(np.unique(g))} rows x top-{args.topk}, positive rate {y.mean():.3f}")
    print(f"{'feature':12s} {'pooled AUC':>10s} {'within-row AUC':>15s} {'P@1 if ranked by it':>20s}")
    for j, n in enumerate(names):
        print(f"{n:12s} {roc_auc_score(y, X[:, j]):10.3f} {within_auc(X[:, j], y, g):15.3f} {p_at1(X[:, j], y, g):20.3f}")
    tf = lambda Z: np.column_stack([np.log(Z[:, :5] + 1e-6), Z[:, 5]])  # noqa: E731
    for cols, tag in ((list(range(6)), "all agreement + logp"), ([5], "logp only"),
                      ([0, 1, 2, 3, 5], "agreement + logp, no pop")):
        clf = LogisticRegression(max_iter=2000).fit(tf(feats["valid"])[:, cols], labels["valid"])
        sc = clf.decision_function(tf(X)[:, cols])
        print(f"LR[{tag:26s}] fit valid -> test: pooled AUC {roc_auc_score(y, sc):.3f}, "
              f"within-row AUC {within_auc(sc, y, g):.3f}, P@1 {p_at1(sc, y, g):.3f}")
    # Genericness check: how concentrated is the top-1 motive when ranking by each score?
    cand = np.argsort(-np.load(corpus / "retriever" / "cls" / f"scores_{s}.npy").astype(np.float32), 1)[:, :args.topk]
    for j, n in ((5, "logp"), (1, "hist_lift"), (3, "user_lift")):
        winners = collections.Counter()
        for r in np.unique(g):
            m = np.where(g == r)[0]
            winners[int(cand[r][np.argmax(X[m, j])])] += 1
        w, c = winners.most_common(1)[0]
        print(f"  top-1 by {n:9s}: {len(winners)} distinct motives, most frequent {c / len(np.unique(g)):.2f} "
              f"({protos[w]['motive']})")


if __name__ == "__main__":
    main()
