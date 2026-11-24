#!/usr/bin/env python3
"""Test-split motive inputs [n, 3] for the predicted-motive k=3 reader (run 364239), to check whether
the reader uses motive content:
  top3  retriever top-3 by 'cls' score (what 364239 saw in training/test; must reproduce HR@10 0.1123)
  true  the row's true motives first (unique, up to 3), remaining slots = top non-true predictions
  null  "No specific motive." in all 3 slots
Writes collab/select_picks/k3_<name>.npy.
"""
import json
import os

import numpy as np

DATA = "/common/home/users/m/mk.ngohuu.2024/research/IntuRec/data/Toys_and_Games/collab/motive_corpus"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "select_picks")

scores = np.load(f"{DATA}/retriever/cls/scores_test.npy").astype(np.float32)
# exactly CollabRecDataset(motive_source="predicted"): float32 log-softmax, then argsort (tie order matters)
scores -= scores.max(1, keepdims=True)
scores = scores - np.log(np.exp(scores).sum(1, keepdims=True))
with open(f"{DATA}/row_motives_test.json", encoding="utf-8") as f:
    true = json.load(f)
with open(f"{DATA}/prototypes.json", encoding="utf-8") as f:
    null = len(json.load(f))
order = np.argsort(-scores, axis=1)[:, :5]
n = len(true)
assert scores.shape[0] == n, (scores.shape, n)

top3 = order[:, :3].astype(np.int64)
tr = np.empty((n, 3), np.int64)
for i, t in enumerate(true):
    t = list(dict.fromkeys(int(m) for m in t))[:3]
    fill = [int(m) for m in order[i] if int(m) not in set(t)]
    tr[i] = (t + fill)[:3]
nul = np.full((n, 3), null, np.int64)
for name, a in [("top3", top3), ("true", tr), ("null", nul)]:
    np.save(f"{OUT}/k3_{name}.npy", a)
print(f"rows {n}, null id {null}, rows with >=1 true motive {sum(bool(t) for t in true)}, "
      f"true motive in retriever top-3 {np.mean([bool(set(t) & set(top3[i].tolist())) for i, t in enumerate(true)]):.4f}")
