#!/usr/bin/env python3
"""Cache the input features of the Toys 'cls' motive retriever (collab/train_motive_retriever.py, head cls,
saved as motive_corpus/retriever/cls/mlp.pt) so the retriever can be trained jointly with the recommender.

Rebuilds exactly the retriever's features (SASRec history vector, gte-large history-title embedding, item motive
profile, user's past motives), standardised with the saved mu / sd, and checks that mlp.pt on them reproduces the
stored scores_{valid,test}.npy. Writes <out>/x_{train,valid,test}.npy (float16) + check.json.
  python collab/retriever_features.py --out /common/scratch/.../CollabRec_GRPO_select/retriever_feats
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

MAIN = Path("/common/home/users/m/mk.ngohuu.2024/research/IntuRec")
sys.path.insert(0, str(MAIN / "collab"))
os.chdir(MAIN)
from build_support import SasrecEncoder, find_csv, load_split  # noqa: E402
from train_motive_retriever import Retriever, normalise_rows  # noqa: E402

SPLITS = ("train", "valid", "test")
DATASET = "Toys_and_Games"


def build(data_root=Path("data")):
    collab = data_root / DATASET / "collab"
    corpus = collab / "motive_corpus"
    ck = torch.load(corpus / "retriever" / "cls" / "mlp.pt", map_location="cpu", weights_only=False)
    rargs = SimpleNamespace(**ck["args"])
    frames = {s: load_split(find_csv(data_root, DATASET, s)) for s in SPLITS}
    labels = {s: json.loads((corpus / f"row_motives_{s}.json").read_text()) for s in SPLITS}
    n_proto = len(json.loads((corpus / "prototypes.json").read_text()))
    y_train = np.zeros((len(labels["train"]), n_proto), np.float32)
    for i, l in enumerate(labels["train"]):
        y_train[i, l] = 1.0
    train = frames["train"]
    item_counts = collections.defaultdict(lambda: np.zeros(n_proto, dtype=np.float32))
    for item, row in zip(train.item_id, y_train):
        item_counts[int(item)] += row
    item_prof = {k: v / v.sum() for k, v in item_counts.items() if v.sum() > 0}

    def item_profile(frame):
        f = np.zeros((len(frame), n_proto), dtype=np.float32)
        for i, hist in enumerate(frame.history):
            vecs = [item_prof[h] for h in hist if h in item_prof]
            if vecs:
                f[i] = np.mean(vecs, 0)
        return f

    user_rows = collections.defaultdict(list)
    for j, (user, hist) in enumerate(zip(train.user_id, train.history)):
        user_rows[user].append((len(hist), j))

    def user_past(frame, split):
        f = np.zeros((len(frame), n_proto), dtype=np.float32)
        for i, (user, hist) in enumerate(zip(frame.user_id, frame.history)):
            rows = [j for L, j in user_rows.get(user, []) if split != "train" or L < len(hist)]
            if rows:
                f[i] = y_train[rows].sum(0)
        return normalise_rows(f)

    n_items = sum(1 for _ in next((data_root / DATASET / "info").glob("*.txt")).open(encoding="utf-8"))
    sasrec = SasrecEncoder(frames, n_items, SimpleNamespace(sasrec_ckpt=rargs.sasrec_ckpt))
    x = {}
    for s, f in frames.items():
        text = np.load(collab / "hyp_cache" / ("hist_gte-large-en-v1.5.npy" if s == "train"
                                               else f"hist_gte-large-en-v1.5_{s}.npy"))
        parts = {"sasrec": sasrec.history_vectors(f), "text": text,
                 "item_profile": item_profile(f), "user_past": user_past(f, s)}
        raw = torch.from_numpy(np.concatenate([parts[k] for k in ck["keys"]], 1)).float()
        x[s] = ((raw - ck["mu"]) / ck["sd"]).numpy()
    model = Retriever(x["train"].shape[1], n_proto, np.zeros((n_proto, 1), np.float32), rargs)  # cls: text unused
    model.load_state_dict(ck["state_dict"])
    model.eval()
    check = {}
    with torch.no_grad():
        for s in ("valid", "test"):
            ref = np.load(corpus / "retriever" / "cls" / f"scores_{s}.npy").astype(np.float32)
            got = model(torch.from_numpy(x[s])).numpy()
            check[s] = {"max_abs_diff": float(np.abs(got - ref).max()),
                        "top3_agree": float((np.argsort(-got, 1)[:, :3] == np.argsort(-ref, 1)[:, :3]).all(1).mean())}
    return x, check, labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    x, check, _ = build()
    for s in SPLITS:
        np.save(os.path.join(a.out, f"x_{s}.npy"), x[s].astype(np.float16))
    json.dump(check, open(os.path.join(a.out, "check.json"), "w"), indent=1)
    print({s: v.shape for s, v in x.items()}, check, flush=True)


if __name__ == "__main__":
    main()
