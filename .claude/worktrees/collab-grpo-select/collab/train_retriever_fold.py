#!/usr/bin/env python3
"""Fold retriever for joint training without the 'already fitted' confound.

The cls retriever (motive_corpus/retriever/cls/mlp.pt) was fit on ALL train rows, so joint training on those rows
starts at its optimum (train top-3 0.461 vs test 0.310) and any label-free update loses fit. Here: split train rows
by user into halves A / B (GroupKFold(2), as train_motive_retriever's cross-fitting), fit the same cls MLP recipe
(collab/train_motive_retriever.py: hidden 1024, dropout 0.3, AdamW lr 1e-3, wd 1e-4, batch 512, 40 epochs,
multi-positive softmax CE, early stopping on valid MRR) on half A only, on the cached standardised features
(collab/retriever_features.py). Joint training then runs on half B, whose rows this retriever never saw.
Writes <out>/mlp.pt (same format as cls/mlp.pt), rows_A.npy, rows_B.npy, results.json.
  python collab/train_retriever_fold.py --out /common/scratch/.../CollabRec_GRPO_select/retriever_foldA
"""
import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_retriever_k3 import CORPUS, FEATS, retrieval_metrics  # noqa: E402
from joint_reader_selector import DATA  # noqa: E402

sys.path.insert(0, "/common/home/users/m/mk.ngohuu.2024/research/IntuRec/collab")
from train_motive_retriever import Retriever  # noqa: E402


def mrr(scores, labels):
    keep = [i for i, l in enumerate(labels) if l]
    order = np.argsort(-scores[keep], 1)
    return float(np.mean([1.0 / (np.where(np.isin(order[j], labels[i]))[0].min() + 1) for j, i in enumerate(keep)]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(f"{CORPUS}/retriever/cls/mlp.pt", map_location="cpu", weights_only=False)
    rargs = SimpleNamespace(**ck["args"])
    train = pd.read_csv(glob.glob(f"{DATA}/train/*.csv")[0])
    x = {s: torch.from_numpy(np.load(f"{FEATS}/x_{s}.npy").astype(np.float32)).to(device) for s in ("train", "valid", "test")}
    labels = {s: json.load(open(f"{CORPUS}/row_motives_{s}.json")) for s in ("train", "valid", "test")}
    n_proto = ck["state_dict"]["query.3.weight"].shape[0]
    y = np.zeros((len(train), n_proto), np.float32)
    for i, l in enumerate(labels["train"]):
        if l:
            y[i, l] = 1.0 / len(l)
    rows_a, rows_b = next(GroupKFold(n_splits=2).split(np.zeros(len(train)), groups=train.user_id.to_numpy()))
    rows_a, rows_b = np.sort(rows_a), np.sort(rows_b)
    fit = rows_a[y[rows_a].sum(1) > 0]
    xt, yt = x["train"][torch.from_numpy(fit).to(device)], torch.from_numpy(y[fit]).to(device)
    model = Retriever(xt.shape[1], n_proto, np.zeros((n_proto, 1), np.float32), rargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=rargs.lr, weight_decay=rargs.weight_decay)
    best, best_state = -1.0, None
    for epoch in range(rargs.epochs):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        for s in range(0, len(perm), rargs.batch_size):
            idx = perm[s:s + rargs.batch_size]
            loss = -(yt[idx] * F.log_softmax(model(xt[idx]), 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            m = mrr(model(x["valid"]).cpu().numpy(), labels["valid"])
        if m > best:
            best, best_state = m, {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        if epoch % 5 == 4:
            print(f"epoch {epoch + 1}: loss {loss.item():.3f}, valid mrr {m:.4f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    res = {"rows_A": int(len(rows_a)), "rows_B": int(len(rows_b)), "valid_mrr": round(best, 4)}
    with torch.no_grad():
        for name, s, rows in [("train_A (fitted)", "train", rows_a), ("train_B (unseen)", "train", rows_b),
                              ("test", "test", None)]:
            xs = x[s] if rows is None else x[s][torch.from_numpy(rows).to(device)]
            ls = labels[s] if rows is None else [labels[s][i] for i in rows]
            res[name] = retrieval_metrics(model(xs).cpu().numpy(), ls)
        full = Retriever(xt.shape[1], n_proto, np.zeros((n_proto, 1), np.float32), rargs).to(device)
        full.load_state_dict(ck["state_dict"]); full.eval()
        res["full mlp.pt on train_B (fitted)"] = retrieval_metrics(
            full(x["train"][torch.from_numpy(rows_b).to(device)]).cpu().numpy(), [labels["train"][i] for i in rows_b])
    torch.save({"state_dict": best_state, "mu": ck["mu"], "sd": ck["sd"], "keys": ck["keys"], "args": ck["args"]},
               os.path.join(a.out, "mlp.pt"))
    np.save(os.path.join(a.out, "rows_A.npy"), rows_a)
    np.save(os.path.join(a.out, "rows_B.npy"), rows_b)
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
