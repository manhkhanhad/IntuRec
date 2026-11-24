#!/usr/bin/env python3
"""Control for joint_retriever_k3.py --sup_weight: fine-tune the cls retriever ALONE on its motive retrieval loss
(multi-positive CE over 881 prototypes), with the joint run's exact schedule (same row order, 32 rows/step, AdamW lr
1e-4, warmup 50 + cosine over 1685 steps, grad clip 1.0, dropout on) but no recommender. Its test-retrieval curve vs
the joint run's shows how much of the joint run's retrieval change comes from the recommender.
  python collab/retriever_sup_control.py
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from joint_retriever_k3 import CORPUS, FEATS, load_retriever, retrieval_metrics  # noqa: E402
from joint_reader_selector import DATA  # noqa: E402


def main(lr=1e-4, rows_per_step=32, warmup=50, seed=0, eval_every=200):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_train = len(pd.read_csv(glob.glob(f"{DATA}/train/*.csv")[0]))
    x = {s: torch.from_numpy(np.load(f"{FEATS}/x_{s}.npy").astype(np.float32)) for s in ("train", "test")}
    labels = {s: json.load(open(f"{CORPUS}/row_motives_{s}.json")) for s in ("train", "test")}
    retriever, n_proto = load_retriever("cpu")
    y = torch.zeros(n_train, n_proto)
    for i, l in enumerate(labels["train"]):
        if l:
            y[i, l] = 1.0 / len(l)
    order = rng.permutation(n_train)
    n_steps = (len(order) + rows_per_step - 1) // rows_per_step
    opt = torch.optim.AdamW(retriever.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) *
                                              0.5 * (1 + np.cos(np.pi * min(s, n_steps) / n_steps)))

    def test():
        retriever.eval()
        with torch.no_grad():
            m = retrieval_metrics(retriever(x["test"]).numpy(), labels["test"])
        retriever.train()
        return m

    print(f"initial: {test()}", flush=True)
    retriever.train()
    for step in range(n_steps):
        rows = torch.as_tensor(order[step * rows_per_step:(step + 1) * rows_per_step])
        yb = y[rows]
        has = yb.sum(1) > 0
        loss = -(yb * F.log_softmax(retriever(x["train"][rows]), 1)).sum(1)[has].mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(retriever.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % eval_every == 0 or step + 1 == n_steps:
            print(f"step {step + 1}/{n_steps}: test {test()}", flush=True)


if __name__ == "__main__":
    main()
