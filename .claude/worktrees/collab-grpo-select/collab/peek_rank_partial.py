#!/usr/bin/env python3
"""Early read of a rank_utility partial file: how much does the motive change the target's rank?"""
import sys

import numpy as np

z = np.load(sys.argv[1])
n = int(z["done"])
nll = z["ll"][:n] / z["ntok"][:n, None, :]                      # [n, 11, J]; motive 0 = null, 1.. = top-10
rank = (nll > nll[:, :, :1]).sum(-1)                           # target rank among J items, per motive
lse = np.log(np.exp(nll - nll.max(-1, keepdims=True)).sum(-1)) + nll.max(-1)
r = nll[:, :, 0] - lse                                          # rank_lse reward
best = rank[:, 1:].min(1)
print(f"rows {n}, items per row {nll.shape[2]}")
print(f"target mean rank: null {rank[:, 0].mean():.2f}, retriever top-1 {rank[:, 1].mean():.2f}, "
      f"mean over top-10 {rank[:, 1:].mean():.2f}, best of top-10 {best.mean():.2f}")
print(f"target ranked 1st: null {np.mean(rank[:, 0] == 0):.3f}, top-1 {np.mean(rank[:, 1] == 0):.3f}, "
      f"best of top-10 {np.mean(best == 0):.3f}")
print(f"rank_lse reward: in-row std over the 10 motives {r[:, 1:].std(1).mean():.3f}; "
      f"top-1 {r[:, 1].mean():.3f}, best {r[:, 1:].max(1).mean():.3f}")
print(f"rows where the best motive differs from top-1: {np.mean(r[:, 1:].argmax(1) != 0):.3f}")
