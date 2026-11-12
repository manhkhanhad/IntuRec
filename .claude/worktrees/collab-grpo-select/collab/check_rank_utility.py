#!/usr/bin/env python3
"""Check a rank_utility file against the stored utilities: the target's ll under each motive must match
utility_<split>.npz (same judge), and print the target's rank among the scored items per motive."""
import sys

import numpy as np

path, util = sys.argv[1], sys.argv[2]
z, u = np.load(path), np.load(util)
nll = z["ll"] / z["ntok"][:, None, :]
for i, r in enumerate(z["row"]):
    s = u["row"] == r
    ref = {(int(k), int(rk)): float(l) for k, rk, l in zip(u["kind"][s], u["rank"][s], u["ll"][s])}
    ours = [(-1, z["ll"][i, 0, 0])] + [(k, z["ll"][i, k + 1, 0]) for k in range(10)]
    diffs = [abs(v - ref[(0, -1)] if k == -1 else v - ref[(3, k)]) for k, v in ours]
    rank = (nll[i] > nll[i][:, :1]).sum(1)
    print(f"row {r}: |ours - stored| target ll  max {max(diffs):.3f}  mean {np.mean(diffs):.3f}; "
          f"target rank among {z['ll'].shape[2]} items (null, top-1..10): {rank.tolist()}")
