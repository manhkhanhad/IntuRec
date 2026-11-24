#!/usr/bin/env python3
"""E8 sweep: coarsen the motive vocabulary and re-measure it with pilot/mining_diagnostics.evaluate. CPU only.

usage: python pilot/mining_merge_sweep.py <dataset>

The existing prototypes are merged with average-linkage clustering, cut at K in {600, 400, 200, 100, 50}, on
  sem   cosine of the Qwen3 phrase centroids (semantic synonyms);
  beh   cosine of the prototypes' train target-item distributions (motives that drive the same purchases);
  mix   0.5 sem + 0.5 beh.
Row labels are remapped (a row keeps the set of merged ids). Train labels only are used to build the merge.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).parent))
from mining_diagnostics import evaluate, load  # noqa: E402


def merge(sim, k):
    d = np.clip(1 - sim, 0, 2)
    np.fill_diagonal(d, 0)
    z = linkage(squareform(d, checks=False), method="average")
    lab = fcluster(z, t=k, criterion="maxclust") - 1
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def main():
    ds = sys.argv[1]
    corpus = Path("data") / ds / "collab" / "motive_corpus"
    df, rm = load(ds, corpus)
    c = np.load(corpus / "centroids.npy")
    K = len(c)
    n_items = int(max(df["train"].item_id.max(), df["test"].item_id.max()) + 1)
    im = np.zeros((K, n_items))
    for it, ms in zip(df["train"].item_id, rm["train"]):
        for m in ms:
            im[m, it] += 1
    im = im / (np.linalg.norm(im, axis=1, keepdims=True) + 1e-8)
    sims = {"sem": c @ c.T, "beh": im @ im.T}
    sims["mix"] = 0.5 * sims["sem"] + 0.5 * sims["beh"]

    rows = [dict(kind="orig", **evaluate(df, rm, K))]
    for kind, sim in sims.items():
        for k in (600, 400, 200, 100, 50):
            lab = merge(sim, k)
            rm2 = {s: [sorted({int(lab[m]) for m in ms}) for ms in v] for s, v in rm.items()}
            r = evaluate(df, rm2, int(lab.max() + 1))
            rows.append(dict(kind=kind, **r))
            print(kind, k, json.dumps(r["motive_prediction_test"]), r["next_item_HR10_NDCG10"], flush=True)
    out = Path("/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints") / ds / "MotiveRec_pilot"
    (out / "mining_merge_sweep.json").write_text(json.dumps(rows, indent=1))

    print(f"\n{'kind':5} {'K':>4} | {'pop R@1/R@10':>13} | {'hist R@1/R@10':>13} | {'user R@1/R@10':>13} | "
          f"{'tgt-item R@1':>12} | oracle HR@10/NDCG@10")
    for r in rows:
        p = r["motive_prediction_test"]
        f = lambda key: f"{p[key]['R@1']:.3f}/{p[key]['R@10']:.3f}"
        o = r["next_item_HR10_NDCG10"]["true_motives(oracle)"]
        print(f"{r['kind']:5} {r['K']:>4} | {f('popularity'):>13} | {f('history_items_profile'):>13} | "
              f"{f('user_past'):>13} | {p['target_item_profile(oracle)']['R@1']:>12.3f} | {o[0]:.4f}/{o[1]:.4f}")


if __name__ == "__main__":
    main()
