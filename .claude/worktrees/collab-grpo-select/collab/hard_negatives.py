#!/usr/bin/env python3
"""Hard negative items per row for the ranking-aware motive reward: SASRec's top-N items for the
row's history, excluding the row's target (and its history items, which the constrained decode can
also produce but rarely ranks). Same SASRec checkpoint as collab/retriever_analysis.py.

Writes <out_dir>/negatives_<split>.npy  [n_rows, N] item ids.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, "/common/home/users/m/mk.ngohuu.2024/research/IntuRec/collab")
from build_support import SasrecEncoder, find_csv, load_split  # noqa: E402

SASREC = ("/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/baselines/results/sasrec/"
          "Toys_and_Games_5_2017-1-2018-11/models/SASRec/2026-10-31_12-00-01-480407.pkl")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="Toys_and_Games")
    p.add_argument("--data-root", type=Path, default=Path("/common/home/users/m/mk.ngohuu.2024/research/IntuRec/data"))
    p.add_argument("--sasrec-ckpt", type=Path, default=Path(SASREC))
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--splits", default="train,test")
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    splits = ("train", "valid", "test")
    frames = {s: load_split(find_csv(a.data_root, a.dataset, s)) for s in splits}
    n_items = sum(1 for _ in next((a.data_root / a.dataset / "info").glob("*.txt")).open(encoding="utf-8"))
    sasrec = SasrecEncoder(frames, n_items, SimpleNamespace(sasrec_ckpt=a.sasrec_ckpt))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for s in a.splits.split(","):
        f = frames[s]
        sasrec.history_vectors(f)
        lg = sasrec.item_scores(f, range(0, len(f)), None).astype(np.float32)
        lg[~np.isfinite(lg)] = -1e9
        lg[np.arange(len(f)), f.item_id.to_numpy()] = -1e10          # never the target
        for i, h in enumerate(f.history):
            lg[i, list(h)] = -1e10                                    # nor a history item
        neg = np.argsort(-lg, 1)[:, :a.n]
        np.save(a.out_dir / f"negatives_{s}.npy", neg.astype(np.int64))
        print(f"{s}: {neg.shape} negatives saved", flush=True)


if __name__ == "__main__":
    main()
