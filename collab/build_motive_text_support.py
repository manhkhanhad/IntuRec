#!/usr/bin/env python3
"""Text-mode MotiveRec supports from a motive corpus + its retriever scores (CPU, seconds).

  <corpus>/stage2/support_texts.json      "<motive>: <reason>" per prototype (same format as Toys v6 P1000/stage2)
  <corpus>/stage2/support_rows_<split>.npy top-k prototype ids by retriever score (train scores are cross-fitted)

python collab/build_motive_text_support.py --dataset CDs_and_Vinyl --corpus motive_corpus_v6/P1000 [--head cls --k 10]
"""
import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--data-root", type=Path, default=Path("data"))
p.add_argument("--corpus", default="motive_corpus_v6/P1000")
p.add_argument("--head", default="cls")
p.add_argument("--k", type=int, default=10)
a = p.parse_args()

c = a.data_root / a.dataset / "collab" / a.corpus
out = c / "stage2"
out.mkdir(exist_ok=True)
protos = json.load(open(c / "prototypes.json", encoding="utf-8"))
json.dump([f"{q['motive']}: {q['reason']}" for q in protos], open(out / "support_texts.json", "w", encoding="utf-8"),
          ensure_ascii=False)
for s in ("train", "valid", "test"):
    sc = np.load(c / "retriever" / a.head / f"scores_{s}.npy")
    assert sc.shape[1] == len(protos), (s, sc.shape, len(protos))
    rows = np.argsort(-sc, 1)[:, : a.k].astype(np.int64)
    np.save(out / f"support_rows_{s}.npy", rows)
    print(f"{s}: {rows.shape}")
print(f"wrote {out} ({len(protos)} support texts)")
