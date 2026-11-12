#!/usr/bin/env python3
"""HyDE-style support retrieval in reasoning space.

For each query, the rationale PREDICTED from its history alone
(``gen_rationales.py --mode predict``) is embedded and matched against the
train rationales (written with the target known). Writes, per split,
``<data_root>/<dataset>/collab/support_hyde/support_<split>.json`` with:
  hyde_support    top-k train rows by predicted-vs-train rationale cosine
  hybrid_support  top-k by alpha * rationale cosine + (1 - alpha) * SASRec cosine
Both keep at most one row per user; train queries never retrieve themselves or
the same user's later samples.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from build_support import SasrecEncoder, find_csv, forbidden_rows, load_split
from eval_support import load_rationales, reason_embeddings
from hypothesis_test import embed_cached


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CDs_and_Vinyl")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--rationale-tag", default="v5")
    parser.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--sasrec-ckpt", type=Path, default=Path(
        "/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/baselines/results/sasrec/"
        "CDs_and_Vinyl_5_2015-10-2018-11/models/SASRec/2026-09-09_00-49-25-209512.pkl"))
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=300)
    return parser.parse_args()


def top_unique_users(scores: np.ndarray, forbid: set, users: np.ndarray, k: int, candidates: int) -> list:
    cand = np.argpartition(-scores, candidates)[:candidates]
    picked, seen = [], set()
    for j in cand[np.argsort(-scores[cand])]:
        if j in forbid or users[j] in seen:
            continue
        seen.add(users[j])
        picked.append(int(j))
        if len(picked) == k:
            break
    return picked


def main() -> None:
    args = parse_args()
    collab = args.data_root / args.dataset / "collab"
    train = load_split(find_csv(args.data_root, args.dataset, "train"))
    train_recs = load_rationales(collab / f"rationales_facet_train_{args.rationale_tag}.jsonl", len(train))
    train_emb = reason_embeddings(collab, "train", train_recs, args.rationale_tag, args.embedder)
    users, items = train.user_id.to_numpy(), train.item_id.to_numpy()
    by_user = {}
    for j, user in enumerate(users):
        by_user.setdefault(user, []).append(j)

    frames = {s: load_split(find_csv(args.data_root, args.dataset, s)) for s in ["train", *args.splits]}
    n_items = sum(1 for _ in next((args.data_root / args.dataset / "info").glob("*.txt")).open(encoding="utf-8"))
    encoder = SasrecEncoder(frames, n_items, SimpleNamespace(sasrec_ckpt=args.sasrec_ckpt))
    pvecs = encoder.history_vectors(train)

    out_dir = collab / "support_hyde"
    out_dir.mkdir(exist_ok=True)
    for split in args.splits:
        frame = frames[split]
        preds = load_rationales(collab / f"rationales_predict_{split}_{args.rationale_tag}.jsonl", len(frame))
        pred_emb = embed_cached([p["reason"] for p in preds],
                                collab / "hyp_cache" / f"predict_{args.rationale_tag}_{split}.npy", args.embedder)
        qvecs = encoder.history_vectors(frame)
        forbid = forbidden_rows(frame, split, train, by_user)
        supports = []
        for start in range(0, len(frame), 1024):
            hyde = pred_emb[start:start + 1024] @ train_emb.T
            cf = qvecs[start:start + 1024] @ pvecs.T
            hybrid = args.alpha * hyde + (1 - args.alpha) * cf
            for off in range(len(hyde)):
                idx = start + off
                entry = {}
                for key, scores in (("hyde_support", hyde[off]), ("hybrid_support", hybrid[off])):
                    rows = top_unique_users(scores, forbid[idx], users, args.k, args.candidates)
                    entry[key] = [{"row": j, "item_id": int(items[j]), "sim": round(float(scores[j]), 4),
                                   "same_user": bool(users[j] == frame.user_id.iat[idx])} for j in rows]
                supports.append(entry)
        with (out_dir / f"support_{split}.json").open("w", encoding="utf-8") as fout:
            json.dump(supports, fout)
        print(f"{split}: wrote {len(supports)} rows -> {out_dir / f'support_{split}.json'}")


if __name__ == "__main__":
    main()
