#!/usr/bin/env python3
"""Export the "MLP + SASRec candidates" motive scores as a retriever score set for CollabRecDataset.

Same scorer as collab/retriever_analysis.py (main checkout), with its valid-tuned settings fixed:
  cand(m | h) = sum over SASRec's top-K items i of softmax(SASRec logit_i / T) * P(m | i)
  score      = log_softmax(MLP cls scores) + a * log cand(m | h)
P(m | i) = motive distribution of the train purchases of item i, Dirichlet-smoothed towards
popularity (mass --smooth). Valid / test only: rows never contribute to their own item profile.
Writes <data>/<dataset>/collab/motive_corpus/retriever/<name>/scores_<split>.npy (+ meta.json) and
prints HR@1 / HR@10 (rows with >= 1 motive), to check against retriever/analysis.json.

Usage (CPU): python collab/export_mix_scores.py --dataset Toys_and_Games \
    --sasrec-ckpt .../SASRec/2026-10-31_12-00-01-480407.pkl --temp 4.0 --topk 50 --alpha 0.5
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

MAIN_COLLAB = Path("/common/home/users/m/mk.ngohuu.2024/research/IntuRec/collab")
sys.path.insert(0, str(MAIN_COLLAB))
from build_support import SasrecEncoder, find_csv, load_split  # noqa: E402

SPLITS = ("train", "valid", "test")


def hit_rates(scores, labels):
    keep = [i for i, l in enumerate(labels) if l]
    order = np.argsort(-scores[keep], axis=1)[:, :10]
    hit = np.array([np.isin(o, labels[i]) for o, i in zip(order, keep)])
    return {"HR@1": round(float(hit[:, 0].mean()), 4), "HR@10": round(float(hit.any(1).mean()), 4),
            "rows": len(keep)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="Toys_and_Games")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--sasrec-ckpt", type=Path, required=True)
    p.add_argument("--temp", type=float, required=True)
    p.add_argument("--topk", type=int, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--smooth", type=float, default=1.0)
    p.add_argument("--mlp", default="cls")
    p.add_argument("--name", default="mlp_cand")
    p.add_argument("--splits", default="valid,test")
    args = p.parse_args()

    corpus = args.data_root / args.dataset / "collab" / "motive_corpus"
    frames = {s: load_split(find_csv(args.data_root, args.dataset, s)) for s in SPLITS}
    labels = {s: json.loads((corpus / f"row_motives_{s}.json").read_text()) for s in SPLITS}
    n_proto = len(json.loads((corpus / "prototypes.json").read_text()))
    n_items = sum(1 for _ in next((args.data_root / args.dataset / "info").glob("*.txt")).open(encoding="utf-8"))

    y_train = np.zeros((len(labels["train"]), n_proto), np.float32)
    for i, l in enumerate(labels["train"]):
        y_train[i, l] = 1.0
    pop = y_train.sum(0) + 1.0
    pop_p = pop / pop.sum()
    counts = np.zeros((n_items, n_proto), np.float32)
    np.add.at(counts, frames["train"].item_id.to_numpy(), y_train)
    item_prof = (counts + args.smooth * pop_p) / (counts.sum(-1, keepdims=True) + args.smooth)

    sasrec = SasrecEncoder(frames, n_items, SimpleNamespace(sasrec_ckpt=args.sasrec_ckpt))
    out_dir = corpus / "retriever" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for s in args.splits.split(","):
        sasrec.history_vectors(frames[s])
        lg = sasrec.item_scores(frames[s], range(0, len(frames[s])), None).astype(np.float32)
        lg[~np.isfinite(lg)] = -1e9
        idx = np.argpartition(-lg, args.topk, axis=1)[:, :args.topk]
        sel = np.take_along_axis(lg, idx, 1) / args.temp
        w = np.exp(sel - sel.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        cand = np.einsum("nk,nkp->np", w, item_prof[idx])
        mlp = np.load(corpus / "retriever" / args.mlp / f"scores_{s}.npy").astype(np.float32)
        mlp = mlp - mlp.max(1, keepdims=True)
        mlp = mlp - np.log(np.exp(mlp).sum(1, keepdims=True))
        score = mlp + args.alpha * np.log(cand)
        np.save(out_dir / f"scores_{s}.npy", score.astype(np.float32))
        report[s] = {"mlp": hit_rates(mlp, labels[s]), "cand": hit_rates(cand, labels[s]),
                     args.name: hit_rates(score, labels[s])}
        print(s, report[s], flush=True)
    (out_dir / "meta.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()}, "report": report},
                                                  indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
