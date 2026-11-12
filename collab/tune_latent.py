#!/usr/bin/env python3
"""Sweep the cheap knobs of latent HyDE support retrieval.

1. Predictor grid (InfoNCE temperature x hidden size x history-text encoder),
   selected on valid MRR of the predicted vector against the own rationale.
2. Hybrid weight alpha (rationale cosine vs SASRec cosine) for the selected
   predictor, scored on valid with the support harness (target_hit, rank_mrr).
The chosen setting is then reported on test. Writes ``tune_latent.json``.
"""

import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import pandas as pd

from build_hyde_support import top_unique_users
from build_support import SasrecEncoder, find_csv, load_split
from eval_support import centre, load_rationales, rank_metrics, reason_embeddings, score
from hypothesis_test import embed_cached
from latent_hyde import history_text, predict, train_predictor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CDs_and_Vinyl")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--rationale-tag", default="v5")
    parser.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2",
                        help="Rationale embedder (fixed: defines the target space).")
    parser.add_argument("--text-encoders", nargs="+",
                        default=["sentence-transformers/all-mpnet-base-v2", "Alibaba-NLP/gte-large-en-v1.5"])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.03, 0.05, 0.1])
    parser.add_argument("--hiddens", nargs="+", type=int, default=[512, 1024])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    parser.add_argument("--sasrec-ckpt", type=Path, default=Path(
        "/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/baselines/results/sasrec/"
        "CDs_and_Vinyl_5_2015-10-2018-11/models/SASRec/2026-09-09_00-49-25-209512.pkl"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def text_cache_name(encoder: str, split: str) -> str:
    stem = "hist" if encoder.endswith("all-mpnet-base-v2") else "hist_" + encoder.split("/")[-1]
    return f"{stem}.npy" if split == "train" else f"{stem}_{split}.npy"


def embed_text(texts: list, path: Path, encoder: str) -> np.ndarray:
    if path.exists() or encoder.endswith("all-mpnet-base-v2"):
        return embed_cached(texts, path, encoder)
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(encoder, device=DEVICE, trust_remote_code=True).encode(
        texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    np.save(path, emb)
    return emb


def main() -> None:
    args = parse_args()
    collab = args.data_root / args.dataset / "collab"
    cache = collab / "hyp_cache"
    frames = {s: load_split(find_csv(args.data_root, args.dataset, s)) for s in ("train", "valid", "test")}
    train = frames["train"]

    train_recs = load_rationales(collab / f"rationales_facet_train_{args.rationale_tag}.jsonl", len(train))
    raw_train = reason_embeddings(collab, "train", train_recs, args.rationale_tag, args.embedder)
    mean = raw_train.mean(0)
    targets, facets, recs = {"train": centre(raw_train, mean)}, {"train": [set(r["facets"]) for r in train_recs]}, {}
    for split in ("valid", "test"):
        recs[split] = load_rationales(collab / f"rationales_facet_{split}_{args.rationale_tag}.jsonl", len(frames[split]))
        targets[split] = centre(reason_embeddings(collab, split, recs[split], args.rationale_tag, args.embedder), mean)
        facets[split] = [set(r["facets"]) for r in recs[split]]

    n_items = sum(1 for _ in next((args.data_root / args.dataset / "info").glob("*.txt")).open(encoding="utf-8"))
    encoder = SasrecEncoder(frames, n_items, SimpleNamespace(sasrec_ckpt=args.sasrec_ckpt))
    sasrec = {s: encoder.history_vectors(f) for s, f in frames.items()}
    texts = {enc: {s: embed_text(history_text(f), cache / text_cache_name(enc, s), enc) for s, f in frames.items()}
             for enc in args.text_encoders}

    # 1. Predictor grid.
    grid, best = [], None
    for enc, temp, hidden in itertools.product(args.text_encoders, args.temperatures, args.hiddens):
        x = {s: np.concatenate([sasrec[s], texts[enc][s]], 1).astype(np.float32) for s in frames}
        cfg = SimpleNamespace(hidden=hidden, epochs=args.epochs, batch_size=512, lr=1e-3,
                              temperature=temp, seed=args.seed)
        model, valid_mrr = train_predictor(x["train"], targets["train"], x["valid"], targets["valid"], cfg)
        row = {"text_encoder": enc.split("/")[-1], "temperature": temp, "hidden": hidden, "valid_mrr": valid_mrr}
        grid.append(row)
        print("GRID", row, flush=True)
        if best is None or valid_mrr > best[0]:
            best = (valid_mrr, row, model, x)
    _, best_row, model, x = best
    print(f"selected predictor: {best_row}")

    # 2. Alpha sweep on valid (and the chosen alpha on test).
    users, items = train.user_id.to_numpy(), train.item_id.to_numpy()
    train_items = items

    def supports(split: str, alpha: float) -> list:
        pred = predict(model, x[split])
        rows = []
        for start in range(0, len(pred), 1024):
            scores = (alpha * (pred[start:start + 1024] @ targets["train"].T)
                      + (1 - alpha) * (sasrec[split][start:start + 1024] @ sasrec["train"].T))
            rows += [top_unique_users(s, set(), users, args.k, 300) for s in scores]
        return rows

    def harness(split: str, rows: list) -> dict:
        return score(rows, targets[split], facets[split], frames[split].item_id.to_numpy(),
                     targets["train"], facets["train"], train_items)

    sweep = []
    for alpha in args.alphas:
        res = {"alpha": alpha, **{k: float(v) for k, v in harness("valid", supports("valid", alpha)).items()}}
        sweep.append(res)
        print("ALPHA", {k: round(v, 4) for k, v in res.items()}, flush=True)
    sweep_df = pd.DataFrame(sweep)
    # Pick the alpha with the best rank_mrr among those keeping >= 95% of the best target_hit.
    keep = sweep_df[sweep_df.target_hit >= 0.95 * sweep_df.target_hit.max()]
    alpha = float(keep.loc[keep.rank_mrr.idxmax(), "alpha"])
    test_res = {k: float(v) for k, v in harness("test", supports("test", alpha)).items()}
    direct = rank_metrics(predict(model, x["test"]), targets["test"])
    print(f"selected alpha={alpha}; test {({k: round(v, 4) for k, v in test_res.items()})}; "
          f"direct-prediction test mrr {direct['rank_mrr']:.4f}")

    with (collab / "tune_latent.json").open("w", encoding="utf-8") as fout:
        json.dump({"grid": grid, "selected_predictor": best_row, "alpha_sweep": sweep, "selected_alpha": alpha,
                   "test": test_res, "test_direct": direct}, fout, indent=2)


if __name__ == "__main__":
    main()
