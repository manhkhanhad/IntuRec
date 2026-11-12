#!/usr/bin/env python3
"""Build the stage-2 inputs for collaborative reasoning (run_inturec_f2.py --support_dir).

Supports are latent-hybrid retrievals: train rows ranked by
alpha * cos(predicted rationale, train rationale) + (1 - alpha) * cos(SASRec histories),
at most one per user. The predictor/alpha settings come from ``tune_latent.json``.

Leakage control:
  * valid/test use a predictor trained on all train rows;
  * train rows use 5-fold cross-fitting grouped by user, so a row's support query
    never comes from a predictor that saw its own rationale;
  * train queries never retrieve themselves or the same user's later samples;
  * as in IntuRec's target-aware candidate balancing, train supports whose target
    equals the query target are kept with a probability calibrated (bisection) so
    the train target-in-support rate matches test.

Writes to ``<data_root>/<dataset>/collab/stage2/``:
  support_rows_<split>.npy     [n_rows, k] train-row indices
  rationale_emb_train.npy      [n_train, d] mean-centred, normalised rationale embeddings
  own_rationale_{train,valid}.npy  each row's own rationale (aux-loss target; never a model input)
  stats.json
"""

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import GroupKFold

from build_hyde_support import top_unique_users
from build_support import SasrecEncoder, find_csv, forbidden_rows, load_split
from eval_support import centre, load_rationales, reason_embeddings
from latent_hyde import history_text, predict, train_predictor
from tune_latent import embed_text, text_cache_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CDs_and_Vinyl")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--rationale-tag", default="v5")
    parser.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--sasrec-ckpt", type=Path, default=Path(
        "/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/baselines/results/sasrec/"
        "CDs_and_Vinyl_5_2015-10-2018-11/models/SASRec/2026-09-09_00-49-25-209512.pkl"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--extra", type=int, default=50, help="Backfill candidates kept for balancing.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def balance(cands: list, query_targets: np.ndarray, train_items: np.ndarray, k: int, ratio: float, seed: int,
            counts: dict = None) -> list:
    rng = random.Random(seed)
    out = []
    for q, rows in enumerate(cands):
        kept, dropped = [], []
        for j in rows:
            (kept if train_items[j] != query_targets[q] or rng.random() < ratio else dropped).append(j)
        if len(kept) < k:
            # Too few non-target candidates (popular target): refill with dropped ones.
            if counts is not None:
                counts["refilled"] = counts.get("refilled", 0) + 1
            kept = sorted(kept + dropped[:k - len(kept)], key=rows.index)
        out.append(kept[:k])
    return out


def hit_rate(rows: list, query_targets: np.ndarray, train_items: np.ndarray) -> float:
    return float(np.mean([query_targets[q] in set(train_items[r]) for q, r in enumerate(rows)]))


def main() -> None:
    args = parse_args()
    collab = args.data_root / args.dataset / "collab"
    out_dir = collab / "stage2"
    out_dir.mkdir(exist_ok=True)
    tuned = json.loads((collab / "tune_latent.json").read_text())
    pred_cfg = tuned["selected_predictor"]
    alpha = tuned["selected_alpha"]
    text_encoder = next(e for e in ["sentence-transformers/all-mpnet-base-v2", "Alibaba-NLP/gte-large-en-v1.5"]
                        if e.endswith(pred_cfg["text_encoder"]))
    print(f"predictor {pred_cfg}, alpha {alpha}")

    frames = {s: load_split(find_csv(args.data_root, args.dataset, s)) for s in ("train", "valid", "test")}
    train = frames["train"]
    raw_train = reason_embeddings(collab, "train", load_rationales(
        collab / f"rationales_facet_train_{args.rationale_tag}.jsonl", len(train)), args.rationale_tag, args.embedder)
    mean = raw_train.mean(0)
    rationale = {"train": centre(raw_train, mean)}
    for split in ("valid", "test"):
        recs = load_rationales(collab / f"rationales_facet_{split}_{args.rationale_tag}.jsonl", len(frames[split]))
        rationale[split] = centre(reason_embeddings(collab, split, recs, args.rationale_tag, args.embedder), mean)

    n_items = sum(1 for _ in next((args.data_root / args.dataset / "info").glob("*.txt")).open(encoding="utf-8"))
    encoder = SasrecEncoder(frames, n_items, SimpleNamespace(sasrec_ckpt=args.sasrec_ckpt))
    sasrec = {s: encoder.history_vectors(f) for s, f in frames.items()}
    text = {s: embed_text(history_text(f), collab / "hyp_cache" / text_cache_name(text_encoder, s), text_encoder)
            for s, f in frames.items()}
    x = {s: np.concatenate([sasrec[s], text[s]], 1).astype(np.float32) for s in frames}
    cfg = SimpleNamespace(hidden=pred_cfg["hidden"], epochs=args.epochs, batch_size=512, lr=1e-3,
                          temperature=pred_cfg["temperature"], seed=args.seed)

    # Predicted rationale vectors: full-train predictor for valid/test, cross-fitted for train.
    preds = {}
    model, _ = train_predictor(x["train"], rationale["train"], x["valid"], rationale["valid"], cfg)
    for split in ("valid", "test"):
        preds[split] = predict(model, x[split])
    preds["train"] = np.zeros_like(rationale["train"])
    folds = GroupKFold(n_splits=args.folds).split(x["train"], groups=train.user_id.to_numpy())
    for fold, (fit_idx, held_idx) in enumerate(folds):
        fold_model, mrr = train_predictor(x["train"][fit_idx], rationale["train"][fit_idx],
                                          x["valid"], rationale["valid"], cfg)
        preds["train"][held_idx] = predict(fold_model, x["train"][held_idx])
        print(f"fold {fold}: valid mrr {mrr:.4f}")

    users, train_items = train.user_id.to_numpy(), train.item_id.to_numpy()
    by_user = {}
    for j, user in enumerate(users):
        by_user.setdefault(user, []).append(j)
    cands = {}
    for split, frame in frames.items():
        forbid = forbidden_rows(frame, split, train, by_user)
        rows = []
        for start in range(0, len(frame), 1024):
            scores = (alpha * (preds[split][start:start + 1024] @ rationale["train"].T)
                      + (1 - alpha) * (sasrec[split][start:start + 1024] @ sasrec["train"].T))
            rows += [top_unique_users(s, forbid[start + off], users, args.k + args.extra, 400)
                     for off, s in enumerate(scores)]
        cands[split] = rows

    targets = {s: f.item_id.to_numpy() for s, f in frames.items()}
    final = {s: [r[:args.k] for r in cands[s]] for s in ("valid", "test")}
    test_hit = hit_rate(final["test"], targets["test"], train_items)
    lo, hi = 0.0, 1.0
    for _ in range(12):
        mid = (lo + hi) / 2
        hit = hit_rate(balance(cands["train"], targets["train"], train_items, args.k, mid, args.seed),
                       targets["train"], train_items)
        lo, hi = (mid, hi) if hit < test_hit else (lo, mid)
    ratio = (lo + hi) / 2
    counts = {}
    final["train"] = balance(cands["train"], targets["train"], train_items, args.k, ratio, args.seed, counts)

    stats = {"predictor": pred_cfg, "alpha": alpha, "balance_ratio": ratio,
             "train_rows_refilled": counts.get("refilled", 0),
             "target_hit_before": {s: hit_rate([r[:args.k] for r in cands[s]], targets[s], train_items) for s in cands},
             "target_hit_after": {s: hit_rate(final[s], targets[s], train_items) for s in final}}
    for split, rows in final.items():
        assert all(len(r) == args.k for r in rows), f"{split}: some rows have fewer than k supports"
        np.save(out_dir / f"support_rows_{split}.npy", np.array(rows, dtype=np.int64))
    np.save(out_dir / "rationale_emb_train.npy", rationale["train"].astype(np.float32))
    for split in ("train", "valid"):
        np.save(out_dir / f"own_rationale_{split}.npy", rationale[split].astype(np.float32))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
