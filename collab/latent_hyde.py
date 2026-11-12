#!/usr/bin/env python3
"""Latent HyDE: predict the purchase-rationale embedding from the history, then
retrieve the train rationales nearest to the prediction.

A small MLP maps history features (SASRec session embedding, optionally plus a
text embedding of the history titles) into the mean-centred rationale space.
It is trained with InfoNCE on train rows (positive: the row's own rationale;
negatives: the other rationales in the batch), so the prediction must be
discriminative rather than a generic "music lover" vector. Model selection uses
valid MRR of the own rationale among all valid rationales.

Writes ``<data_root>/<dataset>/collab/support_latent/support_<split>.json``:
  latent_support         top-k train rows by predicted-vs-train rationale cosine
  latent_hybrid_support  top-k by alpha * that cosine + (1 - alpha) * SASRec cosine
(one row per user). Train rows are not written: the predictor has seen their
own rationales, so train supports need cross-fitting before stage-2 training.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from build_hyde_support import top_unique_users
from build_support import SasrecEncoder, find_csv, load_split
from eval_support import centre, load_rationales, rank_metrics, reason_embeddings
from hypothesis_test import embed_cached


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CDs_and_Vinyl")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--rationale-tag", default="v5")
    parser.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--sasrec-ckpt", type=Path, default=Path(
        "/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/baselines/results/sasrec/"
        "CDs_and_Vinyl_5_2015-10-2018-11/models/SASRec/2026-09-09_00-49-25-209512.pkl"))
    parser.add_argument("--features", nargs="+", default=["sasrec", "sasrec+text"])
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def history_text(frame) -> list:
    return ["; ".join(eval(t)) for t in frame.history_item_title]


def predict(model, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = F.normalize(model(torch.from_numpy(x)), dim=-1)
    return out.numpy()


def train_predictor(x_train, y_train, x_valid, y_valid, args) -> tuple[torch.nn.Module, float]:
    torch.manual_seed(args.seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], args.hidden), torch.nn.GELU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(args.hidden, y_train.shape[1]))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    xt, yt = torch.from_numpy(x_train), torch.from_numpy(y_train)
    best, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for start in range(0, len(perm), args.batch_size):
            idx = perm[start:start + args.batch_size]
            logits = F.normalize(model(xt[idx]), dim=-1) @ yt[idx].T / args.temperature
            loss = F.cross_entropy(logits, torch.arange(len(idx)))
            opt.zero_grad()
            loss.backward()
            opt.step()
        mrr = rank_metrics(predict(model, x_valid), y_valid)["rank_mrr"]
        if mrr > best:
            best, best_state = mrr, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 4:
            print(f"    epoch {epoch + 1:3d}  loss {loss.item():.3f}  valid mrr {mrr:.4f}")
    model.load_state_dict(best_state)
    return model, best


def main() -> None:
    args = parse_args()
    collab = args.data_root / args.dataset / "collab"
    cache = collab / "hyp_cache"
    frames = {s: load_split(find_csv(args.data_root, args.dataset, s)) for s in ("train", "valid", "test")}
    train = frames["train"]

    raw_train = reason_embeddings(collab, "train", load_rationales(
        collab / f"rationales_facet_train_{args.rationale_tag}.jsonl", len(train)), args.rationale_tag, args.embedder)
    mean = raw_train.mean(0)
    targets = {"train": centre(raw_train, mean)}
    for split in ("valid", "test"):
        recs = load_rationales(collab / f"rationales_facet_{split}_{args.rationale_tag}.jsonl", len(frames[split]))
        targets[split] = centre(reason_embeddings(collab, split, recs, args.rationale_tag, args.embedder), mean)

    n_items = sum(1 for _ in next((args.data_root / args.dataset / "info").glob("*.txt")).open(encoding="utf-8"))
    encoder = SasrecEncoder(frames, n_items, SimpleNamespace(sasrec_ckpt=args.sasrec_ckpt))
    sasrec = {s: encoder.history_vectors(f) for s, f in frames.items()}
    text = {s: embed_cached(history_text(f), cache / ("hist.npy" if s == "train" else f"hist_{s}.npy"), args.embedder)
            for s, f in frames.items()}

    results, models = {}, {}
    for feat in args.features:
        x = {s: (np.concatenate([sasrec[s], text[s]], 1) if feat == "sasrec+text" else sasrec[s]).astype(np.float32)
             for s in frames}
        print(f"[{feat}] training on {len(x['train'])} rows, input dim {x['train'].shape[1]}")
        model, best = train_predictor(x["train"], targets["train"], x["valid"], targets["valid"], args)
        test = rank_metrics(predict(model, x["test"]), targets["test"])
        results[feat] = {"valid_mrr": best, **{f"test_{k}": v for k, v in test.items()}}
        models[feat] = (model, x)
        print(f"[{feat}] direct prediction -> own rationale: valid mrr {best:.4f}, "
              f"test mrr {test['rank_mrr']:.4f}, test r@10 {test['rank_r10']:.4f}")
    chance = np.mean(1 / np.arange(1, len(frames["test"]) + 1))
    print(f"chance mrr on test: {chance:.4f}")

    best_feat = max(results, key=lambda f: results[f]["valid_mrr"])
    model, x = models[best_feat]
    print(f"retrieving supports with [{best_feat}]")
    users, items = train.user_id.to_numpy(), train.item_id.to_numpy()
    out_dir = collab / "support_latent"
    out_dir.mkdir(exist_ok=True)
    for split in ("valid", "test"):
        pred = predict(model, x[split])
        supports = []
        for start in range(0, len(pred), 1024):
            latent = pred[start:start + 1024] @ targets["train"].T
            hybrid = args.alpha * latent + (1 - args.alpha) * (sasrec[split][start:start + 1024] @ sasrec["train"].T)
            for off in range(len(latent)):
                idx = start + off
                entry = {}
                for key, scores in (("latent_support", latent[off]), ("latent_hybrid_support", hybrid[off])):
                    rows = top_unique_users(scores, set(), users, args.k, 300)
                    entry[key] = [{"row": j, "item_id": int(items[j]), "sim": round(float(scores[j]), 4),
                                   "same_user": bool(users[j] == frames[split].user_id.iat[idx])} for j in rows]
                supports.append(entry)
        with (out_dir / f"support_{split}.json").open("w", encoding="utf-8") as fout:
            json.dump(supports, fout)
    with (out_dir / "predictor_results.json").open("w", encoding="utf-8") as fout:
        json.dump({"results": results, "selected": best_feat, "chance_mrr": chance}, fout, indent=2)


if __name__ == "__main__":
    main()
