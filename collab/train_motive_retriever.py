#!/usr/bin/env python3
"""Train a retriever: user history -> motive prototypes (collab/build_motive_corpus.py).

Every row has 0-2 motive prototypes (labels from its own rationale; never inputs).
Features use only the history and past data:
  sasrec        SASRec session embedding of the history (128)
  text          gte-large embedding of the history titles (1024)
  item_profile  mean over history items of the motive distribution of all train
                purchases OF that item, i.e. why people buy these items (P)
  user_past     motive distribution of the user's own EARLIER train rows (P); for a
                train row only rows with a shorter history, for valid/test all of
                the user's train rows (time split)
Models: popularity prior, item_profile alone, user_past alone, and an MLP over all
features trained with multi-positive softmax cross-entropy over all prototypes
(early stopping on valid MRR). --head picks how prototypes are represented:
  cls          a learned weight vector per prototype (plain classifier)
  text_frozen  the frozen Qwen3 embedding of the prototype's canonical motive text;
               score = cos(query, text) / tau (multi-positive InfoNCE, learnable tau)
  text         the text embedding through a learned projection; cosine / tau
  text_id      learned projection of the text plus a learned per-prototype offset
Metrics (rows with >= 1 motive): Recall@k = share of true motives in the top k,
MRR of the first true motive, Soft@5 = per true motive, best cosine of the top-5
retrieved prototype texts to its text (Qwen3 space; partial credit for near-duplicates).
Writes ``<data_root>/<dataset>/collab/motive_corpus/retriever/``.
"""

import argparse
import collections
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

from build_support import SasrecEncoder, find_csv, load_split

SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CDs_and_Vinyl")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--sasrec-ckpt", type=Path, default=Path(
        "/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/baselines/results/sasrec/"
        "CDs_and_Vinyl_5_2015-10-2018-11/models/SASRec/2026-09-09_00-49-25-209512.pkl"))
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head", choices=["cls", "text_frozen", "text", "text_id"], default="cls")
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--tau", type=float, default=0.05, help="Initial temperature for text heads (learned).")
    parser.add_argument("--crossfit-folds", type=int, default=0,
                        help="If > 1, also write out-of-fold train scores (folds grouped by user).")
    parser.add_argument("--prior-bias", action="store_true",
                        help="Text heads: add a learned per-prototype bias initialised to log train frequency "
                             "(cosine scores cannot express how common a motive is).")
    return parser.parse_args()


def normalise_rows(m: np.ndarray) -> np.ndarray:
    s = m.sum(1, keepdims=True)
    return np.divide(m, s, out=np.zeros_like(m), where=s > 0)


class Retriever(torch.nn.Module):
    def __init__(self, in_dim, n_proto, text_emb, args, log_prior=None):
        super().__init__()
        self.head = args.head
        out_dim = {"cls": n_proto, "text_frozen": text_emb.shape[1]}.get(args.head, args.proj_dim)
        self.query = torch.nn.Sequential(
            torch.nn.Linear(in_dim, args.hidden), torch.nn.GELU(), torch.nn.Dropout(args.dropout),
            torch.nn.Linear(args.hidden, out_dim))
        if args.head != "cls":
            self.register_buffer("text", torch.from_numpy(text_emb).float())
            self.log_tau = torch.nn.Parameter(torch.tensor(float(np.log(args.tau))))
        if args.head in ("text", "text_id"):
            self.proj = torch.nn.Linear(text_emb.shape[1], args.proj_dim)
        if args.head == "text_id":
            self.offset = torch.nn.Parameter(torch.zeros(n_proto, args.proj_dim))
        self.bias = None
        if args.head != "cls" and args.prior_bias:
            self.bias = torch.nn.Parameter(torch.as_tensor(log_prior, dtype=torch.float32))

    def forward(self, x):
        q = self.query(x)
        if self.head == "cls":
            return q
        proto = self.text if self.head == "text_frozen" else self.proj(self.text)
        if self.head == "text_id":
            proto = proto + self.offset
        tau = self.log_tau.exp().clamp(0.01, 1.0)
        scores = F.normalize(q, dim=-1) @ F.normalize(proto, dim=-1).T / tau
        return scores if self.bias is None else scores + self.bias


def metrics(scores: np.ndarray, labels: list, text_sim: np.ndarray) -> dict:
    keep = [i for i, l in enumerate(labels) if l]
    order = np.argsort(-scores[keep], axis=1)[:, :20]
    out = {f"recall@{k}": [] for k in (1, 5, 10, 20)}
    out["mrr"], out["soft@5"] = [], []
    for row, i in zip(order, keep):
        true = set(labels[i])
        for k in (1, 5, 10, 20):
            out[f"recall@{k}"].append(len(true & set(row[:k])) / len(true))
        ranks = np.where(np.isin(np.argsort(-scores[i]), list(true)))[0]
        out["mrr"].append(1 / (ranks.min() + 1))
        out["soft@5"].append(np.mean([text_sim[t, row[:5]].max() for t in true]))
    return {k: round(float(np.mean(v)), 4) for k, v in out.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    collab = args.data_root / args.dataset / "collab"
    corpus = collab / "motive_corpus"
    out = corpus / "retriever" / (args.head + ("_prior" if args.prior_bias and args.head != "cls" else ""))
    out.mkdir(parents=True, exist_ok=True)
    frames = {s: load_split(find_csv(args.data_root, args.dataset, s)) for s in SPLITS}
    labels = {s: json.loads((corpus / f"row_motives_{s}.json").read_text()) for s in SPLITS}
    n_proto = len(json.loads((corpus / "prototypes.json").read_text()))
    text_emb = np.load(corpus / "motive_text_emb.npy")
    text_sim = text_emb @ text_emb.T

    def multi_hot(ls):
        m = np.zeros((len(ls), n_proto), dtype=np.float32)
        for i, l in enumerate(ls):
            m[i, l] = 1.0
        return m
    y = {s: multi_hot(labels[s]) for s in SPLITS}
    train = frames["train"]

    # Item motive profile: motive distribution of all train purchases of each item.
    item_counts = collections.defaultdict(lambda: np.zeros(n_proto, dtype=np.float32))
    for item, row in zip(train.item_id, y["train"]):
        item_counts[int(item)] += row
    item_prof = {k: v / v.sum() for k, v in item_counts.items() if v.sum() > 0}

    def item_profile(frame):
        feats = np.zeros((len(frame), n_proto), dtype=np.float32)
        for i, hist in enumerate(frame.history):
            vecs = [item_prof[h] for h in hist if h in item_prof]
            if vecs:
                feats[i] = np.mean(vecs, 0)
        return feats

    # User's past motives: earlier train rows only.
    user_rows = collections.defaultdict(list)
    for j, (user, hist) in enumerate(zip(train.user_id, train.history)):
        user_rows[user].append((len(hist), j))

    def user_past(frame, split):
        feats = np.zeros((len(frame), n_proto), dtype=np.float32)
        for i, (user, hist) in enumerate(zip(frame.user_id, frame.history)):
            rows = [j for L, j in user_rows.get(user, []) if split != "train" or L < len(hist)]
            if rows:
                feats[i] = y["train"][rows].sum(0)
        return normalise_rows(feats)

    n_items = sum(1 for _ in next((args.data_root / args.dataset / "info").glob("*.txt")).open(encoding="utf-8"))
    sasrec = SasrecEncoder(frames, n_items, SimpleNamespace(sasrec_ckpt=args.sasrec_ckpt))
    feats = {}
    for s, f in frames.items():
        text = np.load(collab / "hyp_cache" / ("hist_gte-large-en-v1.5.npy" if s == "train"
                                               else f"hist_gte-large-en-v1.5_{s}.npy"))
        feats[s] = {"sasrec": sasrec.history_vectors(f), "text": text,
                    "item_profile": item_profile(f), "user_past": user_past(f, s)}
        print(f"{s}: rows with item profile {np.mean(feats[s]['item_profile'].sum(1) > 0):.3f}, "
              f"with user past {np.mean(feats[s]['user_past'].sum(1) > 0):.3f}", flush=True)

    results = {}
    pop = y["train"].sum(0)
    for s in ("valid", "test"):
        n = len(frames[s])
        results[f"{s}/popularity"] = metrics(np.tile(pop, (n, 1)), labels[s], text_sim)
        results[f"{s}/item_profile"] = metrics(feats[s]["item_profile"] + 1e-6 * pop / pop.max(), labels[s], text_sim)
        results[f"{s}/user_past"] = metrics(feats[s]["user_past"] + 1e-6 * pop / pop.max(), labels[s], text_sim)

    # MLP over all features.
    keys = ["sasrec", "text", "item_profile", "user_past"]
    x = {s: torch.from_numpy(np.concatenate([feats[s][k] for k in keys], 1)).float() for s in SPLITS}
    mu, sd = x["train"].mean(0), x["train"].std(0) + 1e-6
    x = {s: ((v - mu) / sd).to(device) for s, v in x.items()}
    log_prior = np.log((y["train"].sum(0) + 1) / (y["train"].sum() + n_proto))
    y_soft = torch.from_numpy(normalise_rows(y["train"])).to(device)

    def fit(train_rows: np.ndarray):
        """Train on the given train rows (those with >= 1 motive); early stopping on valid MRR."""
        rows = torch.from_numpy(train_rows[y["train"][train_rows].sum(1) > 0]).to(device)
        xt, yt = x["train"][rows], y_soft[rows]
        model = Retriever(xt.shape[1], n_proto, text_emb, args, log_prior).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best, best_state = -1.0, None
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(len(xt), device=device)
            for start in range(0, len(perm), args.batch_size):
                idx = perm[start:start + args.batch_size]
                loss = -(yt[idx] * F.log_softmax(model(xt[idx]), dim=1)).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                valid_scores = model(x["valid"]).cpu().numpy()
            mrr = metrics(valid_scores, labels["valid"], text_sim)["mrr"]
            if mrr > best:
                best, best_state = mrr, {k: v.clone() for k, v in model.state_dict().items()}
            if epoch % 5 == 4:
                print(f"  epoch {epoch + 1}: loss {loss.item():.3f}, valid mrr {mrr:.4f}", flush=True)
        model.load_state_dict(best_state)
        model.eval()
        return model, best_state

    model, best_state = fit(np.arange(len(frames["train"])))
    with torch.no_grad():
        for s in ("valid", "test"):
            scores = model(x[s]).cpu().numpy()
            results[f"{s}/mlp"] = metrics(scores, labels[s], text_sim)
            np.save(out / f"scores_{s}.npy", scores.astype(np.float16))
    torch.save({"state_dict": best_state, "mu": mu, "sd": sd, "keys": keys, "args": vars(args)}, out / "mlp.pt")

    # Cross-fitted train scores: each train row is scored by a model that never saw its label
    # (folds grouped by user), so train-time supports are as good as test-time ones.
    if args.crossfit_folds > 1:
        oof = np.zeros((len(frames["train"]), n_proto), dtype=np.float32)
        folds = GroupKFold(n_splits=args.crossfit_folds).split(np.zeros(len(oof)), groups=train.user_id.to_numpy())
        for fold, (fit_rows, held) in enumerate(folds):
            fold_model, _ = fit(fit_rows)
            with torch.no_grad():
                oof[held] = fold_model(x["train"][torch.from_numpy(held).to(device)]).cpu().numpy()
            print(f"fold {fold}: done", flush=True)
        results["train_oof/mlp"] = metrics(oof, labels["train"], text_sim)
        np.save(out / "scores_train.npy", oof.astype(np.float16))

    table = pd.DataFrame(results).T
    print(table.to_string())
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
