#!/usr/bin/env python3
"""Retrieve collaborative support samples (whose rationales are reused) for every row.

The pool is the train split. Retrieval is keyed on the query's history only, so
it is identical at train and test time. With ``--encoder sasrec`` (default):
  user-side  train samples whose SASRec session embedding is closest to the
             query's, at most one per user
  item-side  SASRec's top next-item candidates for the query; for each, the
             train sample with that target whose history is closest
``--encoder svd`` instead uses recency-weighted SVD item factors and item-item
co-occurrence (no trained model needed).

Train queries never retrieve themselves or the same user's later samples (whose
history contains the query target). As in IntuRec's target-aware candidate
balancing, supports whose target equals the query target are kept in train with
probability ratio = test_hit / train_hit, and backfilled otherwise.

Writes ``<data_root>/<dataset>/collab/support_<encoder>/support_<split>.json`` (one entry per CSV
row) and ``support_stats.json``.
"""

import argparse
import io
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CDs_and_Vinyl")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--k-user", type=int, default=10)
    parser.add_argument("--k-item", type=int, default=10)
    parser.add_argument("--encoder", choices=["sasrec", "svd"], default="sasrec")
    parser.add_argument("--sasrec-ckpt", type=Path, default=Path(
        "/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/baselines/results/sasrec/"
        "CDs_and_Vinyl_5_2015-10-2018-11/models/SASRec/2026-09-09_00-49-25-209512.pkl"),
        help="Cornac SASRec checkpoint (items keyed by ASIN).")
    parser.add_argument("--dim", type=int, default=64, help="SVD factors for history vectors (--encoder svd).")
    parser.add_argument("--decay", type=float, default=0.8, help="Recency decay over history positions.")
    parser.add_argument("--candidates", type=int, default=300,
                        help="Nearest pool rows scanned before filtering/deduplication.")
    parser.add_argument("--balance-ratio", type=float, default=None,
                        help="Keep-probability for target-matching train supports (default: test_hit/train_hit).")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def find_csv(data_root: Path, dataset: str, split: str) -> Path:
    matches = sorted((data_root / dataset / split).glob(f"{dataset}*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {dataset}*.csv in {data_root / dataset / split}")
    return matches[0]


def load_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["history"] = frame.history_item_id.map(eval)
    return frame


def item_factors(train: pd.DataFrame, n_items: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """SVD item factors and dense item-item cosine from train user-item sets."""
    user_items: dict[str, set] = {}
    for user, hist, target in zip(train.user_id, train.history, train.item_id):
        user_items.setdefault(user, set()).update(hist, [target])
    rows, cols = [], []
    for u, items in enumerate(user_items.values()):
        rows += list(items)
        cols += [u] * len(items)
    mat = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n_items, len(user_items)))
    norms = np.sqrt(np.asarray(mat.multiply(mat).sum(1)).ravel()) + 1e-8
    normed = csr_matrix(mat.multiply(1 / norms[:, None]))
    u, s, _ = svds(normed, k=dim)
    factors = (u * s).astype(np.float32)
    item_sim = (normed @ normed.T).toarray().astype(np.float32)
    np.fill_diagonal(item_sim, 0.0)
    return factors, item_sim


def history_weights(hist: list, decay: float) -> np.ndarray:
    return decay ** np.arange(len(hist) - 1, -1, -1, dtype=np.float32)


def history_vectors(frame: pd.DataFrame, factors: np.ndarray, decay: float) -> np.ndarray:
    vecs = np.stack([history_weights(h, decay) @ factors[h] for h in frame.history])
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)


class SvdEncoder:
    """Recency-weighted SVD history vectors; item scores from item-item co-occurrence."""

    def __init__(self, train: pd.DataFrame, n_items: int, args):
        self.factors, self.item_sim = item_factors(train, n_items, args.dim)
        self.decay = args.decay

    def history_vectors(self, frame: pd.DataFrame) -> np.ndarray:
        return history_vectors(frame, self.factors, self.decay)

    def item_scores(self, frame: pd.DataFrame, rows: range, qvecs: np.ndarray) -> np.ndarray:
        return np.stack([history_weights(frame.history.iat[i], self.decay) @ self.item_sim[frame.history.iat[i]]
                         for i in rows])


class SasrecEncoder:
    """Last-position SASRec hidden state; item scores = SASRec next-item logits."""

    def __init__(self, frames: dict, n_items: int, args):
        import torch
        import torch.storage
        # The checkpoint was pickled on GPU.
        torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        with args.sasrec_ckpt.open("rb") as fin:
            model = pickle.load(fin)
        model.model.eval()
        self.torch, self.model = torch, model
        asin_of = {}
        for frame in frames.values():
            for ids, asins in zip(frame.history, frame.item_asins.map(eval)):
                asin_of.update(zip(ids, asins))
            asin_of.update(zip(frame.item_id, frame.item_asin))
        self.to_cornac = np.full(n_items, -1, dtype=np.int64)
        for item, asin in asin_of.items():
            idx = model.iid_map.get(asin)
            if idx is not None and idx < model.num_items:
                self.to_cornac[item] = idx
        known = self.to_cornac >= 0
        net = model.model
        with torch.no_grad():
            emb = net.item_emb.weight.detach().cpu().numpy()
            bias = (net.item_biases.weight.detach().cpu().numpy().ravel()
                    if hasattr(net, "item_biases") else np.zeros(len(emb), np.float32))
        self.item_emb = np.zeros((n_items, emb.shape[1]), np.float32)
        self.item_bias = np.full(n_items, -np.inf, np.float32)
        self.item_emb[known] = emb[self.to_cornac[known]]
        self.item_bias[known] = bias[self.to_cornac[known]]
        self.raw = {}
        print(f"SASRec: {known.sum()}/{n_items} items mapped")

    def history_vectors(self, frame: pd.DataFrame) -> np.ndarray:
        pad, max_len = self.model.pad_idx, self.model.max_len
        seqs = []
        for hist in frame.history:
            ids = [int(self.to_cornac[i]) for i in hist if self.to_cornac[i] >= 0][-max_len:]
            seqs.append([pad] * (max_len - len(ids)) + ids)
        out = []
        with self.torch.no_grad():
            for start in range(0, len(seqs), 2048):
                batch = self.torch.tensor(seqs[start:start + 2048], dtype=self.torch.long)
                out.append(self.model.model._encode(batch).cpu().numpy())
        # A history whose items are all unknown to SASRec encodes to NaN; give it a zero vector.
        raw = np.nan_to_num(np.concatenate(out).astype(np.float32))
        self.raw[id(frame)] = raw
        return raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)

    def item_scores(self, frame: pd.DataFrame, rows: range, qvecs: np.ndarray) -> np.ndarray:
        return self.raw[id(frame)][rows.start:rows.stop] @ self.item_emb.T + self.item_bias


def forbidden_rows(frame: pd.DataFrame, split: str, train: pd.DataFrame, by_user: dict) -> list:
    """Train rows a query may not retrieve: itself and the same user's later samples."""
    if split != "train":
        return [set() for _ in range(len(frame))]
    out = []
    for idx, (user, hist) in enumerate(zip(frame.user_id, frame.history)):
        out.append({j for j in by_user[user] if len(train.history.iat[j]) >= len(hist)} | {idx})
    return out


def retrieve(frame, split, train, qvecs, pvecs, encoder, pool_by_item, by_user, args):
    forbid = forbidden_rows(frame, split, train, by_user)
    pool_users = train.user_id.to_numpy()
    extra = args.k_user + args.k_item
    results = []
    for start in range(0, len(frame), 1024):
        sims = qvecs[start:start + 1024] @ pvecs.T
        item_scores = encoder.item_scores(frame, range(start, min(start + 1024, len(frame))), qvecs)
        top = np.argpartition(-sims, args.candidates, axis=1)[:, :args.candidates]
        for off, cand in enumerate(top):
            idx = start + off
            row_sims = sims[off]
            user_side, seen_users = [], set()
            for j in cand[np.argsort(-row_sims[cand])]:
                if j in forbid[idx] or pool_users[j] in seen_users:
                    continue
                seen_users.add(pool_users[j])
                user_side.append(int(j))
                if len(user_side) == args.k_user + extra:
                    break

            hist = frame.history.iat[idx]
            scores = item_scores[off].copy()
            scores[hist] = -np.inf
            item_side = []
            for item in np.argsort(-scores)[:4 * (args.k_item + extra)]:
                pool = [j for j in pool_by_item.get(int(item), ()) if j not in forbid[idx]]
                if pool:
                    item_side.append(int(pool[int(np.argmax(row_sims[pool]))]))
                    if len(item_side) == args.k_item + extra:
                        break
            results.append({"user_cand": user_side, "item_cand": item_side,
                            "sims": {j: float(row_sims[j]) for j in user_side + item_side}})
    return results


def hit_rate(frame, train, cands, key, k) -> float:
    targets = train.item_id.to_numpy()
    return float(np.mean([frame.item_id.iat[i] in set(targets[c[key][:k]]) for i, c in enumerate(cands)]))


def finalize(frame, split, train, cands, ratio, args, rng) -> list:
    targets = train.item_id.to_numpy()
    out = []
    for idx, cand in enumerate(cands):
        entry = {}
        for key, k in (("user_cand", args.k_user), ("item_cand", args.k_item)):
            kept = []
            for j in cand[key]:
                if split == "train" and targets[j] == frame.item_id.iat[idx] and rng.random() >= ratio:
                    continue
                kept.append(j)
                if len(kept) == k:
                    break
            entry[key.replace("_cand", "_support")] = [
                {"row": j, "item_id": int(targets[j]), "sim": round(cand["sims"][j], 4),
                 "same_user": bool(train.user_id.iat[j] == frame.user_id.iat[idx])}
                for j in kept]
        out.append(entry)
    return out


def main() -> None:
    args = parse_args()
    frames = {s: load_split(find_csv(args.data_root, args.dataset, s)) for s in SPLITS}
    train = frames["train"]
    info = next((args.data_root / args.dataset / "info").glob("*.txt"))
    n_items = sum(1 for _ in info.open(encoding="utf-8"))

    encoder = (SasrecEncoder(frames, n_items, args) if args.encoder == "sasrec"
               else SvdEncoder(train, n_items, args))
    pvecs = encoder.history_vectors(train)
    pool_by_item, by_user = {}, {}
    for j, (item, user) in enumerate(zip(train.item_id, train.user_id)):
        pool_by_item.setdefault(int(item), []).append(j)
        by_user.setdefault(user, []).append(j)

    cands, stats = {}, {"args": {k: str(v) for k, v in vars(args).items()}, "hit": {}}
    for split, frame in frames.items():
        qvecs = encoder.history_vectors(frame)
        cands[split] = retrieve(frame, split, train, qvecs, pvecs, encoder, pool_by_item, by_user, args)
        stats["hit"][split] = {
            "user": hit_rate(frame, train, cands[split], "user_cand", args.k_user),
            "item": hit_rate(frame, train, cands[split], "item_cand", args.k_item),
        }
        print(f"{split}: target-in-support rate before balancing {stats['hit'][split]}")

    # Backfilled supports can share the dropped target, so test_hit/train_hit
    # undershoots; bisect the keep-probability until train matches test.
    ratios = {}
    for key in ("user", "item"):
        if args.balance_ratio is not None:
            ratios[key] = args.balance_ratio
            continue
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = (lo + hi) / 2
            trial = finalize(train, "train", train, cands["train"], mid, args, random.Random(args.seed))
            hit = np.mean([t in {s["item_id"] for s in e[f"{key}_support"]}
                           for t, e in zip(train.item_id, trial)])
            lo, hi = (mid, hi) if hit < stats["hit"]["test"][key] else (lo, mid)
        ratios[key] = (lo + hi) / 2
    print(f"balance ratios: {ratios}")
    stats["balance_ratio"] = ratios

    rng = random.Random(args.seed)
    out_dir = args.data_root / args.dataset / "collab" / f"support_{args.encoder}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        supports = finalize(frame, split, train, cands[split], 1.0, args, rng) if split != "train" else None
        if split == "train":
            # Balance each side with its own ratio.
            user = finalize(frame, split, train, cands[split], ratios["user"], args, rng)
            item = finalize(frame, split, train, cands[split], ratios["item"], args, rng)
            supports = [{"user_support": u["user_support"], "item_support": i["item_support"]}
                        for u, i in zip(user, item)]
        targets = frame.item_id.to_numpy()
        stats.setdefault("hit_after", {})[split] = {
            side: float(np.mean([targets[i] in {s["item_id"] for s in e[f"{side}_support"]}
                                 for i, e in enumerate(supports)]))
            for side in ("user", "item")}
        stats.setdefault("same_user_share", {})[split] = float(np.mean(
            [s["same_user"] for e in supports for s in e["user_support"] + e["item_support"]]))
        with (out_dir / f"support_{split}.json").open("w", encoding="utf-8") as fout:
            json.dump(supports, fout)
        print(f"{split}: after balancing {stats['hit_after'][split]}, "
              f"same-user share {stats['same_user_share'][split]:.3f}")
    with (out_dir / "support_stats.json").open("w", encoding="utf-8") as fout:
        json.dump(stats, fout, indent=2)


if __name__ == "__main__":
    main()
