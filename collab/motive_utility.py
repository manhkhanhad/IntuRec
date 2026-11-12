#!/usr/bin/env python3
"""Utility judge for retrieved motives: how much does a motive help predict the true target?

Judge = a trained O2 model (supports = the row's own motive prototypes; it has learned to read motives).
For each row and candidate motive m (fed as the 2-slot support [m, m], O2's format for 1-motive rows):
    u(m) = log p(target title | history, m) - log p(target title | history, null)
null = "No specific motive." (what O2 saw for rows without motives). Teacher-forced, no decoding.
Candidates per row: null, the true pair (O2's exact training input), each true motive alone,
the retriever's top-10 (motive_corpus/retriever/<scores>/scores_<split>.npy), and one random motive.

Writes <out>/utility_<split>.npz (per-row candidate ids, kinds, log-likelihoods) and
<out>/utility_<split>.json (summary). Summary:
  - retrieval quality as share of the oracle gain: mean u(top-1 predicted) / mean u(true pair), etc.
  - label audit: u of predicted motives that are true vs not; AUC of "is true" for "u > 0";
    mean u and label precision of the most frequently predicted motives.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.CollabRecDataset import CollabRecDataset, user_fold  # noqa: E402
from model.CollabRecF2 import CollabRecF2  # noqa: E402

NULL, PAIR, TRUE, PRED, RAND = 0, 1, 2, 3, 4
KIND = {NULL: "null", PAIR: "true_pair", TRUE: "true", PRED: "pred", RAND: "random"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="O2 run folder (best checkpoint).")
    p.add_argument("--dataset", default="Toys_and_Games")
    p.add_argument("--data_path", default="./data")
    p.add_argument("--split", default="test")
    p.add_argument("--scores", default="cls")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--fold", default="", help="'i/n': score only rows of users IN fold i (held out by the judge).")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="", help="Default: <ckpt>/motive_utility")
    return p.parse_args()


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    s = np.concatenate([pos, neg])
    ranks = s.argsort().argsort() + 1.0
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    args = parse_args()
    out_dir = args.out or os.path.join(args.ckpt, "motive_utility")
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    model = CollabRecF2.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16).to(device).eval()
    assert model.support_mode == "text" and model.readout == "inject"

    ds = CollabRecDataset(args.data_path, args.dataset, tokenizer, support_dir=None, support_k=0,
                          dataset_type=args.split, support_mode="text")
    collab = os.path.join(args.data_path, args.dataset, "collab")
    texts, _ = CollabRecDataset.motive_texts(collab, args.split)  # prototypes + "No specific motive."
    null_id = len(texts) - 1
    motive_tokens = torch.tensor(ds.tokenize_texts(texts, 64))
    truth = json.load(open(os.path.join(collab, "motive_corpus", f"row_motives_{args.split}.json")))
    scores = np.load(os.path.join(collab, "motive_corpus", "retriever", args.scores,
                                  f"scores_{args.split}.npy")).astype(np.float32)
    pred = np.argsort(-scores, axis=1)[:, :args.topk]
    rng = np.random.default_rng(args.seed)
    pad = tokenizer.pad_token_id

    rows = list(range(len(ds.data)))
    if args.fold:
        import glob
        import pandas as pd
        i, n = map(int, args.fold.split("/"))
        frame = pd.read_csv(glob.glob(os.path.join(args.data_path, args.dataset, args.split, f"{args.dataset}*.csv"))[0])
        folds = user_fold(frame.user_id, n)
        rows = [r for r in rows if folds[r] == i]
    if args.max_rows:
        rows = rows[:args.max_rows]
    # Flat list of (row, kind, slot0, slot1, rank)
    items = []
    for r in rows:
        true = list(dict.fromkeys(truth[r]))
        items.append((r, NULL, null_id, null_id, -1))
        if true:
            items.append((r, PAIR, true[0], true[-1] if len(true) > 1 else true[0], -1))
            items += [(r, TRUE, m, m, -1) for m in true]
        items += [(r, PRED, int(m), int(m), k) for k, m in enumerate(pred[r])]
        m = int(rng.integers(0, null_id))
        items.append((r, RAND, m, m, -1))
    seqs = {}
    for r in rows:
        part1, part2, _ = ds.prompt_parts(ds.data[r])
        seqs[r] = (part1 + part2, len(part2))

    tag = args.split + (f"_fold{args.fold.replace('/', 'of')}" if args.fold else "")
    # Partial results survive a preemption (small: one float per item); items are deterministic (seeded).
    partial = os.path.join(out_dir, f"partial_{tag}.npz")
    ll, first = np.zeros(len(items), np.float32), 0
    if os.path.exists(partial):
        z = np.load(partial)
        if len(z["ll"]) == len(items):
            ll, first = z["ll"].copy(), int(z["done"])
            print(f"resuming from {partial}: {first}/{len(items)} items done", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for start in range(first, len(items), args.batch):
            chunk = items[start:start + args.batch]
            L = max(len(seqs[it[0]][0]) for it in chunk)
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            tgt_len = []
            for i, it in enumerate(chunk):
                s, n = seqs[it[0]]
                ids[i, L - len(s):] = torch.tensor(s)
                tgt_len.append(n)
            mask = (ids != pad).long()
            sup = motive_tokens[torch.tensor([[it[2], it[3]] for it in chunk])]
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device), support_tokens=sup.to(device))
            logp = torch.log_softmax(out.logits[:, :-1].float(), -1)
            tok = torch.gather(logp, 2, ids[:, 1:].to(device).unsqueeze(-1)).squeeze(-1)  # [B, L-1]
            for i, n in enumerate(tgt_len):
                ll[start + i] = tok[i, -n:].sum().item()  # target title + <|EndPredict|>
            if (start // args.batch) % 200 == 0:
                print(f"{start}/{len(items)} items, {time.time() - t0:.0f}s", flush=True)
                np.savez(partial + ".tmp.npz", ll=ll, done=start + len(chunk))
                os.replace(partial + ".tmp.npz", partial)

    rec = np.array([(it[0], it[1], it[2], it[4]) for it in items], dtype=np.int64)
    np.savez(os.path.join(out_dir, f"utility_{tag}.npz"), row=rec[:, 0], kind=rec[:, 1],
             motive=rec[:, 2], rank=rec[:, 3], ll=ll)

    # ---- summary ----
    null_ll = {r: l for (r, k, *_), l in zip(items, ll) if k == NULL}
    u = np.array([l - null_ll[it[0]] for it, l in zip(items, ll)])
    kind, row, motive, rank = rec[:, 1], rec[:, 0], rec[:, 2], rec[:, 3]
    has_true = np.array([bool(truth[r]) for r in row])
    is_true = np.array([m in set(truth[r]) for r, m in zip(row, motive)])
    S = {"ckpt": args.ckpt, "split": args.split, "rows": len(rows), "null_ll_mean": float(np.mean(list(null_ll.values())))}
    sel = lambda m: float(np.mean(u[m])) if m.any() else float("nan")  # noqa: E731
    S["u_true_pair"] = sel((kind == PAIR))
    S["u_true_single_mean"] = sel(kind == TRUE)
    S["u_random"] = sel(kind == RAND)
    S["u_pred_top1"] = sel((kind == PRED) & (rank == 0))
    S["u_pred_top10_mean"] = sel(kind == PRED)
    best = {}
    for r, k, l in zip(row, kind, u):
        if k == PRED:
            best[r] = max(best.get(r, -1e9), l)
    S["u_pred_best_of_top10"] = float(np.mean(list(best.values())))  # needs a perfect verifier
    S["share_of_oracle_gain"] = {k: S[k] / S["u_true_pair"] for k in
                                 ("u_pred_top1", "u_pred_top10_mean", "u_pred_best_of_top10", "u_random",
                                  "u_true_single_mean")}
    pm = kind == PRED
    S["label_audit"] = {
        "pred_label_precision@1": float(np.mean(is_true[pm & (rank == 0) & has_true])),
        "u_pred_is_true": sel(pm & is_true), "u_pred_not_true": sel(pm & ~is_true),
        "share_pred_with_u>0": {"is_true": sel_share(u, pm & is_true), "not_true": sel_share(u, pm & ~is_true)},
        "auc_u_true_vs_wrong_pred": auc(u[pm & is_true], u[pm & ~is_true]),  # P(u(true pred) > u(wrong pred))
    }
    freq = {}
    for m in motive[pm & (rank == 0)]:
        freq[m] = freq.get(m, 0) + 1
    protos = json.load(open(os.path.join(collab, "motive_corpus", "prototypes.json")))
    top = sorted(freq, key=freq.get, reverse=True)[:10]
    S["frequent_top1_motives"] = [
        {"id": int(m), "motive": protos[m]["motive"], "share_of_top1": freq[m] / int((pm & (rank == 0)).sum()),
         "label_precision": float(np.mean(is_true[pm & (rank == 0) & (motive == m)])),
         "u_mean": sel(pm & (motive == m)), "u_mean_when_true": sel(pm & (motive == m) & is_true)} for m in top]
    with open(os.path.join(out_dir, f"utility_{tag}.json"), "w") as f:
        json.dump(S, f, indent=2)
    if os.path.exists(partial):
        os.remove(partial)
    print(json.dumps(S, indent=2))


def sel_share(u, mask):
    return float(np.mean(u[mask] > 0)) if mask.any() else float("nan")


if __name__ == "__main__":
    main()
