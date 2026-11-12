#!/usr/bin/env python3
"""Ranking-aware motive utilities: score the target AND hard-negative items under each candidate motive.

For each row and each motive m in {null} + the retriever's top-10 (from the existing utility file of the
same judge), a frozen motive reader scores the target title and N hard negatives (collab/hard_negatives.py,
SASRec top-N) with supports [m, m]:
    ll[m, j] = log p(item_j + <|EndPredict|> | prompt, [m, m])   (j = 0 is the target), ntok[j] tokens.
Rewards derived later (collab/grpo_motive_select.py --reward):
    rank_lse:  ll_norm[m, 0] - logsumexp_j ll_norm[m, j]   (length-normalised, as the beam ranks titles)
The prompt + motive prefix runs once per motive; the N+1 continuations reuse its KV cache.

Judges (as collab/motive_utility.py): train rows of fold i are scored by the judge trained WITHOUT those
users (357943 / 357944), test rows by the full judge 351001. Saves partial results every --save_every rows
(resumable after preemption) and finally <out>/rank_utility_<tag>.npz with
row [n], motive [n, 11], rank [n, 11] (-1 = null), ll [n, 11, N+1], ntok [n, N+1].
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motive_reader as mr  # noqa: E402  (chdir's to the main checkout; CollabRec code lives there)

CK = "/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints/Toys_and_Games/CollabRec_oracle"
JUDGES = {
    "train0": (f"{CK}/text-micro2-oracle_own_motive-cq0.1_user+target+motive-fold0of2_k2_beta0_rs1_lr0.0003_job357943",
               "utility_train_fold0of2.npz", "train"),
    "train1": (f"{CK}/text-micro2-oracle_own_motive-cq0.1_user+target+motive-fold1of2_k2_beta0_rs1_lr0.0003_job357944",
               "utility_train_fold1of2.npz", "train"),
    "test": (f"{CK}/text-micro2-oracle_own_motive-cq0.1_user+target+motive_k2_beta0_rs1_lr0.0003_job351001",
             "utility_test.npz", "test"),
}
NULL = 881


def candidates(util_file):
    z = np.load(util_file)
    rows = np.unique(z["row"])
    pos = {r: i for i, r in enumerate(rows)}
    cand = np.full((len(rows), 10), -1, np.int64)
    s = z["kind"] == 3
    for r, k, m in zip(z["row"][s], z["rank"][s], z["motive"][s]):
        cand[pos[r], k] = m
    return rows, cand


@torch.no_grad()
def score_row(model, prompt, sup, items, device):
    """prompt [L] ids, sup [M, 2, len] support tokens, items list of id lists -> ll [M, J], ntok [J]."""
    M, J = sup.shape[0], len(items)
    ids = torch.tensor([prompt] * M, device=device)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), support_tokens=torch.as_tensor(sup, device=device),
                use_cache=True)
    first = torch.log_softmax(out.logits[:, -1].float(), -1)                       # [M, V]
    cache = out.past_key_values
    cache.batch_repeat_interleave(J)
    T = max(len(x) for x in items)
    pad = model.config.pad_token_id
    cont = torch.full((J, T), pad, dtype=torch.long, device=device)
    cmask = torch.zeros((J, T), dtype=torch.long, device=device)
    for j, x in enumerate(items):
        cont[j, :len(x)] = torch.tensor(x, device=device)
        cmask[j, :len(x)] = 1
    cont_b, cmask_b = cont.repeat(M, 1), cmask.repeat(M, 1)                     # [M*J, T], motive-major
    attn = torch.cat([torch.ones((M * J, len(prompt)), dtype=torch.long, device=device), cmask_b], 1)
    logits = model(input_ids=cont_b, attention_mask=attn, past_key_values=cache, use_cache=False).logits
    lp = torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, cont_b[:, 1:].unsqueeze(-1)).squeeze(-1)
    lp = (lp * cmask_b[:, 1:]).sum(1).view(M, J)
    lp0 = first.gather(1, cont[:, 0].unsqueeze(0).expand(M, J))                  # first item token
    ntok = cmask.sum(1)
    return (lp + lp0).cpu().numpy(), ntok.cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", choices=list(JUDGES), required=True)
    p.add_argument("--neg_dir", default="/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints/Toys_and_Games/"
                                        "CollabRec_GRPO_select/rank_reward")
    p.add_argument("--n_neg", type=int, default=20)
    p.add_argument("--limit", type=int, default=0, help="score only the first N rows (smoke tests)")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    judge, util_name, split = JUDGES[a.tag]
    device = a.device if torch.cuda.is_available() else "cpu"
    rows, cand = candidates(f"{judge}/motive_utility/{util_name}")
    if a.limit:
        rows, cand = rows[:a.limit], cand[:a.limit]
    neg = np.load(f"{a.neg_dir}/negatives_{split}.npy")[:, :a.n_neg]
    out = f"{a.neg_dir}/rank_utility_{a.tag}{'_limit' + str(a.limit) if a.limit else ''}.npz"
    part = out.replace(".npz", ".partial.npz")

    model, tok = mr.load_reader(judge, device)
    ds = mr.CollabRecDataset("./data", "Toys_and_Games", tok, support_dir=None, support_k=0, dataset_type=split,
                             reason_step=1, seed=0, support_mode="text")
    sup_table = ds.tokenize_texts(mr.motive_texts("Toys_and_Games"), 64)                 # [882, 64]
    titles = {}
    with open(next(iter(__import__("glob").glob("./data/Toys_and_Games/info/*.txt"))), encoding="utf-8") as f:
        for line in f:
            t, i = line.rstrip("\n").rsplit("\t", 1)
            titles[int(i)] = t
    enc = lambda title: ds.tokenizer.encode('"' + title + '"', bos=False, eos=False) + ds.end_token
    frame_items = [d["tgt_item"] for d in ds.data]                                     # '"title"'

    M = 11
    ll = np.full((len(rows), M, a.n_neg + 1), np.nan, np.float32)
    ntok = np.zeros((len(rows), a.n_neg + 1), np.int32)
    motive = np.concatenate([np.full((len(rows), 1), NULL), cand], 1)
    rank = np.concatenate([np.full((len(rows), 1), -1), np.tile(np.arange(10), (len(rows), 1))], 1)
    start = 0
    if os.path.exists(part):
        z = np.load(part)
        start = int(z["done"])
        ll[:start], ntok[:start] = z["ll"][:start], z["ntok"][:start]
        print(f"resumed at row {start}/{len(rows)}", flush=True)
    for i in range(start, len(rows)):
        r = int(rows[i])
        s = ds[r]
        keep = np.asarray(s["attention_mask"]) == 1
        if "labels" in s:
            keep &= np.asarray(s["labels"]) == -100
        prompt = np.asarray(s["input_ids"])[keep].tolist()
        items = [ds.tokenizer.encode(frame_items[r], bos=False, eos=False) + ds.end_token] + [enc(titles[int(j)]) for j in neg[r]]
        sup = sup_table[motive[i]][:, None, :].repeat(2, 1)                                # [11, 2, 64] = [m, m]
        ll[i], ntok[i] = score_row(model, prompt, sup, items, device)
        if (i + 1) % a.save_every == 0:
            np.savez(part, ll=ll, ntok=ntok, done=i + 1)
            print(f"{i + 1}/{len(rows)} rows", flush=True)
    np.savez(out, row=rows, motive=motive, rank=rank, ll=ll, ntok=ntok)
    if os.path.exists(part):
        os.remove(part)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
