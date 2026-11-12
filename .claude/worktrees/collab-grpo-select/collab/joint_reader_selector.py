#!/usr/bin/env python3
"""Joint motive-reader + motive-selector training (Atlas / RAG / RA-DIT style marginal likelihood).

Selector pi(m | history) over the retriever's top-10 motives + "no motive" (the phase-1 selector with
an extra null logit). Reader = CollabRec motive reader (init: O2 + InfoNCE run 351001), supports [m, m].
Per train row, candidate set S = {null} + top-(K-1) motives under the current selector; loss
    -log sum_{m in S} pi_S(m) * p(target | history, [m, m]),     pi_S = selector softmax renormalised on S.
The reader's gradient is weighted by the posterior p(m | history, target), so it keeps learning to USE
helpful motives while also learning a good null path (the O2 reader collapses without a motive); the
selector moves toward motives that explain the target (EMDR / perplexity-distillation fixed point).
One pass over the train rows (as R3 / SFT here). Test: argmax selector choice -> picks for
collab/motive_reader.py decode with the jointly trained reader (--reader <out_dir>/reader).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motive_reader as mr  # noqa: E402  (chdir's to the main checkout)
from grpo_motive_select import JUDGES, Selector, load_utilities, parse_hist, pca  # noqa: E402

import glob  # noqa: E402
import pandas as pd  # noqa: E402

NULL = 881
DATA = "/common/home/users/m/mk.ngohuu.2024/research/IntuRec/data/Toys_and_Games"


class SelectorNull(nn.Module):
    """Phase-1 selector over the 10 candidates + a null logit (bias + linear in the history vector)."""

    def __init__(self, n_scalar, d_hist, d_mot):
        super().__init__()
        self.cands = Selector(n_scalar, d_hist, d_mot)
        self.null_b = nn.Parameter(torch.zeros(()))
        self.null_w = nn.Linear(d_hist, 1, bias=False)
        nn.init.zeros_(self.null_w.weight)

    def forward(self, logp, scal, h, e):
        return torch.cat([self.cands(logp, scal, h, e), (self.null_b + self.null_w(h)).view(-1, 1)], 1)  # [B, 11]


def selector_features(frames, cand_all):
    """Same features as collab/grpo_motive_select.py (retriever log-probs, rank, popularity, history agreement,
    history / motive PCA vectors)."""
    C = f"{DATA}/collab/motive_corpus"
    labels = json.load(open(f"{C}/row_motives_train.json"))
    n_proto = len(json.load(open(f"{C}/prototypes.json")))
    logp_all = {}
    for s in ("train", "test"):
        sc = np.load(f"{C}/retriever/cls/scores_{s}.npy").astype(np.float32)
        sc -= sc.max(1, keepdims=True)
        logp_all[s] = sc - np.log(np.exp(sc).sum(1, keepdims=True))
    y = np.zeros((len(labels), n_proto), np.float32)
    for i, l in enumerate(labels):
        y[i, l] = 1.0
    pop = (y.sum(0) + 1.0) / (y.sum() + n_proto)
    n_items = int(max(frames["train"].item_id.max(), frames["test"].item_id.max())) + 1
    counts = np.zeros((n_items, n_proto), np.float32)
    np.add.at(counts, frames["train"].item_id.to_numpy(), y)
    item_prof = (counts + pop) / (counts.sum(1, keepdims=True) + 1.0)
    mot_vec = pca(np.load(f"{C}/motive_text_emb.npy").astype(np.float32), 64, np.arange(n_proto))
    n_tr = len(frames["train"])
    hv = pca(np.concatenate([logp_all["train"], logp_all["test"]]), 64, np.arange(n_tr))
    hist_vec = {"train": hv[:n_tr], "test": hv[n_tr:]}
    feats = {}
    for s in ("train", "test"):
        cand = np.clip(cand_all[s], 0, n_proto - 1)
        hist = frames[s].history_item_id.map(parse_hist)
        hp = np.stack([item_prof[h].mean(0) if len(h) else pop for h in hist])
        r = np.arange(len(cand))[:, None]
        logp = logp_all[s][r, cand]
        lhp, lpop = np.log(hp[r, cand] + 1e-9), np.log(pop[cand])
        scal = np.concatenate([np.eye(10, dtype=np.float32)[None].repeat(len(cand), 0),
                               np.stack([lpop, lhp, lhp - lpop, logp - logp[:, :1]], -1)], -1)
        feats[s] = dict(logp=logp.astype(np.float32), scal=scal.astype(np.float32), h=hist_vec[s], e=mot_vec[cand])
    return feats


def seq_logp(logits, ids, tmask):
    """Sum of log p(target tokens): logits [N, L, V], ids [N, L], tmask [N, L] (1 on target tokens)."""
    lp = torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (lp * tmask[:, 1:]).sum(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reader_init", default=mr.READER)
    p.add_argument("--k", type=int, default=4, help="candidate set size incl. null")
    p.add_argument("--rows_per_micro", type=int, default=2)
    p.add_argument("--rows_per_step", type=int, default=32)
    p.add_argument("--lr_reader", type=float, default=1e-5)
    p.add_argument("--lr_selector", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--limit_train", type=int, default=0)
    p.add_argument("--resume_every", type=int, default=200, help="optimizer steps between resume saves")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = a.device if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out_dir, exist_ok=True)

    frames = {s: pd.read_csv(glob.glob(f"{DATA}/{s}/*.csv")[0]) for s in ("train", "test")}
    cand_all = {s: load_utilities(JUDGES["Toys_and_Games"][s], len(frames[s]))[0] for s in ("train", "test")}
    feats = selector_features(frames, cand_all)
    T = lambda s, rows: [torch.from_numpy(feats[s][k][rows]).to(device) for k in ("logp", "scal", "h", "e")]

    reader, tok = mr.load_reader(a.reader_init, device)
    reader.train()
    ds = mr.CollabRecDataset("./data", "Toys_and_Games", tok, support_dir=None, support_k=0, dataset_type="train",
                             reason_step=1, seed=0, support_mode="text")
    sup_table = torch.from_numpy(ds.tokenize_texts(mr.motive_texts("Toys_and_Games"), 64)).to(device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else reader.config.pad_token_id

    selector = SelectorNull(feats["train"]["scal"].shape[-1], 64, 64).to(device)
    with torch.no_grad():  # start as the retriever's distribution, with "no motive" as a minor option
        selector.null_b.fill_(float(np.median(feats["train"]["logp"][:, 0])) - 1.0)
    with torch.no_grad():
        p0 = torch.softmax(selector(*[torch.from_numpy(feats["test"][k][:2048]).to(device)
                                      for k in ("logp", "scal", "h", "e")]), 1)
    print(f"initial selector on test: P(null) {p0[:, 10].mean():.3f}, argmax null {(p0.argmax(1) == 10).float().mean():.3f}, "
          f"argmax retriever top-1 {(p0.argmax(1) == 0).float().mean():.3f}", flush=True)
    rows_all = np.where((cand_all["train"] >= 0).all(1))[0]
    order = rng.permutation(rows_all)
    if a.limit_train:
        order = order[:a.limit_train]
    n_steps = (len(order) + a.rows_per_step - 1) // a.rows_per_step
    opt = torch.optim.AdamW([{"params": reader.parameters(), "lr": a.lr_reader},
                             {"params": selector.parameters(), "lr": a.lr_selector}], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) *
                                              0.5 * (1 + np.cos(np.pi * min(s, n_steps) / n_steps)))
    state_path = os.path.join(a.out_dir, "resume_state.pt")
    start = 0
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        reader.load_state_dict(st["reader"]); selector.load_state_dict(st["selector"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); start = st["step"]
        print(f"resumed at step {start}/{n_steps}", flush=True)

    def row_inputs(r):
        s = ds[int(r)]
        keep = np.asarray(s["attention_mask"]) == 1
        ids = np.asarray(s["input_ids"])[keep]
        tm = (np.asarray(s["labels"])[keep] != -100).astype(np.int64)
        return ids, tm

    log, t0 = {"loss": [], "post_null": [], "pi_null": [], "sel_entropy": []}, time.time()
    accum = a.rows_per_step // a.rows_per_micro
    for step in range(start, n_steps):
        rows = order[step * a.rows_per_step:(step + 1) * a.rows_per_step]
        opt.zero_grad()
        for mi in range(0, len(rows), a.rows_per_micro):
            mrows = rows[mi:mi + a.rows_per_micro]
            scores = selector(*T("train", mrows))                                    # [b, 11]; col 10 = null
            with torch.no_grad():                                                    # S = null + top-(k-1)
                top = scores[:, :10].topk(a.k - 1, -1).indices
            S = torch.cat([top, torch.full((len(mrows), 1), 10, device=device)], 1)  # [b, k]
            logpi_S = torch.log_softmax(scores.gather(1, S), -1)                     # [b, k]
            seqs, tms, sups = [], [], []
            for bi, r in enumerate(mrows):
                ids, tm = row_inputs(r)
                for j in range(a.k):
                    o = int(S[bi, j])
                    m = NULL if o == 10 else int(cand_all["train"][r, o])
                    seqs.append(ids); tms.append(tm); sups.append(m)
            L = max(len(x) for x in seqs)
            ids_t = torch.full((len(seqs), L), pad, dtype=torch.long, device=device)
            att = torch.zeros_like(ids_t); tm_t = torch.zeros_like(ids_t)
            for i, (x, t) in enumerate(zip(seqs, tms)):                              # left padding
                ids_t[i, L - len(x):] = torch.from_numpy(x); att[i, L - len(x):] = 1
                tm_t[i, L - len(x):] = torch.from_numpy(t)
            sup = sup_table[torch.tensor(sups, device=device)].unsqueeze(1).repeat(1, 2, 1)   # [N, 2, 64]
            logits = reader(input_ids=ids_t, attention_mask=att, support_tokens=sup).logits
            ll = seq_logp(logits, ids_t, tm_t).view(len(mrows), a.k)                # log p(target | m)
            joint = logpi_S + ll
            loss = -torch.logsumexp(joint, 1).mean()
            (loss / accum).backward()
            with torch.no_grad():
                post = torch.softmax(joint, 1)
                pi_full = torch.softmax(scores, 1)
                log["loss"].append(loss.item()); log["post_null"].append(post[:, -1].mean().item())
                log["pi_null"].append(pi_full[:, 10].mean().item())
                log["sel_entropy"].append(float(-(pi_full * torch.log(pi_full + 1e-12)).sum(1).mean()))
        torch.nn.utils.clip_grad_norm_(list(reader.parameters()) + list(selector.parameters()), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 20 == 0:
            print(f"step {step + 1}/{n_steps}: " + ", ".join(f"{k} {np.mean(v):.4f}" for k, v in log.items())
                  + f", {time.time() - t0:.0f}s", flush=True)
            log, t0 = {k: [] for k in log}, time.time()
        if a.resume_every and (step + 1) % a.resume_every == 0:
            torch.save({"reader": reader.state_dict(), "selector": selector.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step + 1}, state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)

    reader.save_pretrained(os.path.join(a.out_dir, "reader"))
    tok.save_pretrained(os.path.join(a.out_dir, "reader"))
    torch.save(selector.state_dict(), os.path.join(a.out_dir, "selector.pt"))
    selector.eval()
    with torch.no_grad():
        te = np.arange(len(frames["test"]))
        pk = torch.cat([selector(*T("test", te[i:i + 2048])).argmax(1) for i in range(0, len(te), 2048)]).cpu().numpy()
    picks = np.where(pk == 10, NULL, cand_all["test"][te, np.clip(pk, 0, 9)])
    np.save(os.path.join(a.out_dir, "picks_test.npy"), picks)
    print(f"test picks: null {np.mean(pk == 10):.3f}, retriever top-1 {np.mean(pk == 0):.3f}, "
          f"distinct {len(set(picks.tolist()))}", flush=True)
    if os.path.exists(state_path):
        os.remove(state_path)


if __name__ == "__main__":
    main()
