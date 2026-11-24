#!/usr/bin/env python3
"""Joint motive-retriever (selector) + recommender (reader) training with 3 motive slots, no InfoNCE.

Reader: CollabRec predicted-motive model, k=3, inject readout (init: run 364239, trained on the retriever's
top-3 in rank order). Needs COLLAB_CODE=<recovered CollabRec code> in the environment (collab/motive_reader.py).
Selector: the phase-1 selector + a "no motive" logit (joint_reader_selector.SelectorNull) over 11 options =
retriever top-10 (cls scores) + null; a motive SET of 3 options is drawn Plackett-Luce (without replacement).
Per train row, S = {selector's greedy top-3} + (n_sets - 1) Gumbel-top-3 samples (duplicates masked); loss
    -log sum_{s in S} pi_S(s) * p(target | history, s),   pi_S = PL set probability renormalised on S
(Atlas / RAG / RA-DIT marginal likelihood). The reader is trained on the sets weighted by their posterior, the
selector moves toward sets that explain the target. One pass over train. Slots are fed in candidate-rank order
(null last), as in the reader's training. Test: greedy top-3 -> picks_test.npy [n, 3] for motive_reader.py decode
with --reader <out_dir>/reader.
"""
import argparse
import glob
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
assert os.environ.get("COLLAB_CODE"), "set COLLAB_CODE to the CollabRec code the k=3 reader was trained with"
import motive_reader as mr  # noqa: E402
from joint_reader_selector import DATA, NULL, SelectorNull, selector_features, seq_logp  # noqa: E402

READER_K3 = ("/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints/Toys_and_Games/CollabRec/"
             "text-micro2-predicted_k3_beta0_rs1_lr0.0003_job364239")
PERMS = list(itertools.permutations(range(3)))
K = 3


def retriever_top10(split):
    """Exactly CollabRecDataset(motive_source='predicted') ordering: float32 log-softmax, argsort."""
    sc = np.load(f"{DATA}/collab/motive_corpus/retriever/cls/scores_{split}.npy").astype(np.float32)
    sc -= sc.max(1, keepdims=True)
    sc = sc - np.log(np.exp(sc).sum(1, keepdims=True))
    return np.argsort(-sc, axis=1)[:, :10].astype(np.int64)


def set_logprob(scores, sets):
    """log P(set) under Plackett-Luce over `scores` [b, O], sets [b, S, 3] of distinct options -> [b, S]
    (sum over the 6 orderings)."""
    b, S, _ = sets.shape
    s = scores.unsqueeze(1).expand(b, S, scores.shape[1])
    per = []
    for p in PERMS:
        idx = sets[..., list(p)]
        mask = torch.zeros_like(s, dtype=torch.bool)
        lp = 0.0
        for t in range(K):
            st = s.masked_fill(mask, float("-inf"))
            lp = lp + st.gather(-1, idx[..., t:t + 1]).squeeze(-1) - torch.logsumexp(st, -1)
            mask = mask.scatter(-1, idx[..., t:t + 1], True)
        per.append(lp)
    return torch.logsumexp(torch.stack(per, -1), -1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reader_init", default=READER_K3)
    p.add_argument("--n_sets", type=int, default=4, help="motive sets per row: greedy top-3 + (n_sets-1) samples")
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
    cand_all = {s: retriever_top10(s) for s in ("train", "test")}
    for s in cand_all:
        assert len(cand_all[s]) == len(frames[s]), (s, cand_all[s].shape, len(frames[s]))
    feats = selector_features(frames, cand_all)
    T = lambda s, rows: [torch.from_numpy(feats[s][k][rows]).to(device) for k in ("logp", "scal", "h", "e")]

    reader, tok = mr.load_reader(a.reader_init, device)
    print(f"reader {a.reader_init}  readout {getattr(reader, 'readout', '?')}", flush=True)
    reader.train()
    ds = mr.CollabRecDataset("./data", "Toys_and_Games", tok, support_dir=None, support_k=0, dataset_type="train",
                             reason_step=1, seed=0, support_mode="text")
    sup_table = torch.from_numpy(ds.tokenize_texts(mr.motive_texts("Toys_and_Games"), 64)).to(device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else reader.config.pad_token_id
    cand_t = {s: torch.from_numpy(np.concatenate([cand_all[s], np.full((len(cand_all[s]), 1), NULL)], 1)).to(device)
              for s in cand_all}                                                    # option -> motive id (10 = null)

    selector = SelectorNull(feats["train"]["scal"].shape[-1], 64, 64).to(device)
    with torch.no_grad():  # start as the retriever (greedy set = its top-3), "no motive" just below rank 3
        selector.null_b.fill_(float(np.quantile(feats["train"]["logp"][:, 2], 0.05)) - 1.0)

    def greedy(scores):
        return scores.topk(K, -1).indices.sort(-1).values                          # [b, 3], canonical order

    with torch.no_grad():
        g0 = greedy(selector(*T("test", np.arange(2048))))
    print(f"initial selector on test: greedy = retriever top-3 {(g0 == torch.arange(3, device=device)).all(1).float().mean():.3f}, "
          f"null in greedy {(g0 == 10).any(1).float().mean():.3f}", flush=True)

    order = rng.permutation(len(frames["train"]))
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

    log, t0 = {"loss": [], "post_greedy": [], "greedy_is_top3": [], "null_in_greedy": [], "sel_entropy": []}, time.time()
    accum = a.rows_per_step // a.rows_per_micro
    for step in range(start, n_steps):
        rows = order[step * a.rows_per_step:(step + 1) * a.rows_per_step]
        opt.zero_grad()
        for mi in range(0, len(rows), a.rows_per_micro):
            mrows = rows[mi:mi + a.rows_per_micro]
            b = len(mrows)
            scores = selector(*T("train", mrows))                                    # [b, 11]
            with torch.no_grad():
                gum = -torch.log(-torch.log(torch.rand(b, a.n_sets - 1, scores.shape[1], device=device)
                                            .clamp_min(1e-20)))
                samp = (scores.unsqueeze(1) + gum).topk(K, -1).indices.sort(-1).values
                S = torch.cat([greedy(scores).unsqueeze(1), samp], 1)                 # [b, n_sets, 3]
                dup = torch.zeros(b, a.n_sets, dtype=torch.bool, device=device)
                for j in range(1, a.n_sets):
                    dup[:, j] = (S[:, j:j + 1] == S[:, :j]).all(-1).any(-1)
            lp_set = set_logprob(scores, S).masked_fill(dup, float("-inf"))
            logpi_S = lp_set - torch.logsumexp(lp_set, 1, keepdim=True)
            seqs, tms = [], []
            for r in mrows:
                ids, tm = row_inputs(r)
                seqs += [ids] * a.n_sets
                tms += [tm] * a.n_sets
            L = max(len(x) for x in seqs)
            ids_t = torch.full((len(seqs), L), pad, dtype=torch.long, device=device)
            att = torch.zeros_like(ids_t); tm_t = torch.zeros_like(ids_t)
            for i, (x, t) in enumerate(zip(seqs, tms)):                              # left padding
                ids_t[i, L - len(x):] = torch.from_numpy(x); att[i, L - len(x):] = 1
                tm_t[i, L - len(x):] = torch.from_numpy(t)
            rows_t = torch.as_tensor(mrows, device=device).view(b, 1, 1).expand_as(S)
            mot = cand_t["train"][rows_t, S]                                          # [b, n_sets, 3] motive ids
            sup = sup_table[mot.reshape(-1, K)]                                       # [N, 3, 64]
            logits = reader(input_ids=ids_t, attention_mask=att, support_tokens=sup).logits
            ll = seq_logp(logits, ids_t, tm_t).view(b, a.n_sets)
            joint = (logpi_S + ll).masked_fill(dup, float("-inf"))
            loss = -torch.logsumexp(joint, 1).mean()
            (loss / accum).backward()
            with torch.no_grad():
                post = torch.softmax(joint, 1)
                pi = torch.softmax(scores, 1)
                log["loss"].append(loss.item()); log["post_greedy"].append(post[:, 0].mean().item())
                log["greedy_is_top3"].append((S[:, 0] == torch.arange(3, device=device)).all(1).float().mean().item())
                log["null_in_greedy"].append((S[:, 0] == 10).any(1).float().mean().item())
                log["sel_entropy"].append(float(-(pi * torch.log(pi + 1e-12)).sum(1).mean()))
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
        g = torch.cat([greedy(selector(*T("test", te[i:i + 2048]))) for i in range(0, len(te), 2048)])
        picks = cand_t["test"][torch.as_tensor(te, device=device).view(-1, 1).expand_as(g), g].cpu().numpy()
    np.save(os.path.join(a.out_dir, "picks_test.npy"), picks)
    print(f"test picks: = retriever top-3 {(g == torch.arange(3, device=device)).all(1).float().mean():.3f}, "
          f"null in set {(g == 10).any(1).float().mean():.3f}, distinct motives {len(set(picks.ravel().tolist()))}",
          flush=True)
    if os.path.exists(state_path):
        os.remove(state_path)


if __name__ == "__main__":
    main()
