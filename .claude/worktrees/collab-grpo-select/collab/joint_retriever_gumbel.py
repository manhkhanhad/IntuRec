#!/usr/bin/env python3
"""End-to-end motive retriever + recommender with a straight-through Gumbel-softmax top-3 (user's design):

    context -> cls retriever (881 motive scores) -> Gumbel-softmax top-3 -> recommender (reader) -> next-item loss
                ^------------------------------- gradient ---------------------------------------|

Retriever: the cls MLP (init motive_corpus/retriever/cls/mlp.pt; inputs cached by collab/retriever_features.py).
Reader: init run 364239 (k=3 motive slots, inject readout, no InfoNCE; needs COLLAB_CODE).
Selection, per row: z = scores + Gumbel noise; 3 slots in turn, slot t: w_t = softmax(z / tau) over motives not yet
taken, hard choice = argmax. Straight-through: the forward pass feeds the reader EXACTLY the token embeddings of the
3 chosen motives (as at test time); the backward pass uses the soft mixture sum_m w_t[m] * emb(m), so the reader's
loss reaches every retriever score. Implemented as a forward hook on the reader's embed_tokens for the motive-slot
tokens only (out = sum_m (onehot + w - w.detach())[m] * emb(m), which equals emb(chosen) exactly in the forward).
Loss = -log p(target | history, chosen 3 motives) [+ sup_weight * retriever CE on true motives]. One reader pass per
row. tau anneals geometrically tau_start -> tau_end. Test retrieval (true motive in top-1/3/10) every --eval_every
steps; logs the retriever's gradient norm (how much signal reaches it). Test: deterministic top-3 -> picks_test.npy.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
assert os.environ.get("COLLAB_CODE"), "set COLLAB_CODE to the CollabRec code the k=3 reader was trained with"
import motive_reader as mr  # noqa: E402
from joint_k3 import K, READER_K3  # noqa: E402
from joint_reader_selector import DATA, seq_logp  # noqa: E402
from joint_retriever_k3 import CORPUS, FEATS, load_retriever, retrieval_metrics  # noqa: E402
from train_motive_retriever import Retriever  # noqa: E402  (path added by joint_retriever_k3)
from types import SimpleNamespace  # noqa: E402


def load_retriever_from(path, device):
    """A cls retriever in mlp.pt format (e.g. collab/train_retriever_fold.py output)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    n_proto = ck["state_dict"]["query.3.weight"].shape[0]
    m = Retriever(ck["mu"].shape[0], n_proto, np.zeros((n_proto, 1), np.float32), SimpleNamespace(**ck["args"]))
    m.load_state_dict(ck["state_dict"])
    return m.to(device), n_proto


def st_gumbel_topk(scores, k, tau, noise_scale=1.0):
    """scores [b, n] -> (idx [b, k] hard choices in pick order, W [b, k, n] straight-through weights:
    forward = one-hot of idx, backward = d softmax((scores + g) / tau) restricted to not-yet-taken motives)."""
    g = -torch.log(-torch.log(torch.rand_like(scores).clamp_min(1e-20)))
    z = scores + noise_scale * g   # noise_scale 1 = standard Gumbel sampling; 0 = deterministic top-k
    taken = torch.zeros_like(scores, dtype=torch.bool)
    idx, W = [], []
    for _ in range(k):
        zt = z.masked_fill(taken, float("-inf"))
        w = torch.softmax(zt / tau, -1)
        i = zt.argmax(-1)
        y = F.one_hot(i, scores.shape[-1]).to(w.dtype)
        idx.append(i)
        W.append(y + w - w.detach())
        taken = taken | y.bool()
    return torch.stack(idx, 1), torch.stack(W, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reader_init", default=READER_K3)
    p.add_argument("--rows_per_micro", type=int, default=8)
    p.add_argument("--rows_per_step", type=int, default=32)
    p.add_argument("--lr_reader", type=float, default=1e-5)
    p.add_argument("--lr_retriever", type=float, default=1e-4)
    p.add_argument("--tau_start", type=float, default=1.0)
    p.add_argument("--tau_end", type=float, default=0.3)
    p.add_argument("--noise_scale", type=float, default=1.0,
                   help="Gumbel noise scale: 1 = standard Gumbel-softmax sampling, <1 = closer to the retriever's top-3")
    p.add_argument("--sup_weight", type=float, default=0.0, help="retriever CE on true motives (0 = reader loss only)")
    p.add_argument("--retriever_ckpt", default="", help="init retriever (mlp.pt format); default = cls/mlp.pt")
    p.add_argument("--train_rows", default="", help=".npy of train rows to use (e.g. rows the retriever never saw)")
    p.add_argument("--retr_optim", choices=["adamw", "sgd"], default="adamw",
                   help="sgd: plain SGD for the retriever (step size proportional to the gradient, unlike AdamW)")
    p.add_argument("--retr_momentum", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--limit_train", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--resume_every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = a.device if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out_dir, exist_ok=True)

    n_train = len(pd.read_csv(glob.glob(f"{DATA}/train/*.csv")[0]))
    x = {s: torch.from_numpy(np.load(f"{FEATS}/x_{s}.npy").astype(np.float32)).to(device) for s in ("train", "test")}
    assert len(x["train"]) == n_train, (x["train"].shape, n_train)
    labels = {s: json.load(open(f"{CORPUS}/row_motives_{s}.json")) for s in ("train", "test")}
    retriever, n_proto = load_retriever_from(a.retriever_ckpt, device) if a.retriever_ckpt else load_retriever(device)
    y_train = torch.zeros(n_train, n_proto, device=device)
    for i, l in enumerate(labels["train"]):
        if l:
            y_train[i, l] = 1.0 / len(l)
    reader, tok = mr.load_reader(a.reader_init, device)
    print(f"reader {a.reader_init}  readout {getattr(reader, 'readout', '?')}; retriever cls MLP, {n_proto} motives; "
          f"straight-through Gumbel-softmax top-{K}, tau {a.tau_start} -> {a.tau_end}, noise scale {a.noise_scale}", flush=True)
    reader.train()
    ds = mr.CollabRecDataset("./data", "Toys_and_Games", tok, support_dir=None, support_k=0, dataset_type="train",
                             reason_step=1, seed=0, support_mode="text")
    assert np.array_equal(np.asarray(ds.row_ids), np.arange(n_train)), "dataset rows must be the CSV rows"
    sup_table = torch.from_numpy(ds.tokenize_texts(mr.motive_texts("Toys_and_Games"), 64)).to(device)  # [882, 64]
    motive_tokens = sup_table[:n_proto]                                                               # [881, 64]
    pad = tok.pad_token_id if tok.pad_token_id is not None else reader.config.pad_token_id

    # Straight-through hook: for the motive-slot tokens ([N, 3, 64]) replace embed_tokens' output by
    # sum_m W[n, slot, m] * emb(motive m); equal to the hard embeddings in the forward pass.
    st = {"W": None}

    def hook(module, inputs, output):
        if st["W"] is None or inputs[0].dim() != 3:
            return None
        table = module(motive_tokens)                                                # [881, 64, dim] (2-D input)
        return torch.einsum("nkm,mld->nkld", st["W"].to(table.dtype), table)

    handle = reader.model.embed_tokens.register_forward_hook(hook)

    @torch.no_grad()
    def test_scores():
        retriever.eval()
        sc = torch.cat([retriever(x["test"][i:i + 4096]) for i in range(0, len(x["test"]), 4096)]).float()
        retriever.train()
        return sc

    print(f"initial retriever on test: {retrieval_metrics(test_scores().cpu().numpy(), labels['test'])}", flush=True)

    order = rng.permutation(np.load(a.train_rows) if a.train_rows else n_train)
    print(f"train rows {len(order)}; retriever {a.retriever_ckpt or 'cls/mlp.pt'}; retriever optimizer {a.retr_optim} "
          f"lr {a.lr_retriever} momentum {a.retr_momentum}", flush=True)
    if a.limit_train:
        order = order[:a.limit_train]
    n_steps = (len(order) + a.rows_per_step - 1) // a.rows_per_step
    lr_lambda = lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + np.cos(np.pi * min(s, n_steps) / n_steps))
    if a.retr_optim == "adamw":   # original single optimizer (resume format unchanged)
        opt = torch.optim.AdamW([{"params": reader.parameters(), "lr": a.lr_reader},
                                 {"params": retriever.parameters(), "lr": a.lr_retriever}], weight_decay=0.0)
        opts = [opt]
    else:
        opt = torch.optim.AdamW(reader.parameters(), lr=a.lr_reader, weight_decay=0.0)
        opts = [opt, torch.optim.SGD(retriever.parameters(), lr=a.lr_retriever, momentum=a.retr_momentum)]
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lr_lambda) for o in opts]
    sched = scheds[0]
    state_path = os.path.join(a.out_dir, "resume_state.pt")
    start = 0
    if os.path.exists(state_path):
        stt = torch.load(state_path, map_location="cpu", weights_only=False)
        reader.load_state_dict(stt["reader"]); retriever.load_state_dict(stt["retriever"])
        opt.load_state_dict(stt["opt"]); sched.load_state_dict(stt["sched"]); start = stt["step"]
        if len(opts) > 1:
            opts[1].load_state_dict(stt["opt_retr"]); scheds[1].load_state_dict(stt["sched_retr"])
        print(f"resumed at step {start}/{n_steps}", flush=True)
    retriever.train()

    def row_inputs(r):
        s = ds[int(r)]
        keep = np.asarray(s["attention_mask"]) == 1
        ids = np.asarray(s["input_ids"])[keep]
        tm = (np.asarray(s["labels"])[keep] != -100).astype(np.int64)
        return ids, tm

    keys = ["loss", "nll", "sup", "pick_true", "pick_is_top3", "ent", "max_p", "retr_grad"]
    log, t0 = {k: [] for k in keys}, time.time()
    accum = a.rows_per_step // a.rows_per_micro
    retr_params = list(retriever.parameters())
    for step in range(start, n_steps):
        tau = a.tau_start * (a.tau_end / a.tau_start) ** (step / max(1, n_steps - 1))
        rows = order[step * a.rows_per_step:(step + 1) * a.rows_per_step]
        for o in opts:
            o.zero_grad()
        for mi in range(0, len(rows), a.rows_per_micro):
            mrows = rows[mi:mi + a.rows_per_micro]
            b = len(mrows)
            rt = torch.as_tensor(mrows, device=device)
            scores = retriever(x["train"][rt]).float()                               # [b, 881]
            idx, W = st_gumbel_topk(scores, K, tau, a.noise_scale)                                   # [b, 3], [b, 3, 881]
            seqs, tms = zip(*[row_inputs(r) for r in mrows])
            L = max(len(z) for z in seqs)
            ids_t = torch.full((b, L), pad, dtype=torch.long, device=device)
            att = torch.zeros_like(ids_t); tm_t = torch.zeros_like(ids_t)
            for i, (z, t) in enumerate(zip(seqs, tms)):                              # left padding
                ids_t[i, L - len(z):] = torch.from_numpy(z); att[i, L - len(z):] = 1
                tm_t[i, L - len(z):] = torch.from_numpy(t)
            st["W"] = W
            try:
                logits = reader(input_ids=ids_t, attention_mask=att, support_tokens=sup_table[idx]).logits
            finally:
                st["W"] = None
            nll = -seq_logp(logits, ids_t, tm_t).mean()
            yb = y_train[rt]
            has = yb.sum(1) > 0
            sup_ce = (-(yb * F.log_softmax(scores, 1)).sum(1)[has].mean() if has.any() else scores.sum() * 0.0)
            loss = nll + a.sup_weight * sup_ce
            (loss / accum).backward()
            with torch.no_grad():
                pi = torch.softmax(scores, 1)
                top3 = scores.topk(K, -1).indices.sort(-1).values
                pt = [bool(set(labels["train"][int(r)]) & set(idx[i].tolist())) for i, r in enumerate(mrows)]
                for k, v in zip(keys[:-1], [loss.item(), nll.item(), sup_ce.item(), float(np.mean(pt)),
                                            (idx.sort(-1).values == top3).all(1).float().mean().item(),
                                            float(-(pi * torch.log(pi + 1e-12)).sum(1).mean()),
                                            pi.max(1).values.mean().item()]):
                    log[k].append(v)
        with torch.no_grad():   # retriever gradient norm before clipping: how much signal reaches the retriever
            log["retr_grad"].append(float(torch.norm(torch.stack([q.grad.norm() for q in retr_params if q.grad is not None]))))
        if len(opts) > 1:   # SGD retriever: clip separately so the reader's large gradient does not shrink it
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(retr_params, 1.0)
        else:
            torch.nn.utils.clip_grad_norm_(list(reader.parameters()) + retr_params, 1.0)
        for o, sc_ in zip(opts, scheds):
            o.step()
            sc_.step()
        if (step + 1) % 20 == 0:
            print(f"step {step + 1}/{n_steps}: tau {tau:.3f}, " + ", ".join(f"{k} {np.mean(v):.4f}" for k, v in log.items())
                  + f", {time.time() - t0:.0f}s", flush=True)
            log, t0 = {k: [] for k in keys}, time.time()
        if a.eval_every and (step + 1) % a.eval_every == 0:
            print(f"step {step + 1}: test retrieval {retrieval_metrics(test_scores().cpu().numpy(), labels['test'])}",
                  flush=True)
        if a.resume_every and (step + 1) % a.resume_every == 0:
            extra_state = ({"opt_retr": opts[1].state_dict(), "sched_retr": scheds[1].state_dict()} if len(opts) > 1 else {})
            torch.save({"reader": reader.state_dict(), "retriever": retriever.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step + 1, **extra_state}, state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)

    handle.remove()
    reader.save_pretrained(os.path.join(a.out_dir, "reader"))
    tok.save_pretrained(os.path.join(a.out_dir, "reader"))
    torch.save(retriever.state_dict(), os.path.join(a.out_dir, "retriever.pt"))
    sc = test_scores()
    np.save(os.path.join(a.out_dir, "scores_test.npy"), sc.cpu().numpy().astype(np.float16))
    picks = sc.topk(K, -1).indices.cpu().numpy()                                     # descending score
    np.save(os.path.join(a.out_dir, "picks_test.npy"), picks)
    print(f"final retriever on test: {retrieval_metrics(sc.cpu().numpy(), labels['test'])}; "
          f"distinct motives in picks {len(set(picks.ravel().tolist()))}", flush=True)
    if os.path.exists(state_path):
        os.remove(state_path)


if __name__ == "__main__":
    main()
