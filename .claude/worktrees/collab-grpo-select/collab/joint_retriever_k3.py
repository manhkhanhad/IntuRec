#!/usr/bin/env python3
"""Joint training of the REAL motive retriever (the 'cls' MLP over all 881 motive prototypes, no selector) and the
3-slot CollabRec recommender (reader), no InfoNCE.

Retriever: collab/train_motive_retriever.py head 'cls' (2-layer MLP on SASRec + gte history + item-motive profile +
user's past motives), init from motive_corpus/retriever/cls/mlp.pt, inputs cached by collab/retriever_features.py.
Reader: init run 364239 (k=3, inject; needs COLLAB_CODE). A motive SET of 3 is drawn Plackett-Luce from the
retriever's softmax over all 881 prototypes. Per row S = greedy top-3 + (n_sets - 1) Gumbel-top-3 samples
(duplicates masked); loss
    -log sum_{s in S} pi_S(s) * p(target | history, s)  [+ sup_weight * retriever multi-positive CE on true motives]
(Atlas / RAG marginal likelihood) so the recommender's likelihood back-propagates into the retriever's MLP.
Slots are fed in descending retriever score (the reader's training order). One pass over train.
Caveat: mlp.pt was fit on all train rows, so its train-row top-3 (0.461 contain a true motive) is better than test
(0.310); test is always unseen. Logs test retrieval (true motive in top-1/3/10) every --eval_every steps.
Test: greedy top-3 -> picks_test.npy [n, 3] for motive_reader.py decode with --reader <out_dir>/reader.
"""
import argparse
import glob
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
assert os.environ.get("COLLAB_CODE"), "set COLLAB_CODE to the CollabRec code the k=3 reader was trained with"
import motive_reader as mr  # noqa: E402
from joint_k3 import K, READER_K3, set_logprob  # noqa: E402
from joint_reader_selector import DATA, seq_logp  # noqa: E402

sys.path.insert(0, "/common/home/users/m/mk.ngohuu.2024/research/IntuRec/collab")
from train_motive_retriever import Retriever  # noqa: E402

CORPUS = f"{DATA}/collab/motive_corpus"
FEATS = "/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints/Toys_and_Games/CollabRec_GRPO_select/retriever_feats"


def load_retriever(device):
    ck = torch.load(f"{CORPUS}/retriever/cls/mlp.pt", map_location="cpu", weights_only=False)
    n_proto = ck["state_dict"]["query.3.weight"].shape[0]
    m = Retriever(ck["mu"].shape[0], n_proto, np.zeros((n_proto, 1), np.float32), SimpleNamespace(**ck["args"]))
    m.load_state_dict(ck["state_dict"])
    return m.to(device), n_proto


def retrieval_metrics(scores, labels):
    order = np.argsort(-scores, 1)[:, :10]
    keep = [i for i, l in enumerate(labels) if l]
    out = {}
    for k in (1, 3, 10):
        out[f"true_in_top{k}"] = round(float(np.mean([bool(set(labels[i]) & set(order[i, :k].tolist())) for i in keep])), 4)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reader_init", default=READER_K3)
    p.add_argument("--n_sets", type=int, default=4)
    p.add_argument("--rows_per_micro", type=int, default=2)
    p.add_argument("--rows_per_step", type=int, default=32)
    p.add_argument("--lr_reader", type=float, default=1e-5)
    p.add_argument("--lr_retriever", type=float, default=1e-4)
    p.add_argument("--sup_weight", type=float, default=0.0, help="retriever CE on true motives (0 = reader signal only)")
    p.add_argument("--gamma", type=float, default=0.0,
                   help="reader contrastive support (margin) loss, IntuRec intuition-loss analog: softplus(neg - pos), "
                        "pos/neg = thought from the chosen / random motives dotted with the anchor")
    p.add_argument("--sup_anchor", choices=["title", "motive"], default="title")
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
    retriever, n_proto = load_retriever(device)
    y_train = torch.zeros(n_train, n_proto, device=device)
    for i, l in enumerate(labels["train"]):
        if l:
            y_train[i, l] = 1.0 / len(l)
    reader, tok = mr.load_reader(a.reader_init, device)
    print(f"reader {a.reader_init}  readout {getattr(reader, 'readout', '?')}; retriever cls MLP, {n_proto} motives",
          flush=True)
    reader.gamma = a.gamma
    reader.train()
    ds = mr.CollabRecDataset("./data", "Toys_and_Games", tok, support_dir=None, support_k=0, dataset_type="train",
                             reason_step=1, seed=0, support_mode="text")
    assert np.array_equal(np.asarray(ds.row_ids), np.arange(n_train)), "dataset rows must be the CSV rows"
    sup_table = torch.from_numpy(ds.tokenize_texts(mr.motive_texts("Toys_and_Games"), 64)).to(device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else reader.config.pad_token_id
    anchor_table = None
    if a.gamma > 0:
        if a.sup_anchor == "title":   # [n_train, 1, 64] target title tokens
            anchor_table = torch.from_numpy(ds.tokenize_texts([d["tgt_item"] for d in ds.data], 64)).to(device)[:, None]
        else:                         # [n_train, 2, 64] true motive texts (one repeated; none -> "No specific motive.")
            ids = np.array([(l + l)[:2] if l else [n_proto] * 2 for l in labels["train"]])
            anchor_table = sup_table[torch.from_numpy(ids).to(device)]
        print(f"support (margin) loss: gamma {a.gamma}, anchor {a.sup_anchor} {tuple(anchor_table.shape)}", flush=True)

    @torch.no_grad()
    def test_scores():
        retriever.eval()
        sc = torch.cat([retriever(x["test"][i:i + 4096]) for i in range(0, len(x["test"]), 4096)]).float()
        retriever.train()
        return sc

    print(f"initial retriever on test: {retrieval_metrics(test_scores().cpu().numpy(), labels['test'])}", flush=True)

    order = rng.permutation(n_train)
    if a.limit_train:
        order = order[:a.limit_train]
    n_steps = (len(order) + a.rows_per_step - 1) // a.rows_per_step
    opt = torch.optim.AdamW([{"params": reader.parameters(), "lr": a.lr_reader},
                             {"params": retriever.parameters(), "lr": a.lr_retriever}], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) *
                                              0.5 * (1 + np.cos(np.pi * min(s, n_steps) / n_steps)))
    state_path = os.path.join(a.out_dir, "resume_state.pt")
    start = 0
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        reader.load_state_dict(st["reader"]); retriever.load_state_dict(st["retriever"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); start = st["step"]
        print(f"resumed at step {start}/{n_steps}", flush=True)
    retriever.train()

    def row_inputs(r):
        s = ds[int(r)]
        keep = np.asarray(s["attention_mask"]) == 1
        ids = np.asarray(s["input_ids"])[keep]
        tm = (np.asarray(s["labels"])[keep] != -100).astype(np.int64)
        return ids, tm

    keys = ["loss", "nll", "sup", "margin", "post_greedy", "greedy_true", "ent", "max_p"]
    log, t0 = {k: [] for k in keys}, time.time()
    accum = a.rows_per_step // a.rows_per_micro
    for step in range(start, n_steps):
        rows = order[step * a.rows_per_step:(step + 1) * a.rows_per_step]
        opt.zero_grad()
        for mi in range(0, len(rows), a.rows_per_micro):
            mrows = rows[mi:mi + a.rows_per_micro]
            b = len(mrows)
            rt = torch.as_tensor(mrows, device=device)
            scores = retriever(x["train"][rt]).float()                               # [b, 881]
            with torch.no_grad():
                gum = -torch.log(-torch.log(torch.rand(b, a.n_sets - 1, n_proto, device=device).clamp_min(1e-20)))
                samp = (scores.unsqueeze(1) + gum).topk(K, -1).indices
                S = torch.cat([scores.topk(K, -1).indices.unsqueeze(1), samp], 1)    # [b, n_sets, 3] motive ids
                # slots in descending current score (the reader's training order)
                S = S.gather(-1, scores.unsqueeze(1).expand(-1, a.n_sets, -1).gather(-1, S).argsort(-1, descending=True))
                key = S.sort(-1).values
                dup = torch.zeros(b, a.n_sets, dtype=torch.bool, device=device)
                for j in range(1, a.n_sets):
                    dup[:, j] = (key[:, j:j + 1] == key[:, :j]).all(-1).any(-1)
            lp_set = set_logprob(scores, S).masked_fill(dup, float("-inf"))
            logpi_S = lp_set - torch.logsumexp(lp_set, 1, keepdim=True)
            seqs, tms = [], []
            for r in mrows:
                ids, tm = row_inputs(r)
                seqs += [ids] * a.n_sets
                tms += [tm] * a.n_sets
            L = max(len(z) for z in seqs)
            ids_t = torch.full((len(seqs), L), pad, dtype=torch.long, device=device)
            att = torch.zeros_like(ids_t); tm_t = torch.zeros_like(ids_t)
            for i, (z, t) in enumerate(zip(seqs, tms)):                              # left padding
                ids_t[i, L - len(z):] = torch.from_numpy(z); att[i, L - len(z):] = 1
                tm_t[i, L - len(z):] = torch.from_numpy(t)
            sup = sup_table[S.reshape(-1, K)]                                         # [N, 3, 64]
            extra = {}
            if anchor_table is not None:
                extra = dict(rand_support_tokens=sup_table[torch.randint(0, n_proto, (len(seqs), K), device=device)],
                             anchor_tokens=anchor_table[rt.repeat_interleave(a.n_sets)])
            out = reader(input_ids=ids_t, attention_mask=att, support_tokens=sup, **extra)
            logits = out.logits
            margin = out.support_loss.float() if anchor_table is not None else logits.new_zeros(()).float()
            ll = seq_logp(logits, ids_t, tm_t).view(b, a.n_sets)
            joint = (logpi_S + ll).masked_fill(dup, float("-inf"))
            nll = -torch.logsumexp(joint, 1).mean()
            yb = y_train[rt]
            has = yb.sum(1) > 0
            sup_ce = (-(yb * F.log_softmax(scores, 1)).sum(1)[has].mean() if has.any() else scores.sum() * 0.0)
            loss = nll + a.sup_weight * sup_ce + a.gamma * margin
            (loss / accum).backward()
            with torch.no_grad():
                post = torch.softmax(joint, 1)
                pi = torch.softmax(scores, 1)
                gt = [bool(set(labels["train"][int(r)]) & set(S[i, 0].tolist())) for i, r in enumerate(mrows)]
                for k, v in zip(keys, [loss.item(), nll.item(), sup_ce.item(), margin.item(), post[:, 0].mean().item(), float(np.mean(gt)),
                                       float(-(pi * torch.log(pi + 1e-12)).sum(1).mean()), pi.max(1).values.mean().item()]):
                    log[k].append(v)
        torch.nn.utils.clip_grad_norm_(list(reader.parameters()) + list(retriever.parameters()), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 20 == 0:
            print(f"step {step + 1}/{n_steps}: " + ", ".join(f"{k} {np.mean(v):.4f}" for k, v in log.items())
                  + f", {time.time() - t0:.0f}s", flush=True)
            log, t0 = {k: [] for k in keys}, time.time()
        if a.eval_every and (step + 1) % a.eval_every == 0:
            print(f"step {step + 1}: test retrieval {retrieval_metrics(test_scores().cpu().numpy(), labels['test'])}",
                  flush=True)
        if a.resume_every and (step + 1) % a.resume_every == 0:
            torch.save({"reader": reader.state_dict(), "retriever": retriever.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step + 1}, state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)

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
