#!/usr/bin/env python3
"""Offline GRPO motive selector (phase 1): learn which retrieved motive to inject.

Action: pick 1 of the retriever's top-10 motive candidates for a row. Reward: the utility of that
motive for a frozen motive-reading CollabRec model (the "judge" / reader, an O2-style model),
  u(m) = log p(target | history, [m, m]) - log p(target | history, "No specific motive."),
precomputed by collab/motive_utility.py (worktree collab-refined-infonce, now lost):
  train rows: two cross-fitted judges (each scored the users it did not train on),
              <judge run>/motive_utility/utility_train_fold{0,1}of2.npz
  test rows:  the full-data judge 351001, utility_test.npz
  npz columns: row, kind (0 null, 1 true pair, 2 true single, 3 retriever top-10 with rank 0-9,
               4 random), motive, rank, ll (target log-likelihood).
Rewards are table lookups, so this runs on CPU in minutes.

Policy: score(row, m) = a * retriever log-prob + rank one-hot + popularity / history-agreement
features + bilinear(history vector, motive text vector); softmax over the 10 candidates. Initialised
to the retriever's own distribution (a = 1, rest 0). History vector = PCA of the row's retriever
log-prob vector over all 881 motives (train: out-of-fold scores); motive vector = PCA of the
prototype's Qwen3 text embedding.

Methods compared (same features / model):
  retriever top-1, random top-10, oracle best of top-10 (upper bound),
  supervised listwise (cross-entropy to softmax(u / tau)),
  GRPO: G samples per row from the policy, advantage = z-scored reward within the row's group,
        loss = -A * log pi(a) + beta * KL(pi || pi_retriever) (on-policy, one update per batch).
Early stopping on 10% held-out train users (mean u of the argmax pick).
"""
import argparse
import ast
import collections
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CK = Path("/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints")
DATA = Path("/common/home/users/m/mk.ngohuu.2024/research/IntuRec/data")
JUDGES = {
    "Toys_and_Games": {
        "train": [CK / "Toys_and_Games/CollabRec_oracle/text-micro2-oracle_own_motive-cq0.1_user+target+motive-fold0of2_k2_beta0_rs1_lr0.0003_job357943/motive_utility/utility_train_fold0of2.npz",
                  CK / "Toys_and_Games/CollabRec_oracle/text-micro2-oracle_own_motive-cq0.1_user+target+motive-fold1of2_k2_beta0_rs1_lr0.0003_job357944/motive_utility/utility_train_fold1of2.npz"],
        "test": [CK / "Toys_and_Games/CollabRec_oracle/text-micro2-oracle_own_motive-cq0.1_user+target+motive_k2_beta0_rs1_lr0.0003_job351001/motive_utility/utility_test.npz"],
    },
}
K = 10


def load_utilities(files, n_rows):
    """-> cand [n, 10] motive ids (retriever rank order), u [n, 10], u_pair [n] (nan if no true pair)."""
    cand = np.full((n_rows, K), -1, np.int64)
    u = np.full((n_rows, K), np.nan, np.float32)
    u_pair = np.full(n_rows, np.nan, np.float32)
    for f in files:
        z = np.load(f)
        row, kind, motive, rank, ll = z["row"], z["kind"], z["motive"], z["rank"], z["ll"]
        null = np.full(n_rows, np.nan, np.float32)
        null[row[kind == 0]] = ll[kind == 0]
        s = kind == 3
        cand[row[s], rank[s]] = motive[s]
        u[row[s], rank[s]] = ll[s] - null[row[s]]
        s = kind == 1
        u_pair[row[s]] = ll[s] - null[row[s]]
    return cand, u, u_pair


RANK_DIR = CK / "Toys_and_Games/CollabRec_GRPO_select/rank_reward"
RANK_FILES = {"Toys_and_Games": {"train": [RANK_DIR / "rank_utility_train0.npz", RANK_DIR / "rank_utility_train1.npz"],
                                 "test": [RANK_DIR / "rank_utility_test.npz"]}}


def load_rank_utilities(files, n_rows, norm=True):
    """Ranking-aware reward (collab/rank_utility.py): for candidate m, the target's length-normalised
    log-softmax among target + hard negatives under [m, m]. -> cand [n, 10], u [n, 10], u_pair (all nan),
    trank [n, 11] target rank among the scored items (column 0 = null motive, 1.. = candidates)."""
    cand = np.full((n_rows, K), -1, np.int64)
    u = np.full((n_rows, K), np.nan, np.float32)
    trank = np.full((n_rows, K + 1), np.nan, np.float32)
    for f in files:
        z = np.load(f)
        nll = z["ll"] / z["ntok"][:, None, :] if norm else z["ll"]             # [r, 11, J]
        lse = np.log(np.exp(nll - nll.max(-1, keepdims=True)).sum(-1)) + nll.max(-1)
        rows = z["row"]
        cand[rows] = z["motive"][:, 1:]
        u[rows] = (nll[:, :, 0] - lse)[:, 1:]
        trank[rows] = (nll > nll[:, :, :1]).sum(-1)
    return cand, u, np.full(n_rows, np.nan, np.float32), trank


def parse_hist(x):
    return list(ast.literal_eval(x)) if isinstance(x, str) else list(x)


class Selector(nn.Module):
    def __init__(self, n_scalar, d_hist, d_mot, d_bil=32):
        super().__init__()
        self.a = nn.Parameter(torch.ones(()))              # weight on retriever log-prob (init: = retriever)
        self.lin = nn.Linear(n_scalar, 1, bias=False)
        nn.init.zeros_(self.lin.weight)
        self.U = nn.Linear(d_hist, d_bil, bias=False)
        self.V = nn.Linear(d_mot, d_bil, bias=False)
        nn.init.normal_(self.U.weight, std=1e-3)
        nn.init.normal_(self.V.weight, std=1e-3)

    def forward(self, logp, scal, h, e):
        # logp [B,10], scal [B,10,S], h [B,Dh], e [B,10,Dm] -> scores [B,10]
        bil = (self.U(h).unsqueeze(1) * self.V(e)).sum(-1)
        return self.a * logp + self.lin(scal).squeeze(-1) + bil


def pca(x, d, fit_rows):
    mu = x[fit_rows].mean(0)
    _, _, vt = np.linalg.svd(x[fit_rows] - mu, full_matrices=False)
    z = (x - mu) @ vt[:d].T
    return (z / (z[fit_rows].std(0) + 1e-6)).astype(np.float32)


def evaluate(pick, cand, u, u_pair, labels, rows, trank=None):
    """pick [n] index into the 10 candidates (or -1 = abstain, reward 0) over `rows`."""
    up = np.where(pick >= 0, u[rows, np.clip(pick, 0, K - 1)], 0.0)
    best = np.nanmax(u[rows], 1)
    pair = u_pair[rows]
    ok = ~np.isnan(pair)
    hit = [cand[r, p] in labels[r] if p >= 0 else False for r, p in zip(rows, pick)]
    extra = {}
    if trank is not None:
        tr = trank[rows, np.clip(pick, 0, K - 1) + 1]
        extra = {"target_mean_rank": float(np.mean(tr)), "target_top1": float(np.mean(tr == 0)),
                 "target_top5": float(np.mean(tr < 5)), "null_target_mean_rank": float(np.mean(trank[rows, 0])),
                 "null_target_top1": float(np.mean(trank[rows, 0] == 0))}
    return {**extra, "mean_u": float(np.mean(up)),
            "share_of_oracle_pair_gain": float(np.mean(up[ok]) / np.mean(pair[ok])) if ok.any() else float("nan"),
            "share_u_pos": float(np.mean(up > 0)), "label_hit@1": float(np.mean(hit)),
            "picked_rank0": float(np.mean(pick == 0)), "distinct_motives": int(len({cand[r, p] for r, p in zip(rows, pick) if p >= 0})),
            "best_of_10_mean_u": float(np.mean(best))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="Toys_and_Games")
    p.add_argument("--method", choices=["supervised", "grpo"], default="grpo")
    p.add_argument("--rank_train", default="", help="comma list of train rank files to use, e.g. train1")
    p.add_argument("--rank_norm", choices=["mean", "sum"], default="mean",
                   help="rank_lse: length-normalised (as the beam ranks titles) or summed log-likelihood")
    p.add_argument("--reward", choices=["ll", "rank_lse"], default="ll",
                   help="ll: target log-likelihood gain vs null; rank_lse: target log-softmax among hard negatives")
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.05, help="KL to the retriever's top-10 distribution (GRPO)")
    p.add_argument("--tau", type=float, default=0.5, help="target temperature (supervised)")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    p.add_argument("--save_picks", default="", help="save test motive ids [n_test] (.npy) for decoding")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    C = DATA / a.dataset / "collab" / "motive_corpus"
    frames = {s: pd.read_csv(glob.glob(str(DATA / a.dataset / s / "*.csv"))[0]) for s in ("train", "test")}
    labels = {s: json.load(open(C / f"row_motives_{s}.json")) for s in ("train", "test")}
    n_proto = len(json.load(open(C / "prototypes.json")))
    trank = {"train": None, "test": None}
    if a.reward == "rank_lse":
        util = {}
        for s in ("train", "test"):
            files = RANK_FILES[a.dataset][s]
            if s == "train" and a.rank_train:
                files = [RANK_DIR / f"rank_utility_{t}.npz" for t in a.rank_train.split(",")]
            c, uu, up, tr = load_rank_utilities(files, len(frames[s]), a.rank_norm == "mean")
            util[s], trank[s] = (c, uu, up), tr
    else:
        util = {s: load_utilities(JUDGES[a.dataset][s], len(frames[s])) for s in ("train", "test")}

    # Sanity: utility candidates = retriever top-10 in rank order.
    logp_all = {}
    for s in ("train", "test"):
        sc = np.load(C / "retriever" / "cls" / f"scores_{s}.npy").astype(np.float32)
        sc -= sc.max(1, keepdims=True)
        logp_all[s] = sc - np.log(np.exp(sc).sum(1, keepdims=True))
        cand = util[s][0]
        okr = (cand >= 0).all(1)
        top = np.argsort(-logp_all[s], 1)[:, :K]
        print(f"{s}: rows with utilities {okr.mean():.4f}; candidates == retriever top-10 order: "
              f"{(top[okr] == cand[okr]).all(1).mean():.4f}", flush=True)

    # Popularity and history agreement (train labels only).
    y = np.zeros((len(labels["train"]), n_proto), np.float32)
    for i, l in enumerate(labels["train"]):
        y[i, l] = 1.0
    pop = (y.sum(0) + 1.0) / (y.sum() + n_proto)
    n_items = int(max(frames["train"].item_id.max(), frames["test"].item_id.max())) + 1
    counts = np.zeros((n_items, n_proto), np.float32)
    np.add.at(counts, frames["train"].item_id.to_numpy(), y)
    item_prof = (counts + pop) / (counts.sum(1, keepdims=True) + 1.0)

    mot_emb = np.load(C / "motive_text_emb.npy").astype(np.float32)
    mot_vec = pca(mot_emb, 64, np.arange(n_proto))
    n_tr = len(frames["train"])
    hist_vec_all = pca(np.concatenate([logp_all["train"], logp_all["test"]]), 64, np.arange(n_tr))
    hist_vec = {"train": hist_vec_all[:n_tr], "test": hist_vec_all[n_tr:]}

    feats = {}
    for s in ("train", "test"):
        cand = np.clip(util[s][0], 0, n_proto - 1)
        hist = frames[s].history_item_id.map(parse_hist)
        hp = np.stack([item_prof[h].mean(0) if len(h) else pop for h in hist])  # [n, P]
        r = np.arange(len(cand))[:, None]
        logp = logp_all[s][r, cand]
        lhp = np.log(hp[r, cand] + 1e-9)
        lpop = np.log(pop[cand])
        scal = np.concatenate([np.eye(K, dtype=np.float32)[None].repeat(len(cand), 0),
                               np.stack([lpop, lhp, lhp - lpop, logp - logp[:, :1]], -1)], -1)
        feats[s] = dict(logp=logp, scal=scal.astype(np.float32), h=hist_vec[s], e=mot_vec[cand])

    # Train / held-out split by user (rows with complete utilities only).
    users = frames["train"].user_id.to_numpy()
    uniq = np.unique(users)
    held = set(rng.choice(uniq, size=len(uniq) // 10, replace=False))
    complete = ~np.isnan(util["train"][1]).any(1)
    tr_rows = np.where(complete & ~np.isin(users, list(held)))[0]
    va_rows = np.where(complete & np.isin(users, list(held)))[0]
    te_rows = np.where(~np.isnan(util["test"][1]).any(1))[0]
    print(f"rows: train {len(tr_rows)}, held-out {len(va_rows)}, test {len(te_rows)}", flush=True)

    def T(s, rows):
        f = feats[s]
        return [torch.from_numpy(f[k][rows]) for k in ("logp", "scal", "h", "e")]

    model = Selector(feats["train"]["scal"].shape[-1], 64, 64)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    u_tr = torch.from_numpy(util["train"][1])

    def picks(s, rows):
        with torch.no_grad():
            return model(*T(s, rows)).argmax(1).numpy()

    results = {"args": vars(a)}
    base = {"retriever_top1": np.zeros(len(te_rows), np.int64),
            "random_top10": rng.integers(0, K, len(te_rows)),
            "oracle_best_of_10": np.nanargmax(util["test"][1][te_rows], 1)}
    for k, v in base.items():
        results[k] = evaluate(v, *util["test"], labels["test"], te_rows, trank["test"])
    best_va, best_state = -1e9, None
    diag = collections.defaultdict(list)
    for ep in range(a.epochs):
        perm = rng.permutation(tr_rows)
        stats = []
        for i in range(0, len(perm), a.batch):
            rows = perm[i:i + a.batch]
            logits = model(*T("train", rows))
            logpi = torch.log_softmax(logits, -1)
            ur = u_tr[rows]
            if a.method == "supervised":
                target = torch.softmax(ur / a.tau, -1)
                loss = -(target * logpi).sum(-1).mean()
            else:
                acts = torch.multinomial(logpi.exp(), a.group, replacement=True)       # [B, G]
                r = torch.gather(ur, 1, acts)
                adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
                pg = -(adv * torch.gather(logpi, 1, acts)).mean()
                prior = torch.log_softmax(torch.from_numpy(feats["train"]["logp"][rows]), -1)
                kl = (logpi.exp() * (logpi - prior)).sum(-1).mean()
                loss = pg + a.beta * kl
                stats.append(r.mean().item())
                pi = logpi.exp().detach()
                diag["entropy"].append(float(-(pi * logpi.detach()).sum(-1).mean()))
                diag["max_prob"].append(float(pi.max(-1).values.mean()))
                diag["kl_prior"].append(float(kl.detach()))
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()   # measure only
            diag["grad_norm"].append(gn)
            opt.step()
        va = evaluate(picks("train", va_rows), *util["train"], labels["train"], va_rows, trank["train"])
        msg = f"epoch {ep + 1}: held-out mean u {va['mean_u']:.4f}, picked rank0 {va['picked_rank0']:.3f}"
        if stats:
            msg += (f", sampled reward {np.mean(stats):.4f}, entropy {np.mean(diag['entropy']):.3f}, "
                    f"max prob {np.mean(diag['max_prob']):.3f}, KL to retriever {np.mean(diag['kl_prior']):.4f}, "
                    f"grad norm mean {np.mean(diag['grad_norm']):.4f} max {np.max(diag['grad_norm']):.4f}")
        diag = collections.defaultdict(list)
        print(msg, flush=True)
        if va["mean_u"] > best_va:
            best_va, best_state = va["mean_u"], {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pk = picks("test", te_rows)
    results[a.method] = evaluate(pk, *util["test"], labels["test"], te_rows, trank["test"])
    results[a.method]["held_out_mean_u"] = best_va
    results["held_out_retriever_top1_mean_u"] = evaluate(np.zeros(len(va_rows), np.int64), *util["train"],
                                                          labels["train"], va_rows)["mean_u"]
    results[a.method]["a_logp_weight"] = float(model.a)
    protos = json.load(open(C / "prototypes.json"))
    chosen = util["test"][0][te_rows, pk]
    ids, cnt = np.unique(chosen, return_counts=True)
    top = np.argsort(-cnt)[:8]
    results[a.method]["most_picked"] = [(protos[int(ids[i])]["motive"], round(float(cnt[i] / len(pk)), 4)) for i in top]
    ids0, cnt0 = np.unique(util["test"][0][te_rows, 0], return_counts=True)
    results["retriever_top1"]["most_picked"] = [(protos[int(ids0[i])]["motive"], round(float(cnt0[i] / len(te_rows)), 4))
                                                for i in np.argsort(-cnt0)[:8]]
    if a.save_picks:
        full = np.full(len(frames["test"]), n_proto, np.int64)   # n_proto = "No specific motive."
        full[te_rows] = chosen
        np.save(a.save_picks, full)
    for k in ("retriever_top1", "random_top10", a.method, "oracle_best_of_10"):
        r = results[k]
        print(f"{k:20s} test mean u {r['mean_u']:+.4f}  share of oracle-pair gain {r['share_of_oracle_pair_gain']:+.3f}  "
              f"u>0 {r['share_u_pos']:.3f}  label hit@1 {r['label_hit@1']:.4f}  rank0 {r['picked_rank0']:.3f}  "
              f"distinct {r['distinct_motives']}"
              + (f"  | target rank {r['target_mean_rank']:.2f} top1 {r['target_top1']:.4f} top5 {r['target_top5']:.4f}"
                 f" (null motive: rank {r['null_target_mean_rank']:.2f} top1 {r['null_target_top1']:.4f})"
                 if "target_mean_rank" in r else ""), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
