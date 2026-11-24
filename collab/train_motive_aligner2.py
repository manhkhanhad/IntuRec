#!/usr/bin/env python3
"""Motive aligner v2 (E8g): user motive x item motive, IRLLRec-style, with averaged motive texts (cheap check).

IRLLRec (Wang et al., SIGIR'25) aligns a user's intent text with the user's interaction embedding and an item's intent
text with the item's interaction embedding. Here (v6 corpus, Toys):
  user-motive text z_u  = mean of the v6 statements of (a) the user's EARLIER train rows and (b) train rows that bought
                          one of the history items (target-free, usable at test time)
  item-motive text z_i  = mean of the v6 statements of all train buyers of the item (leave-one-out for the row itself)
  towers                q = Q(history features), e = E(SASRec item embedding), text tower T, prototypes p_k = T(c_k)
  pi_u = softmax_k(q.p_k / tau), pi_i = softmax_k(e.p_k / tau)
Losses: ce (pi_u vs the row's v6 labels), user (InfoNCE q<->T(z_u) + KL(softmax(T(z_u).p) || pi_u)),
        item (InfoNCE e<->T(z_i) + KL(softmax(T(z_i).p) || pi_i)),
        inter (next-item cross-entropy with score(u, i) = log sum_k pi_u(k) pi_i(k), all items).
Motives handed to MotiveRec (history only), two read-outs:
  user   top-k of pi_u
  joint  top-k of sum_i p(i | u) pi_u * pi_i over the 50 most likely items under the motive-space score
Judged as before: retrieval of the v6 labels, and the FIXED item scorer sum_k pi_k p(item | motive k) (v6 train links)
on the top-10 motives (plain retriever: test HR@10 0.0695); plus HR@10 of the motive-space score itself.
Early stopping on valid fixed-scorer NDCG@10.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from build_motive_corpus_v6 import statement  # noqa: E402
from build_support import find_csv, load_split  # noqa: E402
from train_motive_aligner import build_features, info_nce, mlp  # noqa: E402
from train_motive_retriever import metrics, normalise_rows  # noqa: E402

SPLITS = ("train", "valid", "test")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="Toys_and_Games")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--corpus", default="motive_corpus_v6/P1000")
    p.add_argument("--tag", default="")
    p.add_argument("--pca", type=int, default=512)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--w-ce", type=float, default=1.0)
    p.add_argument("--w-user", type=float, default=0.5)
    p.add_argument("--w-item", type=float, default=0.5)
    p.add_argument("--w-inter", type=float, default=0.5)
    p.add_argument("--joint-items", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--crossfit-folds", type=int, default=0,
                   help="> 0: also write out-of-fold user scores for train rows (scores_user_train.npy)")
    return p.parse_args()


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    collab = args.data_root / args.dataset / "collab"
    corpus = collab / args.corpus
    root = corpus.parent
    tag = args.tag or f"ce{args.w_ce}_u{args.w_user}_i{args.w_item}_x{args.w_inter}"
    out = corpus / "retriever" / f"aligner2_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    frames = {s: load_split(find_csv(args.data_root, args.dataset, s)) for s in SPLITS}
    tr = frames["train"]
    labels = {s: json.loads((corpus / f"row_motives_{s}.json").read_text()) for s in SPLITS}
    K = len(json.loads((corpus / "prototypes.json").read_text()))
    cent = np.load(corpus / "centroids.npy")
    y = {}
    for s in SPLITS:
        m = np.zeros((len(labels[s]), K), np.float32)
        for i, l in enumerate(labels[s]):
            m[i, l] = 1
        y[s] = m
    xnp, item_emb = build_features(args, frames, y, K, collab)
    n_items = item_emb.shape[0]

    # ---- motive texts (v6 statements, centred, PCA to --pca dims; centroids projected the same way)
    texts = json.loads((root / "statements.json").read_text())
    tid = {t: i for i, t in enumerate(texts)}
    emb = np.load(root / "emb.npy")
    row_ids = [None] * len(tr)
    for line in open(root / "extract_train.jsonl"):
        r = json.loads(line)
        if "error" not in r and r.get("motives"):
            row_ids[r["row"]] = [tid[statement(m)] for m in r["motives"]]
    used = sorted({i for ids in row_ids if ids for i in ids})
    mean = emb[used].mean(0)
    _, _, V = torch.pca_lowrank(torch.from_numpy(emb[used] - mean), q=args.pca, center=False)
    V = V.numpy()
    zrow = np.zeros((len(tr), args.pca), np.float32)
    has_z = np.zeros(len(tr), bool)
    for i, ids in enumerate(row_ids):
        if ids:
            zrow[i] = unit((emb[ids] - mean).mean(0)) @ V
            has_z[i] = True
    zrow = unit(zrow) * has_z[:, None]
    cent_p = unit(cent @ V) if cent.shape[1] == V.shape[0] else None
    assert cent_p is not None, "centroids are not in the statement-embedding space"

    # item-motive text: sum / count of train buyers' row texts (leave-one-out for train rows)
    isum = np.zeros((n_items, args.pca), np.float32)
    icnt = np.zeros(n_items, np.float32)
    np.add.at(isum, tr.item_id.to_numpy()[has_z], zrow[has_z])
    np.add.at(icnt, tr.item_id.to_numpy()[has_z], 1)
    # user-motive text: earlier train rows of the user + train rows that bought a history item
    user_rows = collections.defaultdict(list)
    for j, (u, h) in enumerate(zip(tr.user_id, tr.history)):
        user_rows[u].append((len(h), j))

    def user_text(frame, split):
        z = np.zeros((len(frame), args.pca), np.float32)
        for i, (u, h) in enumerate(zip(frame.user_id, frame.history)):
            own = [j for L, j in user_rows.get(u, []) if (split != "train" or L < len(h)) and has_z[j]]
            v = np.zeros(args.pca, np.float32)
            n = 0
            if own:
                v += unit(zrow[own].mean(0))
                n += 1
            hs = [it for it in h if icnt[it] > 0]
            if hs:
                v += unit(isum[hs].sum(0) / icnt[hs].sum())
                n += 1
            if n:
                z[i] = unit(v)
        return z
    zu = {s: user_text(frames[s], s) for s in SPLITS}
    print("rows with a user-motive text:", {s: round(float((np.abs(v).sum(1) > 0).mean()), 3) for s, v in zu.items()},
          flush=True)

    # fixed item scorer p(item | motive) from v6 train links (same as the pilot scorer)
    cnt = np.zeros((n_items, K), np.float32)
    for it, ms in zip(tr.item_id, labels["train"]):
        cnt[it, ms] += 1
    pop = np.bincount(tr.item_id, minlength=n_items) + 1e-3
    pop_p = pop / pop.sum()
    p_im = torch.from_numpy(((cnt + 0.1 * pop_p[:, None]) / (cnt.sum(0, keepdims=True) + 0.1)).T.copy()).float().to(dev)

    # ---- tensors
    x = {s: torch.from_numpy(v) for s, v in xnp.items()}
    mu, sd = x["train"].mean(0), x["train"].std(0) + 1e-6
    x = {s: ((v - mu) / sd).to(dev) for s, v in x.items()}
    ie = torch.from_numpy(item_emb)
    ie = ((ie - ie.mean(0)) / (ie.std(0) + 1e-6)).to(dev)
    zu_t = {s: torch.from_numpy(v).to(dev) for s, v in zu.items()}
    zrow_t = torch.from_numpy(zrow).to(dev)
    has_z_t = torch.from_numpy(has_z.astype(np.float32)).to(dev)
    isum_t, icnt_t = torch.from_numpy(isum).to(dev), torch.from_numpy(icnt).to(dev)
    tgt = torch.as_tensor(tr.item_id.to_numpy(), device=dev)
    ysoft = torch.from_numpy(normalise_rows(y["train"])).to(dev)
    log_prior = torch.from_numpy(np.log((y["train"].sum(0) + 1) / (y["train"].sum() + K))).float().to(dev)
    C = torch.from_numpy(cent_p).float().to(dev)
    hist_mask = {s: [list(h) for h in frames[s].history] for s in SPLITS}

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.Q = mlp(x["train"].shape[1], args.hidden, args.dim, args.dropout)
            self.E = mlp(ie.shape[1], args.hidden // 2, args.dim, args.dropout)
            self.T = mlp(args.pca, args.hidden, args.dim, args.dropout)
            self.log_tau = torch.nn.Parameter(torch.tensor(float(np.log(args.tau))))
            self.bias = torch.nn.Parameter(log_prior.clone())

        def tau(self):
            return self.log_tau.exp().clamp(0.01, 1.0)

        def protos(self):
            return F.normalize(self.T(C), dim=-1)

        def q(self, xb):
            return F.normalize(self.Q(xb), dim=-1)

        def e(self, items):
            return F.normalize(self.E(ie[items]), dim=-1)

        def t(self, z):
            return F.normalize(self.T(z), dim=-1)

    all_items = torch.arange(n_items, device=dev)

    def readouts(split):
        """user / joint motive scores (logits) and the motive-space item scores for one split."""
        with torch.no_grad():
            P = model.protos()
            tau = model.tau()
            lu = model.q(x[split]) @ P.T / tau + model.bias
            pu = F.softmax(lu, 1)
            pi = F.softmax(model.e(all_items) @ P.T / tau, 1)          # n_items x K
            item_sc = torch.log(pu @ pi.T + 1e-12)                       # rows x n_items
            for r, h in enumerate(hist_mask[split]):
                tgt_r = int(frames[split].item_id.iat[r])
                ex = [j for j in h if j != tgt_r]
                if ex:
                    item_sc[r, ex] = -1e9
            top = item_sc.topk(args.joint_items, 1)
            w = F.softmax(top.values, 1)
            joint = pu * (w[:, :, None] * pi[top.indices]).sum(1)
            return lu, torch.log(joint + 1e-12), item_sc

    def item_metrics(scores, split, k=10):
        t_ = torch.as_tensor(frames[split].item_id.to_numpy(), device=dev)
        rank = (scores > scores.gather(1, t_[:, None])).sum(1)
        hr = (rank < k).float()
        return round(hr.mean().item(), 4), round((hr / torch.log2(rank.float() + 2)).mean().item(), 4)

    def fixed(motive_logits, split):
        top = motive_logits.topk(10, 1)
        pi_ = F.softmax(top.values, 1)
        sc = (pi_[:, :, None] * p_im[top.indices]).sum(1)
        for r, h in enumerate(hist_mask[split]):
            tgt_r = int(frames[split].item_id.iat[r])
            ex = [j for j in h if j != tgt_r]
            if ex:
                sc[r, ex] = -1
        return item_metrics(sc, split)

    def fit(train_idx):
        nonlocal model
        model = M().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        rows = torch.from_numpy(train_idx).to(dev)
        best, best_state = -1, None
        for ep in range(args.epochs):
            model.train()
            perm = rows[torch.randperm(len(rows), device=dev)]
            for st in range(0, len(perm), args.batch_size):
                b = perm[st:st + args.batch_size]
                P = model.protos()
                tau = model.tau()
                q = model.q(x["train"][b])
                lu = q @ P.T / tau
                loss = args.w_ce * -(ysoft[b] * F.log_softmax(lu + model.bias, 1)).sum(1).mean()
                if args.w_user:
                    ok = zu_t["train"][b].abs().sum(1) > 0
                    if ok.sum() > 1:
                        tu = model.t(zu_t["train"][b][ok])
                        loss = loss + args.w_user * (info_nce(q[ok], tu, tau) + F.kl_div(
                            F.log_softmax(lu[ok], 1), F.softmax((tu @ P.T / tau).detach(), 1), reduction="batchmean"))
                e = model.e(tgt[b])
                li = e @ P.T / tau
                if args.w_item:
                    n_loo = icnt_t[tgt[b]] - has_z_t[b]
                    ok = n_loo > 0
                    if ok.sum() > 1:
                        zi = (isum_t[tgt[b]] - zrow_t[b] * has_z_t[b][:, None]) / n_loo.clamp(min=1)[:, None]
                        ti = model.t(F.normalize(zi[ok], dim=-1))
                        loss = loss + args.w_item * (info_nce(e[ok], ti, tau) + F.kl_div(
                            F.log_softmax(li[ok], 1), F.softmax((ti @ P.T / tau).detach(), 1), reduction="batchmean"))
                if args.w_inter:
                    pu = F.softmax(lu + model.bias, 1)
                    pi_all = F.softmax(model.e(all_items) @ P.T / tau, 1)
                    sc = torch.log(pu @ pi_all.T + 1e-12)
                    loss = loss + args.w_inter * F.cross_entropy(sc, tgt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            lu_v, lj_v, _ = readouts("valid")
            crit = max(fixed(lu_v, "valid")[1], fixed(lj_v, "valid")[1])
            if crit > best:
                best, best_state = crit, {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  epoch {ep + 1}: loss {loss.item():.3f} valid fixed NDCG@10 {crit:.4f}", flush=True)
        model.load_state_dict(best_state)
        model.eval()
        return best_state

    model = None
    labelled = np.where(y["train"].sum(1) > 0)[0]
    if args.crossfit_folds > 0:
        oof = np.zeros((len(tr), K), np.float32)
        fold = np.random.RandomState(args.seed).randint(0, args.crossfit_folds, len(tr))
        for f in range(args.crossfit_folds):
            print(f"crossfit fold {f + 1}/{args.crossfit_folds}", flush=True)
            fit(labelled[fold[labelled] != f])
            with torch.no_grad():
                lu_tr = model.q(x["train"]) @ model.protos().T / model.tau() + model.bias
            oof[fold == f] = lu_tr[torch.from_numpy(fold == f).to(dev)].cpu().numpy()
        np.save(out / "scores_user_train.npy", oof.astype(np.float16))
        r = metrics(oof, labels["train"], cent_p @ cent_p.T)
        print("train OOF user:", {k: r[k] for k in ("recall@1", "recall@10", "mrr")}, flush=True)
    best_state = fit(labelled)
    res = {}
    for s in ("valid", "test"):
        lu, lj, isc = readouts(s)
        for name, sc in (("user", lu), ("joint", lj)):
            r = metrics(sc.cpu().numpy(), labels[s], cent_p @ cent_p.T)
            r["fixed_HR@10"], r["fixed_NDCG@10"] = fixed(sc, s)
            res[f"{s}/{name}"] = r
            np.save(out / f"scores_{name}_{s}.npy", sc.cpu().numpy().astype(np.float16))
        res[f"{s}/motive_space_items"] = dict(zip(("HR@10", "NDCG@10"), item_metrics(isc, s)))
    print(pd.DataFrame(res).T.to_string())
    (out / "results.json").write_text(json.dumps(res, indent=2))
    torch.save({"state_dict": best_state, "args": vars(args)}, out / "model.pt")


if __name__ == "__main__":
    main()
