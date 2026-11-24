#!/usr/bin/env python3
"""Motive-interaction aligner (E8c): retrieve motive prototypes from the history with text-interaction alignment.

Motivation (Report/experiments/06_motive_mining): hard 0-2 motive labels are barely predictable from the history
(cls MLP retriever, Toys test MRR 0.290 vs popularity 0.249), while the continuous rationale space keeps a
target-free signal. Following IRLLRec (Wang et al., SIGIR'25: align LLM text intents with interaction intents),
we align, for every train interaction (history -> next item):

  q = Q(history features)                 interaction tower (same features as train_motive_retriever.py:
                                           SASRec state, gte history text, item motive profile, user's past motives)
  t = T(z_row)                            text tower on the row's own motive text embedding
                                           (mean of its v5 motive phrases, Qwen3-Embedding, centred like the corpus)
  p_k = T(c_k)                            prototypes = the same text tower on the prototype centroids
  e = E(SASRec item embedding of target)  item tower

Losses (all cosine / temperature):
  ce     multi-positive cross-entropy of softmax_k(q.p_k) on the row's hard prototype labels (retrieval objective)
  hm     InfoNCE q <-> t over the batch (history <-> its own motive text)          weight --w-hm
  mi     InfoNCE t <-> e over the batch (motive text <-> the item it led to)       weight --w-mi
  match  KL( softmax_k(t.p_k)/stopgrad || softmax_k(q.p_k) ) (IRLLRec-style intent matching)  weight --w-match
Retrieval at inference uses only q: scores_k = q.p_k / tau (+ learned log-prior bias). Metrics as
train_motive_retriever.py (Recall@k, MRR, Soft@5), plus next-item HR@10/NDCG@10 of ranking items by
sum_k pi_k (p_k . e_i) with pi = softmax over the top-10 retrieved prototypes (does the retrieved motive point to the
next item in interaction space?).

Writes <data>/<ds>/collab/motive_corpus/retriever/aligner_<tag>/ (results.json, scores_<split>.npy, model.pt).
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
from train_motive_retriever import metrics, normalise_rows

SPLITS = ("train", "valid", "test")
SASREC = {
    "Toys_and_Games": "/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/baselines/results/sasrec/"
                      "Toys_and_Games_5_2017-1-2018-11/models/SASRec/2026-10-31_12-00-01-480407.pkl",
    "CDs_and_Vinyl": "/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/baselines/results/sasrec/"
                     "CDs_and_Vinyl_5_2015-10-2018-11/models/SASRec/2026-09-09_00-49-25-209512.pkl",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="Toys_and_Games")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--corpus", default="motive_corpus")
    p.add_argument("--tag", default="")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--w-ce", type=float, default=1.0)
    p.add_argument("--w-hm", type=float, default=0.5)
    p.add_argument("--w-mi", type=float, default=0.5)
    p.add_argument("--w-match", type=float, default=0.5)
    p.add_argument("--crossfit-folds", type=int, default=0)
    p.add_argument("--relabel-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--relabel-rounds", type=int, default=0,
                   help="Interaction-fitted relabelling (E8d): after each round, every train row's motive becomes "
                        "the posterior p(m | item) * p(m | history) / prior^gamma (top --relabel-top, soft), with "
                        "p(m | history) from out-of-fold aligner scores; then the aligner is retrained.")
    p.add_argument("--relabel-top", type=int, default=2)
    p.add_argument("--prior-power", type=float, default=0.0, help="gamma: divide the posterior by prior^gamma")
    p.add_argument("--item-alpha", type=float, default=1.0, help="Dirichlet smoothing of p(m | item) toward the prior")
    p.add_argument("--select", choices=["mrr", "item"], default="mrr",
                   help="Early stopping on valid label MRR, or on valid next-item NDCG@10 (non-circular)")
    return p.parse_args()


def mlp(i, h, o, d):
    return torch.nn.Sequential(torch.nn.Linear(i, h), torch.nn.GELU(), torch.nn.Dropout(d), torch.nn.Linear(h, o))


class Aligner(torch.nn.Module):
    def __init__(self, in_q, in_t, in_e, protos, log_prior, args):
        super().__init__()
        self.Q = mlp(in_q, args.hidden, args.dim, args.dropout)
        self.T = mlp(in_t, args.hidden, args.dim, args.dropout)
        self.E = mlp(in_e, args.hidden // 2, args.dim, args.dropout)
        self.register_buffer("protos", protos)
        self.log_tau = torch.nn.Parameter(torch.tensor(float(np.log(args.tau))))
        self.bias = torch.nn.Parameter(torch.as_tensor(log_prior, dtype=torch.float32))

    def tau(self):
        return self.log_tau.exp().clamp(0.01, 1.0)

    def proto_vecs(self):
        return F.normalize(self.T(self.protos), dim=-1)

    def query(self, x):
        return F.normalize(self.Q(x), dim=-1)

    def scores(self, x):
        return self.query(x) @ self.proto_vecs().T / self.tau() + self.bias


def info_nce(a, b, tau):
    logits = a @ b.T / tau
    tgt = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))


def build_features(args, frames, y, n_proto, collab):
    cache = collab / args.corpus / "retriever" / "aligner_features.npz"
    if cache.exists():
        d = np.load(cache)
        return {s: d[f"x_{s}"] for s in SPLITS}, d["item_emb"]
    train = frames["train"]
    item_counts = collections.defaultdict(lambda: np.zeros(n_proto, dtype=np.float32))
    for item, row in zip(train.item_id, y["train"]):
        item_counts[int(item)] += row
    item_prof = {k: v / v.sum() for k, v in item_counts.items() if v.sum() > 0}

    def item_profile(frame):
        f = np.zeros((len(frame), n_proto), dtype=np.float32)
        for i, hist in enumerate(frame.history):
            vecs = [item_prof[h] for h in hist if h in item_prof]
            if vecs:
                f[i] = np.mean(vecs, 0)
        return f
    user_rows = collections.defaultdict(list)
    for j, (u, hist) in enumerate(zip(train.user_id, train.history)):
        user_rows[u].append((len(hist), j))

    def user_past(frame, split):
        f = np.zeros((len(frame), n_proto), dtype=np.float32)
        for i, (u, hist) in enumerate(zip(frame.user_id, frame.history)):
            rows = [j for L, j in user_rows.get(u, []) if split != "train" or L < len(hist)]
            if rows:
                f[i] = y["train"][rows].sum(0)
        return normalise_rows(f)
    n_items = sum(1 for _ in next((args.data_root / args.dataset / "info").glob("*.txt")).open(encoding="utf-8"))
    sasrec = SasrecEncoder(frames, n_items, SimpleNamespace(sasrec_ckpt=Path(SASREC[args.dataset])))
    x = {}
    for s, f in frames.items():
        text = np.load(collab / "hyp_cache" / ("hist_gte-large-en-v1.5.npy" if s == "train"
                                               else f"hist_gte-large-en-v1.5_{s}.npy"))
        x[s] = np.concatenate([sasrec.history_vectors(f), text, item_profile(f), user_past(f, s)], 1).astype(np.float32)
    np.savez(cache, item_emb=sasrec.item_emb, **{f"x_{s}": v for s, v in x.items()})
    return x, sasrec.item_emb


def row_text(args, collab, n_train):
    """Mean Qwen3 embedding of each train row's v5 motive phrases, centred like the corpus centroids."""
    recs = [None] * n_train
    texts = set()
    rec_all = {}
    for s in SPLITS:
        for line in open(collab / f"rationales_facet_{s}_v5.jsonl"):
            r = json.loads(line)
            ph = [m.strip().lower() for m in r["motives"]]
            texts.update(ph)
            if s == "train":
                recs[r["row"]] = ph
    all_texts = sorted(texts)
    emb = np.load(collab / "hyp_cache" / "cluster_phrase_qwen3.npy")
    assert len(all_texts) == len(emb)
    tid = {t: i for i, t in enumerate(all_texts)}
    units = json.loads((collab / "prototypes" / "compare" / "units_phrase_qwen3.json").read_text())
    mean = emb[[tid[u] for u in units]].mean(0)
    z = np.zeros((n_train, emb.shape[1]), np.float32)
    for i, ph in enumerate(recs):
        if ph:
            v = (emb[[tid[p] for p in ph]] - mean).mean(0)
            z[i] = v / (np.linalg.norm(v) + 1e-8)
    return z


def item_rank_metrics(model, xs, frame, ie, dev, k=10):
    """Next item ranked by the softmax-weighted mix of the top-10 retrieved prototypes (interaction space)."""
    with torch.no_grad():
        E = F.normalize(model.E(ie), dim=-1)
        P = model.proto_vecs()
        sc = model.scores(xs)
        top = sc.topk(10, 1)
        pi = F.softmax(top.values, 1)
        mix = F.normalize((pi[:, :, None] * P[top.indices]).sum(1), dim=-1)
        item_sc = mix @ E.T
        tgt = torch.as_tensor(frame.item_id.to_numpy(), device=dev)
        for i, h in enumerate(frame.history):
            ex = [j for j in h if j != tgt[i].item()]
            if ex:
                item_sc[i, ex] = -1e9
        rank = (item_sc > item_sc.gather(1, tgt[:, None])).sum(1)
        hr = (rank < k).float()
        return round(hr.mean().item(), 4), round((hr / torch.log2(rank.float() + 2)).mean().item(), 4)


def row_text_v6(root, n_train):
    """v6: mean Qwen3 embedding of each train row's "need -> item kind" statements, centred on the train mean."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from build_motive_corpus_v6 import statement
    texts = json.loads((root / "statements.json").read_text())
    tid = {t: i for i, t in enumerate(texts)}
    emb = np.load(root / "emb.npy")
    rows = [None] * n_train
    for line in open(root / "extract_train.jsonl"):
        r = json.loads(line)
        if "error" not in r and r.get("motives"):
            rows[r["row"]] = [tid[statement(m)] for m in r["motives"]]
    mean = emb[sorted({i for ids in rows if ids for i in ids})].mean(0)
    z = np.zeros((n_train, emb.shape[1]), np.float32)
    for i, ids in enumerate(rows):
        if ids:
            v = (emb[ids] - mean).mean(0)
            z[i] = v / (np.linalg.norm(v) + 1e-8)
    return z


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    collab = args.data_root / args.dataset / "collab"
    corpus = collab / args.corpus
    tag = args.tag or f"ce{args.w_ce}_hm{args.w_hm}_mi{args.w_mi}_match{args.w_match}"
    out = corpus / "retriever" / f"aligner_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    frames = {s: load_split(find_csv(args.data_root, args.dataset, s)) for s in SPLITS}
    labels = {s: json.loads((corpus / f"row_motives_{s}.json").read_text()) for s in SPLITS}
    n_proto = len(json.loads((corpus / "prototypes.json").read_text()))
    cent = np.load(corpus / "centroids.npy")
    text_emb = np.load(corpus / "motive_text_emb.npy") if (corpus / "motive_text_emb.npy").exists() \
        else cent
    text_sim = text_emb @ text_emb.T

    def multi_hot(ls):
        m = np.zeros((len(ls), n_proto), dtype=np.float32)
        for i, l in enumerate(ls):
            m[i, l] = 1.0
        return m
    y = {s: multi_hot(labels[s]) for s in SPLITS}
    xnp, item_emb = build_features(args, frames, y, n_proto, collab)
    z = row_text(args, collab, len(frames["train"])) if args.corpus == "motive_corpus" \
        else row_text_v6(corpus.parent, len(frames["train"]))

    x = {s: torch.from_numpy(v) for s, v in xnp.items()}
    mu, sd = x["train"].mean(0), x["train"].std(0) + 1e-6
    x = {s: ((v - mu) / sd).to(dev) for s, v in x.items()}
    ie = torch.from_numpy(item_emb)
    ie = ((ie - ie.mean(0)) / (ie.std(0) + 1e-6)).to(dev)
    tgt = torch.as_tensor(frames["train"].item_id.to_numpy(), device=dev)
    prior = (y["train"].sum(0) + 1) / (y["train"].sum() + n_proto)
    log_prior = np.log(prior)
    protos = torch.from_numpy(cent).float().to(dev)
    has_label = y["train"].sum(1) > 0

    # p(m | item) from the ORIGINAL train labels of all buyers of the item (fixed across rounds)
    n_items = item_emb.shape[0]
    item_cnt = np.zeros((n_items, n_proto), np.float32)
    for it, row in zip(frames["train"].item_id, y["train"]):
        item_cnt[it] += row
    for s in ("valid", "test"):  # valid/test targets: train buyers only (no own label)
        pass
    item_p = (item_cnt + args.item_alpha * prior[None]) / (item_cnt.sum(1, keepdims=True) + args.item_alpha)

    def fit(rows, ysoft, zt, valid_labels):
        rows = rows[has_label[rows]]
        rows = torch.from_numpy(rows).to(dev)
        model = Aligner(x["train"].shape[1], zt.shape[1], ie.shape[1], protos, log_prior, args).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best, best_state = -1.0, None
        for ep in range(args.epochs):
            model.train()
            perm = rows[torch.randperm(len(rows), device=dev)]
            for st in range(0, len(perm), args.batch_size):
                b = perm[st:st + args.batch_size]
                q = model.query(x["train"][b])
                P = model.proto_vecs()
                tau = model.tau()
                lq = q @ P.T / tau + model.bias
                loss = -(ysoft[b] * F.log_softmax(lq, 1)).sum(1).mean() * args.w_ce
                t = F.normalize(model.T(zt[b]), dim=-1)
                if args.w_hm:
                    loss = loss + args.w_hm * info_nce(q, t, tau)
                if args.w_mi:
                    e = F.normalize(model.E(ie[tgt[b]]), dim=-1)
                    loss = loss + args.w_mi * info_nce(t, e, tau)
                if args.w_match:
                    teacher = F.softmax((t @ P.T / tau).detach(), 1)
                    loss = loss + args.w_match * F.kl_div(F.log_softmax(q @ P.T / tau, 1), teacher,
                                                          reduction="batchmean")
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                if args.select == "mrr":
                    crit = metrics(model.scores(x["valid"]).cpu().numpy(), valid_labels, text_sim)["mrr"]
                else:
                    crit = item_rank_metrics(model, x["valid"], frames["valid"], ie, dev)[1]
            if crit > best:
                best, best_state = crit, {k: v.clone() for k, v in model.state_dict().items()}
            if ep % 5 == 4:
                print(f"  epoch {ep + 1}: loss {loss.item():.3f} valid {args.select} {crit:.4f}", flush=True)
        model.load_state_dict(best_state)
        model.eval()
        return model, best_state

    def oof_scores(ysoft, zt, valid_labels):
        oof = np.zeros((len(frames["train"]), n_proto), dtype=np.float32)
        folds = GroupKFold(n_splits=args.relabel_folds).split(np.zeros(len(oof)),
                                                               groups=frames["train"].user_id.to_numpy())
        for k, (fr, held) in enumerate(folds):
            m, _ = fit(fr, ysoft, zt, valid_labels)
            with torch.no_grad():
                oof[held] = m.scores(x["train"][torch.from_numpy(held).to(dev)]).cpu().numpy()
            print(f"  fold {k} done", flush=True)
        return oof

    def relabel(hist_logits, item_ids):
        hp = torch.softmax(torch.from_numpy(hist_logits), 1).numpy()
        post = item_p[item_ids] * hp / prior[None] ** args.prior_power
        top = np.argsort(-post, 1)[:, :args.relabel_top]
        w = np.take_along_axis(post, top, 1)
        w = w / w.sum(1, keepdims=True)
        soft = np.zeros_like(post)
        np.put_along_axis(soft, top, w, 1)
        return soft, [list(map(int, t)) for t in top]

    def label_stats(soft, hard, orig):
        top1 = soft.argmax(1)
        use = np.bincount(top1, minlength=n_proto)
        return {"distinct_top1": int((use > 0).sum()), "largest_share": round(float(use.max() / use.sum()), 4),
                "top1_in_v5_labels": round(float(np.mean([t in o for t, o in zip(top1, orig) if o])), 4)}

    ysoft_np = normalise_rows(y["train"])
    zt = torch.from_numpy(z).to(dev)
    valid_labels = labels["valid"]
    results, rounds = {}, args.relabel_rounds
    for rnd in range(rounds + 1):
        ysoft = torch.from_numpy(ysoft_np).to(dev)
        model, state = fit(np.arange(len(frames["train"])), ysoft, zt, valid_labels)
        with torch.no_grad():
            for s in ("valid", "test"):
                sc = model.scores(x[s]).cpu().numpy()
                r = metrics(sc, labels[s], text_sim)  # always against the ORIGINAL v5 labels
                r["item_HR@10"], r["item_NDCG@10"] = item_rank_metrics(model, x[s], frames[s], ie, dev)
                results[f"round{rnd}/{s}"] = r
                np.save(out / f"scores_{s}_round{rnd}.npy", sc.astype(np.float16))
        print(f"round {rnd}:", json.dumps({k: v for k, v in results.items() if k.startswith(f"round{rnd}/")}),
              flush=True)
        torch.save({"state_dict": state, "mu": mu, "sd": sd, "args": vars(args)}, out / f"model_round{rnd}.pt")
        if rnd == rounds:
            break
        # relabel train rows with the interaction-fitted posterior (out-of-fold history term)
        oof = oof_scores(ysoft, zt, valid_labels)
        ysoft_np, hard = relabel(oof, frames["train"].item_id.to_numpy())
        ysoft_np[~has_label] = 0
        results[f"round{rnd + 1}/train_labels"] = label_stats(ysoft_np[has_label], None,
                                                              [l for l, h in zip(labels["train"], has_label) if h])
        print(f"relabel -> round {rnd + 1}:", results[f"round{rnd + 1}/train_labels"], flush=True)
        # motive text of each row = posterior-weighted mix of its assigned prototype centroids
        zmix = ysoft_np @ cent
        zmix /= np.linalg.norm(zmix, axis=1, keepdims=True) + 1e-8
        zt = torch.from_numpy(zmix.astype(np.float32)).to(dev)
        # valid labels for early stopping on mrr: relabel valid with the current full model
        with torch.no_grad():
            vs, vh = relabel(model.scores(x["valid"]).cpu().numpy(), frames["valid"].item_id.to_numpy())
        valid_labels = [h if l else [] for h, l in zip(vh, labels["valid"])]
        json.dump(hard, open(out / f"train_labels_round{rnd + 1}.json", "w"))
    print(pd.DataFrame(results).T.to_string())
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
