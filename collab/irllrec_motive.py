#!/usr/bin/env python3
"""IRLLRec on MotiveRec's motives (E8h): predict the user motive (need) and the item motive (item kind).

Faithful port of the official IRLLRec code (github.com/wangyu0627/IRLLRec, encoder/models/general_cf/lightgcn_int.py,
config lightgcn_int.yml "amazon"), adapted to sequential data in ONE way, documented here:
  * "user" node = one interaction row (its history), because rows of the same user have different histories and
    targets (time split; 2.6 test rows per test user). Graph edges = row -> history items, for every split (past
    interactions only, so no target leaks into message passing). BPR positives = train rows' targets.
Everything else follows the official code:
  * LightGCN, 3 layers, embedding 32, sum of layers (embeds = sum(embeds_list)), score = user . item, BPR + L2 1e-7.
  * RLMRec profile alignment (kd_loss, weight 1e-2, temp 0.2): profile = mxbai text of the item (GEARec item
    embeddings); row profile = mean mxbai of its history items. One shared 2-layer LeakyReLU MLP (self.mlp).
  * K = 128 learnable user / item intent prototypes; per layer intent-aware embeds softmax(e C) C^T; adaptive mask
    alpha = (cos(head, tail) + 1) / 2 normalised by its row sum; iaa embeds = masked propagation of the layer input;
    intent embeds = mean over layers (kd_int: InfoNCE, weight 2e-2, temp 0.2) against int_mlp(intent text).
  * kd_int_2 (multiplicative Gaussian noise, ssl_con_loss, weight 1e-7) and the momentum (0.999) "itm" term (weight
    1e-7) as in the code. Only change: kd_int_2 runs on 4096 random nodes per step (the full N x N matrix does not fit).
  * Adam lr 1e-3, batch 4096, evaluation every 3 epochs, patience 5 (early stopping on valid Recall@10).
Intent texts (the user's request): the v6 motive "need -> item kind" is split in two.
  * user intent text of a train row = mxbai("need") of that row's v6 motives (mean of its 1-2 needs);
  * item intent text of an item = mean mxbai("item kind") over the item's train buyers.
  Rows / items without a text are left out of kd_int (masked), exactly where the official code would have none.
Read-outs after training (history only, i.e. "given an interaction"):
  * user motive score of v6 prototype k = cos(int_mlp(need_k), user intent embed of the row);
  * item motive score of k = sum over the 50 most likely next items (softmax of LightGCN scores) of
    cos(int_mlp(kind_k), item intent embed);  "cond" adds log p(kind cluster | predicted need) (given interaction +
    user motive, predict the item motive);
  * prototype score = z(user score) + z(item score); top-10 -> the same fixed item scorer as E8b/E8g
    (plain retriever test HR@10 0.0695) and retrieval of the v6 labels.
Stages: prep (CPU mxbai encodings) | train.
"""
import argparse
import ast
import collections
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from train_motive_retriever import metrics  # noqa: E402

SPLITS = ("train", "valid", "test")
GEA = Path("/common/home/users/m/mk.ngohuu.2024/research/GEARec_R3_toys")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["prep", "train"], required=True)
    p.add_argument("--dataset", default="Toys_and_Games")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--corpus", default="motive_corpus_v6/P1000")
    p.add_argument("--tag", default="irllrec")
    p.add_argument("--embedding-size", type=int, default=32)
    p.add_argument("--intent-num", type=int, default=128)
    p.add_argument("--layer-num", type=int, default=3)
    p.add_argument("--reg-weight", type=float, default=1e-7)
    p.add_argument("--kd-weight", type=float, default=1e-2)
    p.add_argument("--kd-temperature", type=float, default=0.2)
    p.add_argument("--kd-int-weight", type=float, default=2e-2)
    p.add_argument("--kd-int-temperature", type=float, default=0.2)
    p.add_argument("--kd-int-weight-2", type=float, default=1e-7)
    p.add_argument("--kd-int-weight-3", type=float, default=1e-7)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--test-step", type=int, default=3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--next-items", type=int, default=50)
    p.add_argument("--cond-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2023)
    return p.parse_args()


# ---------------------------------------------------------------- official loss helpers (loss_utils.py)
def cal_bpr_loss(anc, pos, neg):
    return t.sum(F.softplus((anc * neg).sum(-1) - (anc * pos).sum(-1)))


def cal_infonce_loss(e1, e2, all_e2, temp=1.0):
    n1 = e1 / t.sqrt(1e-8 + e1.square().sum(-1, keepdim=True))
    n2 = e2 / t.sqrt(1e-8 + e2.square().sum(-1, keepdim=True))
    na = all_e2 / t.sqrt(1e-8 + all_e2.square().sum(-1, keepdim=True))
    nume = -(n1 * n2 / temp).sum(-1)
    deno = t.log(t.sum(t.exp(n1 @ na.T / temp), dim=-1))
    return (nume + deno).sum()


def ssl_con_loss(x, y, temp=1.0):
    x, y = F.normalize(x), F.normalize(y)
    mole = t.exp(t.sum(x * y, dim=1) / temp)
    deno = t.sum(t.exp(x @ y.T / temp), dim=1)
    return -t.log(mole / (deno + 1e-8) + 1e-8).mean()


def reg_params(model):
    return sum(W.norm(2).square() for W in model.parameters() if W.requires_grad)


# ---------------------------------------------------------------- data
def load(args):
    d = args.data_root / args.dataset
    frames = {s: pd.read_csv(glob.glob(str(d / s / "*.csv"))[0]) for s in SPLITS}
    for f in frames.values():
        f["h"] = f.history_item_id.map(ast.literal_eval)
    return frames


def stage_prep(args):
    from sentence_transformers import SentenceTransformer
    root = args.data_root / args.dataset / "collab" / "motive_corpus_v6"
    out = root / "irllrec"
    out.mkdir(exist_ok=True)
    needs, kinds = {}, {}
    rows = {}
    for line in open(root / "extract_train.jsonl"):
        r = json.loads(line)
        if "error" in r or not r.get("motives"):
            continue
        ms = [(m["need"].strip().lower(), m["item_kind"].strip().lower()) for m in r["motives"]]
        rows[r["row"]] = ms
        for n, k in ms:
            needs.setdefault(n, len(needs))
            kinds.setdefault(k, len(kinds))
    protos = json.loads((root / Path(args.corpus).name / "prototypes.json").read_text())
    st = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cpu")
    enc = lambda xs: st.encode(xs, batch_size=128, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    print(len(needs), "needs", len(kinds), "kinds", flush=True)
    np.save(out / "need_emb.npy", enc(list(needs)))
    np.save(out / "kind_emb.npy", enc(list(kinds)))
    np.save(out / "proto_need_emb.npy", enc([p["need"].lower() for p in protos]))
    np.save(out / "proto_kind_emb.npy", enc([p["item_kind"].lower() for p in protos]))
    (out / "texts.json").write_text(json.dumps({"needs": list(needs), "kinds": list(kinds),
                                                "rows": {str(k): [[needs[n], kinds[q]] for n, q in v]
                                                         for k, v in rows.items()}}))
    print("prep done", flush=True)


# ---------------------------------------------------------------- model (lightgcn_int.py, rows as users)
class LightGCNInt(nn.Module):
    def __init__(self, args, n_user, n_item, adj, h_list, t_list, usrprf, itmprf, usrint, itmint):
        super().__init__()
        init = nn.init.xavier_uniform_
        d, K = args.embedding_size, args.intent_num
        self.args, self.n_user, self.n_item, self.adj = args, n_user, n_item, adj
        self.user_embeds = nn.Parameter(init(t.empty(n_user, d)))
        self.item_embeds = nn.Parameter(init(t.empty(n_item, d)))
        self.user_intent = nn.Parameter(init(t.empty(d, K)))
        self.item_intent = nn.Parameter(init(t.empty(d, K)))
        self.h_list, self.t_list = h_list, t_list
        self.N = n_user + n_item
        self.usrprf, self.itmprf, self.usrint, self.itmint = usrprf, itmprf, usrint, itmint
        D = usrprf.shape[1]
        self.mlp = nn.Sequential(nn.Linear(D, (D + d) // 2), nn.LeakyReLU(), nn.Linear((D + d) // 2, d))
        Di = usrint.shape[1]
        self.int_mlp = nn.Sequential(nn.Linear(Di, (Di + d) // 2), nn.LeakyReLU(), nn.Linear((Di + d) // 2, d))
        self.int_mlp_m = nn.Sequential(nn.Linear(Di, (Di + d) // 2), nn.LeakyReLU(), nn.Linear((Di + d) // 2, d))
        for m in list(self.mlp) + list(self.int_mlp) + list(self.int_mlp_m):
            if isinstance(m, nn.Linear):
                init(m.weight)
        for p, pm in zip(self.int_mlp.parameters(), self.int_mlp_m.parameters()):
            pm.data.copy_(p.data)
            pm.requires_grad = False
        self.momentum = 0.999

    @t.no_grad()
    def momentum_update(self):
        for p, pm in zip(self.int_mlp.parameters(), self.int_mlp_m.parameters()):
            pm.data = pm.data * self.momentum + p.data * (1.0 - self.momentum)

    def adaptive_mask(self, head, tail):
        head, tail = F.normalize(head), F.normalize(tail)
        alpha = (t.sum(head * tail, dim=1).view(-1) + 1) / 2
        deg = t.zeros(self.N, device=alpha.device).index_add_(0, self.h_list, alpha)
        d_inv = deg.pow(-1).nan_to_num(0, 0, 0)
        return d_inv[self.h_list] * alpha

    def forward(self):
        embeds = t.cat([self.user_embeds, self.item_embeds], 0)
        embeds_list, iaa = [embeds], []
        for i in range(self.args.layer_num):
            embeds = t.sparse.mm(self.adj, embeds_list[-1])
            u, it = t.split(embeds_list[i], [self.n_user, self.n_item], 0)
            ui = t.softmax(u @ self.user_intent, 1) @ self.user_intent.T
            ii = t.softmax(it @ self.item_intent, 1) @ self.item_intent.T
            intl = t.cat([ui, ii], 0)
            vals = self.adaptive_mask(intl[self.h_list], intl[self.t_list])
            G = t.sparse_coo_tensor(t.stack([self.h_list, self.t_list]), vals, (self.N, self.N))
            iaa.append(t.sparse.mm(G, embeds_list[-1]))
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
        return embeds[:self.n_user], embeds[self.n_user:], iaa

    def cal_loss(self, ancs, poss, negs, umask, imask):
        a = self.args
        U, I, iaa = self.forward()
        anc, pos, neg = U[ancs], I[poss], I[negs]
        usrprf, itmprf = self.mlp(self.usrprf), self.mlp(self.itmprf)
        bpr = cal_bpr_loss(anc, pos, neg) / len(ancs)
        reg = a.reg_weight * reg_params(self)
        kd = (cal_infonce_loss(anc, usrprf[ancs], usrprf, a.kd_temperature)
              + cal_infonce_loss(pos, itmprf[poss], itmprf[poss], a.kd_temperature)
              + cal_infonce_loss(neg, itmprf[negs], itmprf[negs], a.kd_temperature)) / len(ancs) * a.kd_weight
        int_embeds = t.stack(iaa).mean(0)
        Ui, Ii = t.split(int_embeds, [self.n_user, self.n_item], 0)
        usrint, itmint = self.int_mlp(self.usrint), self.int_mlp(self.itmint)
        # texts exist only for train rows with a v6 motive and items with train buyers -> mask the others
        ua, pa, na = ancs[umask[ancs]], poss[imask[poss]], negs[imask[negs]]
        all_u = usrint[umask]
        kd_int = (cal_infonce_loss(Ui[ua], usrint[ua], all_u, a.kd_int_temperature)
                  + cal_infonce_loss(Ii[pa], itmint[pa], itmint[pa], a.kd_int_temperature)
                  + cal_infonce_loss(Ii[na], itmint[na], itmint[na], a.kd_int_temperature)) / len(ancs) * a.kd_int_weight
        # kd_int_2 (translation alignment) on 4096 random nodes
        sub = t.randint(0, self.N, (4096,), device=U.device)
        all_e = t.cat([U, I], 0)[sub]
        all_int = t.cat([usrint, itmint], 0)[sub]
        kd2 = ssl_con_loss(all_int + all_int * t.randn_like(all_int), all_e + all_e * t.randn_like(all_e)) * a.kd_int_weight_2
        self.momentum_update()
        with t.no_grad():
            int_m = t.cat([self.int_mlp_m(self.usrint), self.int_mlp_m(self.itmint)], 0)[sub]
        itm = t.sum(F.log_softmax(all_int, 1) * F.softmax(int_m, 1), 1).mean()
        itm = (0.4 * kd2 - 0.6 * itm) * a.kd_int_weight_3
        loss = bpr + reg + kd + kd_int + kd2 + itm
        return loss, {"bpr": bpr.item(), "kd": kd.item(), "kd_int": kd_int.item()}


def stage_train(args):
    t.manual_seed(args.seed)
    np.random.seed(args.seed)
    frames = load(args)
    collab = args.data_root / args.dataset / "collab"
    root = collab / "motive_corpus_v6"
    corpus = collab / args.corpus
    pdir = root / "irllrec"
    texts = json.loads((pdir / "texts.json").read_text())
    need_emb, kind_emb = np.load(pdir / "need_emb.npy"), np.load(pdir / "kind_emb.npy")
    n_tr, n_va, n_te = (len(frames[s]) for s in SPLITS)
    off = {"train": 0, "valid": n_tr, "test": n_tr + n_va}
    n_user = n_tr + n_va + n_te
    item_mx = t.load(GEA / "outputs" / "item_emb_mxbai.pt", map_location="cpu").float().numpy()
    item_mx /= np.linalg.norm(item_mx, axis=1, keepdims=True) + 1e-8
    n_item = item_mx.shape[0]

    # graph: row -> history items (all splits); symmetric, LightGCN normalisation
    rows, cols = [], []
    for s in SPLITS:
        for r, h in enumerate(frames[s].h):
            for it in set(h):
                rows.append(off[s] + r)
                cols.append(it)
    rows, cols = np.array(rows), np.array(cols)
    h_list = np.concatenate([rows, cols + n_user])
    t_list = np.concatenate([cols + n_user, rows])
    N = n_user + n_item
    deg = np.bincount(h_list, minlength=N).astype(np.float32)
    dinv = np.power(deg, -0.5, where=deg > 0, out=np.zeros_like(deg))
    vals = dinv[h_list] * dinv[t_list]
    adj = t.sparse_coo_tensor(t.from_numpy(np.stack([h_list, t_list])), t.from_numpy(vals), (N, N)).coalesce()
    h_list, t_list = t.from_numpy(h_list), t.from_numpy(t_list)
    print(f"graph: {n_user} rows, {n_item} items, {len(rows)} history edges", flush=True)

    # profiles (RLMRec): item = mxbai item text; row = mean mxbai of its history items
    usrprf = np.zeros((n_user, item_mx.shape[1]), np.float32)
    for s in SPLITS:
        for r, h in enumerate(frames[s].h):
            usrprf[off[s] + r] = item_mx[h].mean(0)
    # intent texts: row need (train rows), item kind (mean over train buyers)
    usrint = np.zeros((n_user, need_emb.shape[1]), np.float32)
    umask = np.zeros(n_user, bool)
    itmint = np.zeros((n_item, kind_emb.shape[1]), np.float32)
    icnt = np.zeros(n_item, np.float32)
    row_pairs = {int(k): v for k, v in texts["rows"].items()}
    for r, pairs in row_pairs.items():
        usrint[r] = need_emb[[p[0] for p in pairs]].mean(0)
        umask[r] = True
        it = frames["train"].item_id.iat[r]
        itmint[it] += kind_emb[[p[1] for p in pairs]].mean(0)
        icnt[it] += 1
    imask = icnt > 0
    itmint[imask] /= icnt[imask, None]
    print(f"intent texts: rows {umask.sum()} / {n_user}, items {imask.sum()} / {n_item}", flush=True)

    dev = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(f"device {dev}", flush=True)
    T = lambda x: t.from_numpy(x).to(dev)
    model = LightGCNInt(args, n_user, n_item, adj.to(dev), h_list.to(dev), t_list.to(dev),
                        T(usrprf), T(item_mx), T(usrint), T(itmint)).to(dev)
    opt = t.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0)
    umask_t, imask_t = T(umask), T(imask)
    targets = {s: frames[s].item_id.to_numpy() for s in SPLITS}

    def scores(U, I, split):
        sc = U[off[split]:off[split] + len(frames[split])] @ I.T
        for r, h in enumerate(frames[split].h):
            ex = [j for j in h if j != targets[split][r]]
            if ex:
                sc[r, ex] = -1e9
        return sc

    def rec_metrics(sc, split):
        tg = T(targets[split]).long()
        rank = (sc > sc.gather(1, tg[:, None])).sum(1)
        out = {}
        for k in (1, 5, 10):
            hr = (rank < k).float()
            out[f"HR@{k}"] = round(hr.mean().item(), 4)
            out[f"NDCG@{k}"] = round((hr / t.log2(rank.float() + 2)).mean().item(), 4)
        return out

    train_rows = np.arange(n_tr)
    best, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        perm = np.random.permutation(train_rows)
        for st in range(0, len(perm), args.batch_size):
            b = perm[st:st + args.batch_size]
            ancs = T(b).long()
            poss = T(targets["train"][b]).long()
            negs = t.randint(0, n_item, (len(b),), device=dev)
            loss, parts = model.cal_loss(ancs, poss, negs, umask_t, imask_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % args.test_step == 0:
            model.eval()
            with t.no_grad():
                U, I, _ = model.forward()
                m = rec_metrics(scores(U, I, "valid"), "valid")
            print(f"epoch {ep + 1} ({time.time() - t0:.0f}s/ep) loss {loss.item():.4f} {parts} valid {m}", flush=True)
            if m["HR@10"] > best:
                best, bad = m["HR@10"], 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
    model.load_state_dict(best_state)
    model.eval()
    out = corpus / "retriever" / f"irllrec_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    t.save({"state_dict": best_state, "args": vars(args)}, out / "model.pt")

    # ---------------- read-outs
    protos = json.loads((corpus / "prototypes.json").read_text())
    K = len(protos)
    pn, pk = T(np.load(pdir / "proto_need_emb.npy")).float(), T(np.load(pdir / "proto_kind_emb.npy")).float()
    labels = {s: json.loads((corpus / f"row_motives_{s}.json").read_text()) for s in SPLITS}
    # p(kind proto | need proto) from co-occurrence of the train labels' need/kind parts (cond read-out)
    tr_lab = labels["train"]
    cnt = np.zeros((n_item, K), np.float32)
    for it, ms in zip(targets["train"], tr_lab):
        cnt[it, ms] += 1
    pop = np.bincount(targets["train"], minlength=n_item) + 1e-3
    p_im = T(((cnt + 0.1 * (pop / pop.sum())[:, None]) / (cnt.sum(0, keepdims=True) + 0.1)).T.copy()).float()
    # need-kind compatibility: cosine between prototype need and kind texts of co-labelled pairs (protos share a
    # need/kind only through their own text, so use the prototypes' text similarity in mxbai space)
    need_sim = F.normalize(pn, dim=-1) @ F.normalize(pn, dim=-1).T          # K x K
    res = {}
    with t.no_grad():
        U, I, iaa = model.forward()
        Ui, Ii = t.split(t.stack(iaa).mean(0), [n_user, n_item], 0)
        PN = F.normalize(model.int_mlp(pn), dim=-1)
        PK = F.normalize(model.int_mlp(pk), dim=-1)
        Iin = F.normalize(Ii, dim=-1)
        for s in ("valid", "test"):
            rows_s = slice(off[s], off[s] + len(frames[s]))
            sc = scores(U, I, s)
            res[f"{s}/lightgcn_rec"] = rec_metrics(sc, s)
            us = F.normalize(Ui[rows_s], dim=-1) @ PN.T                     # user motive (need) score
            top = sc.topk(args.next_items, 1)
            w = F.softmax(top.values, 1)
            ks = (w[:, :, None] * (Iin[top.indices] @ PK.T)).sum(1)          # item motive (kind) score
            z = lambda x: (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
            # cond: item-kind score re-weighted by the kinds of prototypes whose need matches the predicted need
            need_post = F.softmax(us / 0.05, 1)
            cond = ks + args.cond_weight * z(need_post @ need_sim)
            for name, comb in (("need", z(us)), ("kind", z(ks)), ("need+kind", z(us) + z(ks)),
                               ("need+cond_kind", z(us) + z(cond))):
                r = metrics(comb.cpu().numpy(), labels[s], np.eye(K, dtype=np.float32))
                tk = comb.topk(10, 1)
                pi_ = F.softmax(tk.values, 1)
                isc = (pi_[:, :, None] * p_im[tk.indices]).sum(1)
                for rr, h in enumerate(frames[s].h):
                    ex = [j for j in h if j != targets[s][rr]]
                    if ex:
                        isc[rr, ex] = -1
                fm = rec_metrics(isc, s)
                res[f"{s}/{name}"] = {"R@1": r["recall@1"], "R@10": r["recall@10"], "MRR": r["mrr"],
                                      "fixed_HR@10": fm["HR@10"], "fixed_NDCG@10": fm["NDCG@10"]}
                np.save(out / f"scores_{name}_{s}.npy", comb.cpu().numpy().astype(np.float16))
    print(pd.DataFrame(res).T.to_string())
    (out / "results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    a = parse_args()
    {"prep": stage_prep, "train": stage_train}[a.stage](a)
