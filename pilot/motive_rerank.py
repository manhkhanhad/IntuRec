import sys, json, glob, numpy as np, pandas as pd
sys.path.insert(0, "pilot")
from common import *
T = {"CDs_and_Vinyl": "CollabRec/text-motive_k10_beta0.1_rs1_lr0.0003_job359280",
     "Toys_and_Games": "CollabRec/text-motive_k10_beta0.1_rs1_lr0.0003_job345291"}
for d in T:
    train = pd.read_csv(glob.glob(str(DATA/d/"train"/f"{d}*.csv"))[0])
    titles = {}
    for line in open(glob.glob(str(DATA/d/"info"/"*.txt"))[0], encoding="utf-8"):
        t, i = line.rstrip("\n").rsplit("\t", 1); titles[t.strip()] = int(i)
    mot = json.load(open(DATA/d/"collab/motive_corpus/row_motives_train.json"))
    K = len(json.load(open(DATA/d/"collab/motive_corpus/prototypes.json")))
    prior = np.full(K, 1.0)
    cnt = {}
    for it, ms in zip(train.item_id, mot):
        v = cnt.setdefault(it, np.zeros(K))
        for m in ms: v[m] += 1; prior[m] += 1
    prior /= prior.sum()
    a = 1.0  # Dirichlet smoothing strength toward the prior
    def p_item(it):
        v = cnt.get(it, np.zeros(K)); return (v + a * K * prior) / (v.sum() + a * K)
    s = np.load(DATA/d/"collab/motive_corpus/retriever/cls/scores_test.npy").astype(np.float64)
    ph = np.exp(s - s.max(1, keepdims=True)); ph /= ph.sum(1, keepdims=True)
    for name, path in (("IntuRec", CKPT/d/INTUREC_RUN[d]/"result/standard/meta.json"), ("MotiveRec-T", CKPT/d/T[d]/"result/standard/meta.json")):
        rows = json.load(open(path)); miss = 0
        comp = []
        for r, row in enumerate(rows):
            c = []
            for p in row["predict"]:
                it = titles.get(clean(p))
                if it is None: miss += 1; c.append(np.log((ph[r] * prior).sum()))
                else: c.append(np.log((ph[r] * p_item(it)).sum() / (ph[r] * prior).sum() * (ph[r]*prior).sum()))
            comp.append(c)
        for lam in (0, 0.25, 0.5, 1.0, 2.0):
            hits = []
            for row, c in zip(rows, comp):
                sc = np.array(row["predict_score"][:len(row["predict"])]) + lam * np.array(c)
                order = np.argsort(-sc, kind="stable")
                hits.append(hit_index([row["predict"][i] for i in order], clean(row["tgt_item"])))
            h = np.array(hits, float)
            print(f"{d[:4]} {name:11s} λ{lam:<4} " + " ".join(f"HR@{k} {(h<k).mean():.4f}" for k in (1, 5)) +
                  " " + " ".join(f"N@{k} {np.where(h<k,1/np.log2(np.minimum(h,1e9)+2),0).mean():.4f}" for k in (5, 10)) + (f"  (unmatched titles {miss})" if lam == 0 else ""))
