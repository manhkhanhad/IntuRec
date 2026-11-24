"""Pilot: is a two-stage (need -> item kind) motive retriever promising on v6 P1000? CPU only.

v6 motives are "need -> item kind". Hypothesis (user, 2026-11-16): the need part is predictable from the history,
the item-kind part is about the next item and confuses a flat 998-way retriever. This script:
  1. embeds every distinct need phrase (extract_<split>.jsonl) and each prototype's need with all-mpnet-base-v2,
     clusters train needs into G need groups (MiniBatchKMeans);
  2. labels each row with the need groups of its extracted motives, each prototype with the group of its need;
  3. reports, on test: need-group predictability (flat cls retriever probabilities summed per group, vs
     popularity), branching (prototypes per group), and the kind stage given the TRUE need group (oracle:
     within-group popularity vs the flat retriever restricted to the group), vs the flat retriever's top-k.
Run: python collab/need_kind_analysis.py [G ...]   (from the repo root)
"""
import json
import os
import sys

import numpy as np

D = "data/Toys_and_Games/collab/motive_corpus_v6"
P = f"{D}/P1000"
CACHE = f"{D}/need_emb_mpnet.npz"


def load_extract(split):
    """Need phrases per CSV row (rows absent from the extract file, e.g. no reasoning, get [])."""
    n = len(json.load(open(f"{P}/row_motives_{split}.json")))
    rows = [[] for _ in range(n)]
    with open(f"{D}/extract_{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows[d["row"]] = [m["need"].strip().rstrip(".").lower() for m in d.get("motives", [])]
    return rows


def main():
    groups_list = [int(g) for g in sys.argv[1:]] or [100, 300]
    ext = {s: load_extract(s) for s in ("train", "valid", "test")}
    protos = json.load(open(f"{P}/prototypes.json", encoding="utf-8"))
    pneed = [p["need"].strip().rstrip(".").lower() for p in protos]
    texts = sorted({n for s in ext.values() for r in s for n in r} | set(pneed))
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        assert list(z["texts"]) == texts
        emb = z["emb"]
    else:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
        emb = st.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
        np.savez(CACHE, texts=np.array(texts, dtype=object), emb=emb)
    idx = {t: i for i, t in enumerate(texts)}
    train_need_ids = sorted({idx[n] for r in ext["train"] for n in r})

    lab = {s: json.load(open(f"{P}/row_motives_{s}.json")) for s in ("train", "test")}
    K = len(protos)
    sc = np.load(f"{P}/retriever/cls/scores_test.npy").astype(np.float32)
    sc -= sc.max(1, keepdims=True)
    prob = np.exp(sc)
    prob /= prob.sum(1, keepdims=True)
    flat = np.argsort(-prob, 1)
    pop_p = np.bincount([m for r in lab["train"] for m in r], minlength=K)
    test = [i for i, r in enumerate(lab["test"]) if r and ext["test"][i]]
    print(f"test rows with labels + extracted needs: {len(test)}")

    def hit(rank_fn, truth, k):
        return np.mean([len(set(rank_fn(i)[:k]) & truth(i)) > 0 for i in test])

    print("flat cls retriever, prototype R@1/3/10: " + " / ".join(
        f"{hit(lambda i: flat[i], lambda i: set(lab['test'][i]), k):.3f}" for k in (1, 3, 10)))

    from sklearn.cluster import MiniBatchKMeans
    for G in groups_list:
        km = MiniBatchKMeans(G, random_state=0, batch_size=4096, n_init=3).fit(emb[train_need_ids])
        grp = km.predict(emb)
        row_g = {s: [{int(grp[idx[n]]) for n in r} for r in ext[s]] for s in ext}
        proto_g = np.array([grp[idx[n]] for n in pneed])
        # consistency: does the prototype's need group match the row's extracted need groups?
        agree = np.mean([bool({int(proto_g[m]) for m in lab["test"][i]} & row_g["test"][i]) for i in test])
        per = np.bincount(proto_g, minlength=G)
        pop_g = np.bincount([g for r in row_g["train"] for g in r], minlength=G)
        pg = np.zeros((len(prob), G), np.float32)
        np.add.at(pg.T, proto_g, prob.T)  # P(group | history) = sum of the flat retriever's prototype probs
        rg = np.argsort(-pg, 1)
        popg = np.argsort(-pop_g)
        print(f"\n== G={G} need groups: prototypes per group min/median/max {per.min()}/{int(np.median(per))}/{per.max()}, "
              f"empty {np.sum(per == 0)}; label-proto group = row need group on {agree:.3f} of test rows")
        for k in (1, 3, 10):
            r = hit(lambda i: rg[i], lambda i: row_g["test"][i], k)
            p = hit(lambda i: popg, lambda i: row_g["test"][i], k)
            print(f"  need group R@{k}: retriever-summed {r:.3f} | popularity {p:.3f} | ratio {r / max(p, 1e-9):.1f}x")
        # kind stage given the TRUE need group(s) of the row's labels (oracle need), prototypes inside them
        def within(i, score):
            gs = {int(proto_g[m]) for m in lab["test"][i]}
            cand = np.where(np.isin(proto_g, list(gs)))[0]
            return cand[np.argsort(-score[cand])]
        print("  oracle need -> kind, prototype R@1/3/10: within-group popularity "
              + " / ".join(f"{hit(lambda i: within(i, pop_p), lambda i: set(lab['test'][i]), k):.3f}" for k in (1, 3, 10))
              + " | retriever inside group "
              + " / ".join(f"{hit(lambda i: within(i, prob[i]), lambda i: set(lab['test'][i]), k):.3f}" for k in (1, 3, 10)))
        # two-stage, deployable: top-1 predicted need group, then the retriever inside it
        def two(i):
            cand = np.where(proto_g == rg[i, 0])[0]
            return cand[np.argsort(-prob[i, cand])]
        print("  two-stage (top-1 group -> retriever inside), prototype R@1/3/10: "
              + " / ".join(f"{hit(two, lambda i: set(lab['test'][i]), k):.3f}" for k in (1, 3, 10)))


if __name__ == "__main__":
    main()
