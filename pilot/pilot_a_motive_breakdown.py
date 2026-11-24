#!/usr/bin/env python3
"""Pilot A: are recommendation failures motive-dependent? (CPU, existing test predictions)

Test rows are grouped by the coarse purchase-motive facets of their target purchase (13 types per dataset, from
the v5 rationale; a row with two facets counts in both groups). Facets are analysis labels only. Per group and
model: HR@10 / NDCG@10, plus the group's mean target popularity and history length (confounders).

Spread test: the across-facet standard deviation of HR@10 vs a permutation null (facet sets shuffled across rows,
same group sizes); p = share of permutations with spread >= observed.

Second cut, motive continuation vs shift: does any facet of the target purchase already occur among the user's
own train-row facets (all of which precede the test rows)? Users with no train rows are "no past".

Writes <scratch>/IntuRec/checkpoints/<dataset>/MotiveRec_pilot/pilot_a.json and prints the tables.
"""
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CKPT, DATA, DATASETS, INTUREC_RUN, O0_RUN, R3_RUN, SHORT, bootstrap_ci,  # noqa: E402
                    hits_from_meta, hits_from_r3_final, load_facets, metrics_from_hits, pilot_dir, r3_checkpoint)

N_PERM = 2000


def model_hits(dataset):
    out = {"O0 (no thought)": hits_from_meta(CKPT / dataset / O0_RUN[dataset] / "result/standard/meta.json"),
           "IntuRec": hits_from_meta(CKPT / dataset / INTUREC_RUN[dataset] / "result/standard/meta.json")}
    for name in R3_RUN:
        out[name] = hits_from_r3_final(r3_checkpoint(dataset, name) / "final_result.json")
    return out


def spread(hr, groups, labels):
    vals = [hr[groups[g]].mean() for g in labels]
    return float(np.std(vals))


def run(dataset):
    test = pd.read_csv(glob.glob(str(DATA / dataset / "test" / f"{dataset}*.csv"))[0])
    train = pd.read_csv(glob.glob(str(DATA / dataset / "train" / f"{dataset}*.csv"))[0])
    facets = load_facets(dataset, "test")
    train_facets = load_facets(dataset, "train")
    assert len(facets) == len(test), (len(facets), len(test))
    hits = model_hits(dataset)
    for name, h in hits.items():
        assert len(h) == len(test), (name, len(h))

    pop = train.item_id.value_counts()
    target_pop = test.item_id.map(pop).fillna(0).to_numpy()
    hist_len = test.history_item_id.map(lambda s: len(eval(s))).to_numpy()

    labels = [f for f, c in Counter(f for fs in facets for f in fs).most_common() if c >= 50]
    groups = {f: np.array([f in fs for fs in facets]) for f in labels}

    # motive continuation vs shift
    past = defaultdict(set)
    for u, fs in zip(train.user_id, train_facets):
        past[u].update(fs)
    cont = np.array([bool(set(fs) & past[u]) if past[u] else False for u, fs in zip(test.user_id, facets)])
    nopast = np.array([not past[u] for u in test.user_id])
    shift = ~cont & ~nopast & np.array([bool(fs) for fs in facets])

    rng = np.random.default_rng(42)
    res = {"n_test": len(test), "facets": {}, "spread": {}, "continuation": {}}
    for f in labels:
        m = groups[f]
        res["facets"][f] = {"n": int(m.sum()), "share": float(m.mean()), "target_pop": float(np.median(target_pop[m])),
                            "hist_len": float(hist_len[m].mean())}
    for name, h in hits.items():
        hr, ndcg = metrics_from_hits(h)
        res.setdefault("overall", {})[name] = {"hr10": float(hr.mean()), "ndcg10": float(ndcg.mean())}
        for f in labels:
            m = groups[f]
            res["facets"][f][name] = {"hr10": float(hr[m].mean()), "ndcg10": float(ndcg[m].mean())}
        obs = spread(hr, groups, labels)
        null = []
        for _ in range(N_PERM):
            perm = rng.permutation(len(facets))
            null.append(spread(hr, {f: groups[f][perm] for f in labels}, labels))
        null = np.array(null)
        res["spread"][name] = {"std_hr10": obs, "null_mean": float(null.mean()), "null_95": float(np.percentile(null, 95)),
                               "p": float((null >= obs).mean()),
                               "max_over_min": float(max(res["facets"][f][name]["hr10"] for f in labels)
                                                     / max(min(res["facets"][f][name]["hr10"] for f in labels), 1e-9))}
        c = {}
        for key, m in (("continuation", cont), ("shift", shift), ("no_past", nopast)):
            c[key] = {"n": int(m.sum()), "hr10": float(hr[m].mean()), "ndcg10": float(ndcg[m].mean()),
                      "target_pop": float(np.median(target_pop[m])), "hist_len": float(hist_len[m].mean())}
        diff = hr[cont].mean() - hr[shift].mean()
        # bootstrap the difference of group means
        b = []
        ic, is_ = np.where(cont)[0], np.where(shift)[0]
        for _ in range(N_PERM):
            b.append(hr[rng.choice(ic, len(ic))].mean() - hr[rng.choice(is_, len(is_))].mean())
        c["cont_minus_shift"] = {"hr10": float(diff), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
        res["continuation"][name] = c

    # Controlled test: do facets still explain hits once target-history lexical similarity, target popularity and
    # history length are accounted for? Logistic regression of hit@10 on controls vs controls + facet indicators,
    # likelihood-ratio test (df = number of facets).
    from scipy.stats import chi2
    from sklearn.linear_model import LogisticRegression

    def toks(t):
        return set(str(t).lower().replace(",", " ").replace("(", " ").replace(")", " ").split())

    sim = np.array([max((len(toks(t) & toks(h)) / max(len(toks(t) | toks(h)), 1) for h in eval(hs)), default=0.0)
                    for t, hs in zip(test.item_title, test.history_item_title)])
    controls = np.column_stack([sim, np.log1p(target_pop), np.log1p(hist_len)])
    fac = np.column_stack([groups[f] for f in labels]).astype(float)
    res["controlled"] = {"title_sim_mean": float(sim.mean())}
    for name, h in hits.items():
        y = (np.asarray(h) < 10).astype(int)

        def ll(x):
            x = (x - x.mean(0)) / (x.std(0) + 1e-9)
            m = LogisticRegression(C=1e4, max_iter=5000).fit(x, y)
            p = np.clip(m.predict_proba(x)[:, 1], 1e-12, 1 - 1e-12)
            return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())

        l0, l1 = ll(controls), ll(np.column_stack([controls, fac]))
        stat = 2 * (l1 - l0)
        res["controlled"][name] = {"lr_stat": stat, "df": len(labels), "p": float(chi2.sf(stat, len(labels))),
                                   "mcfadden_gain": float((l1 - l0) / -l0)}

    out = pilot_dir(dataset) / "pilot_a.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)

    names = list(hits)
    print(f"\n=== Pilot A — {SHORT[dataset]} (n={len(test)}) — HR@10 by target-purchase motive facet ===")
    print(f"{'facet':28s} {'n':>5s} {'pop':>5s} {'hlen':>5s} " + " ".join(f"{n[:14]:>14s}" for n in names))
    for f in labels:
        r = res["facets"][f]
        print(f"{f:28s} {r['n']:5d} {r['target_pop']:5.0f} {r['hist_len']:5.1f} "
              + " ".join(f"{r[n]['hr10']:14.4f}" for n in names))
    print(f"{'overall':28s} {len(test):5d} {'':5s} {'':5s} " + " ".join(f"{res['overall'][n]['hr10']:14.4f}" for n in names))
    for n in names:
        s = res["spread"][n]
        print(f"spread {n:16s} std {s['std_hr10']:.4f} (null mean {s['null_mean']:.4f}, 95% {s['null_95']:.4f}, "
              f"p {s['p']:.4f}), max/min {s['max_over_min']:.2f}")
    print("controlled (hit@10 ~ title-sim + log pop + log hist-len [+ facets]), likelihood-ratio test:")
    for n in names:
        c = res["controlled"][n]
        print(f"  {n:16s} LR {c['lr_stat']:.1f} (df {c['df']}), p {c['p']:.2e}, McFadden gain {c['mcfadden_gain']:.4f}")
    print("motive continuation vs shift (HR@10):")
    for n in names:
        c = res["continuation"][n]
        print(f"  {n:16s} cont {c['continuation']['hr10']:.4f} (n {c['continuation']['n']}, pop {c['continuation']['target_pop']:.0f})"
              f"  shift {c['shift']['hr10']:.4f} (n {c['shift']['n']}, pop {c['shift']['target_pop']:.0f})"
              f"  no-past {c['no_past']['hr10']:.4f} (n {c['no_past']['n']})"
              f"  diff {c['cont_minus_shift']['hr10']:+.4f} [{c['cont_minus_shift']['ci'][0]:+.4f}, {c['cont_minus_shift']['ci'][1]:+.4f}]")


if __name__ == "__main__":
    for d in (sys.argv[1:] or DATASETS):
        run(d)
