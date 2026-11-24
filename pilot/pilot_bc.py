#!/usr/bin/env python3
"""Pilot B (does the latent thought encode the purchase motive?) and Pilot C, CPU part (collapse geometry +
intervention table). Run after the thoughts / decodes exist.

  python pilot/pilot_bc.py probe     # Pilot B
  python pilot/pilot_bc.py collapse  # Pilot C (uses pilot_c_r3.py decodes when present)

Pilot B: linear probes from a test-row representation to the 13 motive facets of the row's purchase
(multi-label, one logistic output per facet), 5-fold cross-validation grouped by user, features standardised on
the training folds; metric = macro ROC-AUC over facets (0.5 = no information). Representations: each model's
<|Thought|> vector, the gte-large embedding of the history titles (what the history alone says), history +
thought (does the thought add anything?), and Gaussian noise (floor).

Pilot C: collapse statistics of the test thoughts (same as collab/inturec_latent_diagnostics.py) and paired
ΔHR@10 / ΔNDCG@10 (bootstrap 95% CI) of thought interventions vs the raw model.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CKPT, DATA, DATASETS, INTUREC_RUN, R3_RUN, SHORT, bootstrap_ci, collapse_stats,  # noqa: E402
                    hits_from_meta, hits_from_r3_generation, load_facets, metrics_from_hits, pilot_dir,
                    r3_checkpoint, r3_generation_path)


def load_thoughts(dataset):
    out = {"IntuRec": np.load(CKPT / dataset / INTUREC_RUN[dataset] / "diagnostics/thoughts_test.npy").astype(np.float32)}
    for name in R3_RUN:
        new = pilot_dir(dataset) / "r3" / R3_RUN[name] / "evidence" / dataset / "pilot" / "test" / "thoughts.pt"
        old = r3_checkpoint(dataset, name) / "analysis" / "evidence" / dataset / "test" / "thoughts.pt"
        path = new if new.exists() else old
        out[name] = torch.load(path, map_location="cpu", weights_only=True).float().numpy()
    return out


# ---------------------------------------------------------------- Pilot B
def fit_probe(xtr, ytr, xte, wd, steps=300):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = torch.tensor((xtr - mu) / sd), torch.tensor((xte - mu) / sd)
    ytr = torch.tensor(ytr)
    lin = torch.nn.Linear(xtr.shape[1], ytr.shape[1])
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(lin(xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(lin(xte)).numpy()


def probe_auc(x, y, groups, wd):
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(x, y, groups):
        pred[te] = fit_probe(x[tr], y[tr], x[te], wd)
    aucs = [roc_auc_score(y[:, j], pred[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(aucs)), pred


def probe(dataset):
    import glob
    import pandas as pd
    torch.manual_seed(0)
    test = pd.read_csv(glob.glob(str(DATA / dataset / "test" / f"{dataset}*.csv"))[0])
    facets = load_facets(dataset, "test")
    labels = sorted({f for fs in facets for f in fs})
    y = np.array([[f in fs for f in labels] for fs in facets], dtype=np.float32)
    users = pd.factorize(test.user_id)[0]
    hist = np.load(DATA / dataset / "collab" / "hyp_cache" / "hist_gte-large-en-v1.5_test.npy").astype(np.float32)
    thoughts = load_thoughts(dataset)
    rng = np.random.default_rng(0)
    feats = {"noise (floor)": rng.standard_normal((len(y), 1536)).astype(np.float32), "history text (gte)": hist}
    for name, t in thoughts.items():
        feats[f"{name} thought"] = t
        feats[f"history + {name} thought"] = np.concatenate([hist, t], 1)
    res = {"labels": labels, "n": len(y), "auc": {}}
    for name, x in feats.items():
        best = max((probe_auc(x, y, users, wd)[0], wd) for wd in (1e-4, 1e-3, 1e-2))
        res["auc"][name] = {"macro_auc": best[0], "weight_decay": best[1]}
        print(f"  {SHORT[dataset]} {name:34s} macro AUC {best[0]:.4f} (wd {best[1]:g})", flush=True)
    json.dump(res, open(pilot_dir(dataset) / "pilot_b.json", "w"), indent=2)


# ---------------------------------------------------------------- Pilot C
def collapse(dataset):
    res = {"geometry": {}, "interventions": {}}
    for name, t in load_thoughts(dataset).items():
        res["geometry"][name] = collapse_stats(t)
    diag = CKPT / dataset / INTUREC_RUN[dataset] / "diagnostics"
    conds = {"IntuRec": {c: diag / f"meta_{c}.json" for c in
                         ("raw", "mask", "mean", "shuffle", "oracle_intuitions", "remove_target")}}
    for name in R3_RUN:
        conds[name] = {c: r3_generation_path(dataset, name, c) for c in
                       ("raw", "mask", "mean", "shuffle_user", "shuffle_target")}
    for name, paths in conds.items():
        if not paths["raw"].exists():
            print(f"  {SHORT[dataset]} {name}: raw decode missing, skipped", flush=True)
            continue
        load = hits_from_meta if name == "IntuRec" else hits_from_r3_generation
        hr0, nd0 = metrics_from_hits(load(paths["raw"]))
        out = {"raw": {"hr10": float(hr0.mean()), "ndcg10": float(nd0.mean())}}
        for c, p in paths.items():
            if c == "raw" or not p.exists():
                continue
            hr, nd = metrics_from_hits(load(p))
            out[c] = {"hr10": float(hr.mean()), "ndcg10": float(nd.mean()),
                      "d_hr10": float((hr - hr0).mean()), "d_hr10_ci": bootstrap_ci(hr - hr0),
                      "d_ndcg10": float((nd - nd0).mean()), "d_ndcg10_ci": bootstrap_ci(nd - nd0)}
        res["interventions"][name] = out
    json.dump(res, open(pilot_dir(dataset) / "pilot_c.json", "w"), indent=2)
    print(f"\n=== Pilot C — {SHORT[dataset]} geometry of test thoughts ===")
    for name, g in res["geometry"].items():
        print(f"  {name:14s} eff. rank {g['effective_rank']:.2f}  PR {g['participation_ratio']:.2f}  "
              f"PC1 {g['pc1_variance']:.3f}  PCs90% {g['pcs_for_90pct']}  pair cos {g['random_pair_cos']:.4f}  "
              f"cos-to-mean {g['cos_to_mean']:.4f}")
    print(f"=== Pilot C — {SHORT[dataset]} interventions (HR@10, Δ vs raw [95% CI]) ===")
    for name, out in res["interventions"].items():
        line = f"  {name:14s} raw {out['raw']['hr10']:.4f}"
        for c, v in out.items():
            if c != "raw":
                line += f" | {c} {v['d_hr10']:+.4f} [{v['d_hr10_ci'][0]:+.4f}, {v['d_hr10_ci'][1]:+.4f}]"
        print(line)


if __name__ == "__main__":
    cmd = sys.argv[1]
    for d in (sys.argv[2:] or DATASETS):
        {"probe": probe, "collapse": collapse}[cmd](d)
