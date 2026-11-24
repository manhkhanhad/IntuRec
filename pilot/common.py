"""Shared paths and helpers for the MotiveRec pilot studies (A: motive breakdown, B: motive probe, C: collapse).

Every model here is evaluated on the same test split (R3 and IntuRec use byte-identical CSVs), so per-row
results line up by test row index. Motive facets (13 coarse purchase-motive types per dataset) come from the
v5 rationales; they are used only as analysis labels, never as model inputs.
"""
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
CKPT = Path("/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints")
R3_ORI = Path("/common/home/users/m/mk.ngohuu.2024/research/R3_ori/output_dir")
DATASETS = ("CDs_and_Vinyl", "Toys_and_Games")
SHORT = {"CDs_and_Vinyl": "CDs", "Toys_and_Games": "Toys"}
PILOT_DIR = "MotiveRec_pilot"  # <CKPT>/<dataset>/MotiveRec_pilot/...

# Trained checkpoints (all Qwen2.5-1.5B, micro-batch 2 for IntuRec / O0).
INTUREC_RUN = {"CDs_and_Vinyl": "IntuRec_original/IntuRecF2_micro2_job349506",
               "Toys_and_Games": "IntuRec_original/IntuRecF2_micro2_job348597"}
O0_RUN = {"CDs_and_Vinyl": "CollabRec_oracle/text-micro2_k0_beta0_rs1_lr0.0003_job351003",
          "Toys_and_Games": "CollabRec_oracle/text-micro2_k0_beta0_rs1_lr0.0003_job350999"}
R3_RUN = {"LatentR3-SFT": "latent", "LatentR3-RL": "nomul_attn_lr_5e-5"}


def r3_checkpoint(dataset, model):
    return R3_ORI / dataset / R3_RUN[model]


def pilot_dir(dataset):
    return CKPT / dataset / PILOT_DIR


def clean(title):
    return str(title).strip(" \n").strip('"').strip(" ")


def hit_index(pred, target):
    """0-based position of the target in the top-k (last match, as utils.util.calRecallAndNdcgFromFile); inf if absent."""
    hit = math.inf
    for i, p in enumerate(pred):
        if clean(p) == target:
            hit = i
    return hit


def metrics_from_hits(hits, k=10):
    hits = np.asarray(hits, dtype=float)
    hr = (hits < k).astype(float)
    ndcg = np.where(hits < k, 1.0 / np.log2(np.minimum(hits, 1e9) + 2), 0.0)
    return hr, ndcg


def hits_from_meta(path):
    """IntuRec / CollabRec meta.json: list of rows with 'predict' and 'tgt_item'."""
    rows = json.load(open(path))
    return np.array([hit_index(r["predict"], clean(r["tgt_item"])) for r in rows], dtype=float)


def hits_from_r3_final(path):
    """R3 final_result.json: list of rows with 'predict' and 'output'."""
    rows = json.load(open(path))
    return np.array([hit_index(r["predict"], clean(r["output"])) for r in rows], dtype=float)


def hits_from_r3_generation(path):
    """R3 experiment.py generate output: records with 1-based 'rank' (1e9 = miss), in test-row order."""
    recs = sorted(json.load(open(path))["records"], key=lambda r: int(r["sample_id"]))
    return np.array([r["rank"] - 1 if r["rank"] < 10 ** 8 else math.inf for r in recs], dtype=float)


def r3_generation_path(dataset, model, mode):
    return (pilot_dir(dataset) / "r3" / R3_RUN[model] / "results" / "intervention" / "pilot"
            / f"generation_test_all_k32_s42_{mode}_t4_temp0.1_a0.5.json")


def load_facets(dataset, split="test"):
    """Per row: list of coarse motive facets (excluding 'other'), in row order."""
    out = {}
    with open(DATA / dataset / "collab" / f"rationales_facet_{split}_v5.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "error" not in rec:
                out[rec["row"]] = [x for x in rec.get("facets", []) if x != "other"]
    return [out.get(i, []) for i in range(max(out) + 1)]


def bootstrap_ci(delta, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    delta = np.asarray(delta, dtype=float)
    means = [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def collapse_stats(x, seed=42):
    """Same statistics as collab/inturec_latent_diagnostics.py."""
    rng = np.random.default_rng(seed)
    x = x.astype(np.float64)
    n = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    pairs = rng.integers(0, len(x), (200000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    mean = x.mean(0)
    ev = np.linalg.svd(x - mean, compute_uv=False) ** 2
    f = ev / ev.sum()
    f = f[f > 0]
    cum = np.cumsum(f)
    return {"effective_rank": float(np.exp(-(f * np.log(f)).sum())), "participation_ratio": float(1 / (f ** 2).sum()),
            "pc1_variance": float(f[0]), "pcs_for_90pct": int(np.searchsorted(cum, 0.9)) + 1,
            "random_pair_cos": float((n[pairs[:, 0]] * n[pairs[:, 1]]).sum(1).mean()),
            "cos_to_mean": float((n @ (mean / np.linalg.norm(mean))).mean())}
