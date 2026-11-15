#!/usr/bin/env python3
"""Per-step geometry of multi-step IntuRec thoughts (model/IntuRecMS.py; also IntuRecF2 rs=1 checkpoints).

For every test row, runs generate_embs once (no decoding) and records the vector written into each of the S
<|Thought|> slots, plus the history query h_u (last-layer hidden state at thought 1, the InjectNet query).
Reports:
  per_step[t]   collapse statistics of thought t across users (same definitions as
                collab/inturec_latent_diagnostics.py: effective rank, participation ratio, PC1 variance share,
                PCs for 90%, random-pair cosine, cosine to the mean) and its mean L2 norm
  history_query the same statistics for h_u (what the backbone offers before any thought)
  cross_step    S x S matrices over rows: mean within-row cosine cos(thought_i, thought_j) of the same user,
                and the cross-user baseline (thought_i of a user vs thought_j of a random other user); within ~
                cross means step j carries no user-specific change from step i
  step_mode / reason_step as read from the checkpoint
Writes <out_dir>/geometry_summary.json and thoughts_steps_test.npz (float16, [n, S, dim] + h_u).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.IntuRecF2Dataset import IntuRecF2Dataset  # noqa: E402
from model.IntuRecMS import IntuRecMS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--dataset", default="Toys_and_Games")
    p.add_argument("--data_path", default="./data")
    p.add_argument("--reason_step", type=int, required=True, help="S the checkpoint was trained with.")
    p.add_argument("--intuition_num", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--limit", type=int, default=-1, help="Random test rows (smoke tests); -1 = all.")
    p.add_argument("--out_dir", default="", help="Default: <ckpt_dir>/diagnostics")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def collapse_stats(x, rng):
    """Same statistics as collab/inturec_latent_diagnostics.py."""
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
            "cos_to_mean": float((n @ (mean / np.linalg.norm(mean))).mean()),
            "mean_norm": float(np.linalg.norm(x, axis=1).mean())}


@torch.no_grad()
def collect(model, ds, batch_size, device, S):
    pad = model.config.pad_token_id
    thought_id = model.model.embed_tokens.num_embeddings - 3
    thoughts, queries = [], []
    for start in tqdm(range(0, len(ds), batch_size), desc="thoughts"):
        items = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
        ids = torch.tensor(np.stack([b["input_ids"] for b in items])).to(device)
        width = max(b["intuition_tokens"].shape[1] for b in items)
        intu = torch.tensor(np.stack([np.pad(b["intuition_tokens"], ((0, 0), (width - b["intuition_tokens"].shape[1], 0)),
                                             constant_values=pad) for b in items])).to(device)
        embeds, _ = model.generate_embs(ids, intu, None, None, None)
        idxs = torch.arange(ids.shape[1], device=device).unsqueeze(0).expand(ids.shape[0], -1)
        first = torch.where(ids == thought_id, idxs, ids.new_full(idxs.shape, ids.shape[1])).min(1).values
        assert bool((ids.gather(1, (first.unsqueeze(1) + torch.arange(S, device=device))) == thought_id).all())
        rows = torch.arange(ids.shape[0], device=device)
        thoughts.append(torch.stack([embeds[rows, first + s] for s in range(S)], 1).float().cpu())
        # history query: the hidden state at thought 1 before injection (first pass of generate_embs)
        h = model.model(input_ids=ids, attention_mask=(ids != pad).int()).last_hidden_state
        queries.append(h[rows, first].float().cpu())
    return torch.cat(thoughts).numpy(), torch.cat(queries).numpy()


def cross_step(th, rng):
    n, S, _ = th.shape
    z = th / (np.linalg.norm(th, axis=-1, keepdims=True) + 1e-12)
    perm = rng.permutation(n)
    perm = np.where(perm == np.arange(n), (perm + 1) % n, perm)
    within = np.einsum("nid,njd->ij", z, z) / n
    cross = np.einsum("nid,njd->ij", z, z[perm]) / n
    return within.round(4).tolist(), cross.round(4).tolist()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir or os.path.join(args.ckpt_dir, "diagnostics"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    config = AutoConfig.from_pretrained(args.ckpt_dir, trust_remote_code=True)
    config.step_mode = getattr(config, "step_mode", "copy")  # IntuRecF2 checkpoints: rs=1, mode irrelevant
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir, trust_remote_code=True)
    model = IntuRecMS.from_pretrained(args.ckpt_dir, config=config, torch_dtype=torch.bfloat16).to(device).eval()
    model.reason_step = args.reason_step
    ds = IntuRecF2Dataset(args.data_path, args.dataset, tokenizer, sample=args.limit, dataset_type="test",
                          intuition_num=args.intuition_num, seed=args.seed, reason_step=args.reason_step)
    th, hq = collect(model, ds, args.batch_size, device, args.reason_step)
    np.savez_compressed(out_dir / "thoughts_steps_test.npz", thoughts=th.astype(np.float16), history_query=hq.astype(np.float16))

    within, cross = cross_step(th, rng)
    summary = {"ckpt_dir": args.ckpt_dir, "dataset": args.dataset, "n_rows": int(th.shape[0]),
               "step_mode": config.step_mode, "reason_step": args.reason_step,
               "history_query": collapse_stats(hq, rng),
               "per_step": [collapse_stats(th[:, s], rng) for s in range(args.reason_step)],
               "cross_step_within_row_cos": within, "cross_step_cross_user_cos": cross}
    (out_dir / "geometry_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
