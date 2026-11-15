#!/usr/bin/env python3
"""Does the decoder use each thought? Probe for LatentMS2 checkpoints (model/LatentMS2.py), CPU or GPU.

On --limit validation rows (teacher forcing) reports the mean per-row NLL of the target title when a thought
is left as is, masked out of the attention mask, or replaced by another user's thought; the norm and
cross-user cosine of every thought; and the RATTNorm attention (entropy, top-1 mass, learnt temperature and
gain). A thought the decoder uses must raise the NLL when masked or swapped.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.IntuRecF2Dataset import IntuRecF2Dataset  # noqa: E402
from model.LatentMS2 import LatentMS2  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--ckpt_dir", required=True)
p.add_argument("--dataset", default="Toys_and_Games")
p.add_argument("--reason_step", type=int, default=2)
p.add_argument("--limit", type=int, default=48)
p.add_argument("--batch_size", type=int, default=8)
p.add_argument("--device", default="cpu")
p.add_argument("--threads", type=int, default=24)
args = p.parse_args()
torch.set_num_threads(args.threads)
torch.manual_seed(0)
S, dev = args.reason_step, torch.device(args.device)
cfg = AutoConfig.from_pretrained(args.ckpt_dir)
tok = AutoTokenizer.from_pretrained(args.ckpt_dir)
dtype = torch.float32 if dev.type == "cpu" else torch.bfloat16
model = LatentMS2.from_pretrained(args.ckpt_dir, config=cfg, torch_dtype=dtype).to(dev).eval()
model.reason_step = S
ds = IntuRecF2Dataset("./data", args.dataset, tok, sample=args.limit, dataset_type="valid", intuition_num=15, seed=0,
                      reason_step=S)


def nll(emb, mask, labels):
    h = model.model(inputs_embeds=emb, attention_mask=mask).last_hidden_state
    return model._row_nll(h, labels)[0].float().cpu(), h


res, TH, ent, top = {}, [], [], []
with torch.no_grad():
    for s in range(0, len(ds), args.batch_size):
        it = [ds[i] for i in range(s, min(s + args.batch_size, len(ds)))]
        g = lambda k: torch.tensor(np.stack([b[k] for b in it])).to(dev)  # noqa: E731
        ids, mask, labels = g("input_ids"), g("attention_mask"), g("labels")
        emb, first = model.generate_embs(ids)
        rows = torch.arange(ids.shape[0], device=dev)
        roll = torch.roll(rows, 1)
        th = torch.stack([emb[rows, first + t] for t in range(S)], 1)  # [b, S, dim]
        TH.append(th.float().cpu())

        def var(name, e=None, m=None):
            v, h = nll(emb if e is None else e, mask if m is None else m, labels)
            res.setdefault(name, []).append(v)
            return h

        h = var("normal")
        for t in range(S):
            m = mask.clone(); m[rows, first + t] = 0; var(f"thought {t + 1} masked", m=m)
            e = emb.clone(); e[rows, first + t] = th[roll, t]; var(f"thought {t + 1} from another user", e=e)
        m = mask.clone(); e = emb.clone()
        for t in range(S):
            m[rows, first + t] = 0; e[rows, first + t] = th[roll, t]
        var("all thoughts masked", m=m); var("all thoughts from another user", e=e)
        for t in range(1, S):  # attention that built thought t+1 (causal: h at earlier positions is what RATT saw)
            a, _ = model.ratt.attention(h, mask, first + t)
            ent.append(-(a * (a + 1e-12).log()).sum(1).cpu()); top.append(a.max(1).values.cpu())
TH = torch.cat(TH)
n = len(TH)
perm = torch.roll(torch.arange(n), 3)
cs = lambda a, b: float(F.cosine_similarity(a, b, dim=-1).mean())  # noqa: E731
out = {"ckpt_dir": args.ckpt_dir, "n_rows": n,
       "nll": {k: round(float(torch.cat(v).mean()), 4) for k, v in res.items()},
       "thought_norm": [float(TH[:, t].norm(dim=-1).mean()) for t in range(S)],
       "cross_user_cos": [cs(TH[:, t], TH[perm, t]) for t in range(S)],
       "user_specific_share": [float(((TH[:, t] - TH[:, t].mean(0)) ** 2).sum() / (TH[:, t] ** 2).sum()) for t in range(S)],
       "cos_thought1_vs_later_same_user": [cs(TH[:, 0], TH[:, t]) for t in range(1, S)],
       "ratt": {"entropy": float(torch.cat(ent).mean()), "top1_mass": float(torch.cat(top).mean()),
                "temperature": float(model.ratt.log_temp.float().exp()), "gain": float(model.ratt.log_gain.float().exp()),
                "residual": bool(model.ratt.residual)}}
print(json.dumps(out, indent=1))
