"""CPU checks for the motive-utilization readouts and noisy-oracle supports.

Run from the repo root: python tests/test_motive_readout.py
1. gated / soft readouts on a tiny random Qwen2: forward + backward, selection BCE only with labels,
   closed gates reduce the thought to the no-support base (O0 behaviour).
2. noisy-oracle slots on the real Toys labels: per-slot precision ~ p, no duplicates, valid fixed per row.
"""
import json
import os
import sys

import numpy as np
import torch
from transformers import Qwen2Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data.CollabRecDataset import CollabRecDataset  # noqa: E402
from model.CollabRecF2 import CollabRecF2  # noqa: E402


def tiny(readout):
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, pad_token_id=0)
    cfg.support_mode, cfg.readout, cfg.relevance_dim = "text", readout, 8
    model = CollabRecF2(cfg).to(torch.bfloat16)
    model.sel_weight = 1.0
    return model


def batch(bsz=4, k=2, length=12, labels=True):
    ids = torch.randint(1, 60, (bsz, length))
    ids[:, 6] = 64 - 3  # <|Thought|>
    out = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
               support_tokens=torch.randint(1, 60, (bsz, k, 5)), support_logp=torch.randn(bsz, k))
    if labels:
        lab = ids.clone()
        lab[:, :7] = -100
        out.update(labels=lab, support_is_true=torch.randint(0, 2, (bsz, k)).float())
    return out


def test_readouts():
    torch.manual_seed(0)
    for readout in ("gated", "soft"):
        model = tiny(readout)
        res = model(**batch())
        assert torch.isfinite(res.loss), readout
        res.loss.backward()
        if readout == "gated":
            assert res.selection_loss.item() > 0
            assert model.rel_q.weight.grad is not None and model.rel_q.weight.grad.abs().sum() > 0
            assert model.last_gates.shape == (4, 2)
        else:
            assert res.selection_loss.item() == 0
        assert model.injectnet.W_v_2.weight.grad.abs().sum() > 0
        print(f"{readout}: loss {res.loss.item():.4f} selection {res.selection_loss.item():.4f} ok")

    # Closed gates -> thought == no_support_proj(query), i.e. the supports are switched off.
    model = tiny("gated")
    with torch.no_grad():
        model.rel_bias.fill_(-1e4)
    b = batch(labels=False)
    model.eval()
    with torch.no_grad():
        _, _, _, thought, sel = model.generate_embs(b["input_ids"], support_tokens=b["support_tokens"])
        base = model.no_support_proj(model.last_his_query.unsqueeze(1)).squeeze(1)
    assert sel.item() == 0
    assert torch.allclose(thought.float(), base.float(), atol=1e-2), (thought - base).abs().max()
    print("closed gates -> no-support thought: ok")


def test_noisy_slots(dataset="Toys_and_Games", k=2):
    corpus = os.path.join(ROOT, "data", dataset, "collab", "motive_corpus")
    for split in ("train", "valid"):
        truth = json.load(open(os.path.join(corpus, f"row_motives_{split}.json")))
        scores = np.load(os.path.join(corpus, "retriever", "cls", f"scores_{split}.npy")).astype(np.float32)
        for p in (0.25, 0.5, 1.0):
            ds = object.__new__(CollabRecDataset)  # slot logic only: no tokenizer / prompts needed
            ds.motive_source, ds.noisy_precision, ds.seed, ds.support_k = "noisy", p, 0, k
            ds.dataset_type, ds.rng, ds.true_motives = split, np.random.default_rng(0), truth
            ds.pred_order = np.argsort(-scores, axis=1)[:, :k + 2]
            rows = range(0, len(truth), 7)
            kept = n_true = 0
            for r in rows:
                slots = ds.motive_slots(r)
                assert len(slots) == k and len(set(slots.tolist())) == k, slots
                true = set(truth[r])
                kept += sum(s in true for s in slots)
                n_true += min(len(dict.fromkeys(truth[r])), k)
                if split == "valid":
                    assert (slots == ds.motive_slots(r)).all()
            print(f"{split} p={p}: kept true / true slots {kept / n_true:.3f}, "
                  f"slot precision {kept / (k * len(rows)):.3f}")
            assert abs(kept / n_true - p) < 0.03


if __name__ == "__main__":
    test_readouts()
    test_noisy_slots()
    print("all ok")
