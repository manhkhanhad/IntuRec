"""CPU check of the collaborative InfoNCE queue on the refined thought (tiny random Qwen2).

Run from the repo root: python tests/test_collab_queue.py
"""
import os
import sys

import torch
from transformers import Qwen2Config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.CollabRecF2 import CollabRecF2  # noqa: E402


def tiny_model(bank_size=20):
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, pad_token_id=0)
    cfg.support_mode = "text"
    cfg.collab_queue = {"relations": ["user", "target", "motive"], "bank_size": bank_size, "projection_dim": 16,
                        "temperature": 0.1, "min_token_accuracy": 0.0, "max_positives": 16, "num_negatives": 256}
    # bf16 as in training: generate_embs writes the thought into inputs_embeds in place after the
    # first pass; in fp32 the first RMSNorm's .float() is a no-op that saves that same tensor.
    model = CollabRecF2(cfg).to(torch.bfloat16)
    model.collab_weights = {r: 0.1 for r in cfg.collab_queue["relations"]}
    return model


def batch(step, bsz=8, length=12):
    ids = torch.randint(1, 60, (bsz, length))
    ids[:, 6] = 64 - 3  # <|Thought|> id flag
    labels = ids.clone()
    labels[:, :7] = -100
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                support_tokens=torch.randint(1, 60, (bsz, 2, 5)),
                cq_user=torch.randint(0, 3, (bsz,)), cq_target=torch.randint(0, 4, (bsz,)),
                cq_sample=torch.arange(bsz) + step * bsz, cq_motives=torch.randint(-1, 5, (bsz, 2)))


def main():
    torch.manual_seed(0)
    model = tiny_model()
    model.train()
    for step in range(4):
        out = model(**batch(step))
        out.loss.backward()
        print(step, f"ntp {out.ntp_loss.item():.3f}", {k: round(v.item(), 3) for k, v in out.collab_losses.items()},
              {k: round(v, 2) for k, v in out.collab_rates.items()})
        assert set(out.collab_losses) == {"user", "target", "motive"}
    assert int(model.bank_count) == 20 and int(model.bank_pointer) == 32 % 20, "FIFO wrap-around"
    head_grad = model.collab_heads["motive"][1].weight.grad
    assert head_grad is not None and head_grad.abs().sum() > 0, "motive head gets no gradient"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.injectnet.parameters()), \
        "InfoNCE gradient does not reach the refined thought (InjectNet)"

    # Motive positives: a shared id counts, the -1 padding never matches.
    mo = torch.tensor([[2, -1]])
    cm = torch.tensor([[5, 2], [-1, -1], [3, 4]])
    match = ((mo[:, :, None, None] == cm[None, None]) & (mo[:, :, None, None] >= 0)).any(3).any(1)
    assert match.tolist() == [[True, False, False]], match

    # Eval mode: no collaborative loss and no queue push, so valid NTP stays comparable.
    model.eval()
    count = int(model.bank_count)
    with torch.no_grad():
        out = model(**batch(9))
    assert out.collab_losses == {} and torch.allclose(out.loss, out.ntp_loss) and int(model.bank_count) == count
    print("all checks passed")


if __name__ == "__main__":
    main()
