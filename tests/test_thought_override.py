"""CPU check of CollabRecF2's thought_override (used by collab/latent_diagnostics.py), tiny random Qwen2.

Run from the repo root: python tests/test_thought_override.py
"""
import os
import sys

import torch
from transformers import GenerationConfig, Qwen2Config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.CollabRecF2 import CollabRecF2  # noqa: E402


def main():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, pad_token_id=0, eos_token_id=63)
    cfg.support_mode = "text"
    model = CollabRecF2(cfg).to(torch.bfloat16).eval()
    model.beta, model.gamma = 0.0, 0.0

    ids = torch.randint(1, 60, (3, 10))
    ids[:, 9] = 64 - 3  # <|Thought|> as the last prompt token, as in the test prompts
    mask = torch.ones_like(ids)
    supports = torch.randint(1, 60, (3, 4, 5))
    gen = GenerationConfig(num_beams=2, num_return_sequences=2, max_new_tokens=4, pad_token_id=0,
                           eos_token_id=63, do_sample=False)

    with torch.no_grad():
        _, _, _, thought, _ = model.generate_embs(ids, support_tokens=supports)
        assert thought.shape == (3, 32)
        assert model.last_his_query.shape == (3, 32)
        raw = model.generate(input_ids=ids, attention_mask=mask, generation_config=gen,
                             support_tokens=supports).tolist()
        same = model.generate(input_ids=ids, attention_mask=mask, generation_config=gen,
                              support_tokens=supports, thought_override=thought).tolist()
        assert raw == same, "overriding with the model's own thought must reproduce the plain decode"
        # The override replaces the thought: different supports + the original thought == original decode.
        other = torch.randint(1, 60, (3, 4, 5))
        pinned = model.generate(input_ids=ids, attention_mask=mask, generation_config=gen,
                                support_tokens=other, thought_override=thought).tolist()
        assert pinned == raw, "with thought_override the supports must not matter"
        zero = model.generate(input_ids=ids, attention_mask=mask, generation_config=gen,
                              support_tokens=supports, thought_override=torch.zeros_like(thought))
        assert zero.shape[0] == 3 * 2
    print("ok: thought_override reproduces the raw decode, overrides the supports, and survives beam expansion")


if __name__ == "__main__":
    main()
