"""CPU checks of run_collabrec_grpo.py on a tiny random Qwen2 CollabRecF2, including a line-by-line
re-run of R3's NoiseGRPORecTrainer rollout / loss (research/R3/src/grpo_attention_tuning/grpo_trainer.py,
trl 0.16.1) on the same batch.

Run from the repo root: python tests/test_grpo.py
"""
import copy
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from transformers import Qwen2Config
from trl.trainer.utils import selective_log_softmax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.CollabRecF2 import CollabRecF2  # noqa: E402
import run_collabrec_grpo as G  # noqa: E402

V, D, PAD, THOUGHT = 64, 32, 0, 64 - 3


def tiny_model(dtype=torch.float64):
    cfg = Qwen2Config(vocab_size=V, hidden_size=D, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, pad_token_id=PAD, eos_token_id=63,
                      tie_word_embeddings=True)
    cfg.support_mode = "text"
    # float64 for exact gradient checks: in float32 Qwen2RMSNorm.to(float32) aliases the embeddings that
    # generate_embs then writes in place (autograd version error); real runs are bf16, where it copies.
    model = CollabRecF2(cfg).to(dtype)
    model.beta = model.gamma = model.sel_weight = 0.0
    return model


def make_row(rng, n_pad=3, n_prompt=8, n_target=4):
    prompt = list(rng.integers(1, 60, n_prompt)) + [THOUGHT, 62]  # ... <|Thought|> <|StartPredict|>
    target = list(rng.integers(1, 60, n_target))
    ids = [PAD] * n_pad + prompt + target
    return {"input_ids": np.array(ids), "attention_mask": np.array([0] * n_pad + [1] * (len(ids) - n_pad)),
            "labels": np.array([-100] * (n_pad + len(prompt)) + target),
            "support_tokens": rng.integers(1, 60, (3, 5))}


def args(**kw):
    a = dict(num_generations=4, groups_per_gather=2, noise_std=0.5, noise_relative=False, kl_beta=0.01,
             epsilon=0.2, epsilon_high=0.28, num_iterations=1, temperature=1.0, fp32_logps=False, pad_multiple=8)
    a.update(kw)
    return SimpleNamespace(**a)


def test_padding_and_logps_match_ntp(model, row):
    ids, mask, n_target, extra = G.prompt_tensors(row, PAD, "cpu", 8)
    assert ids.shape[1] % 8 == 0 and int(mask.sum()) == int(row["attention_mask"].sum())
    with torch.no_grad():
        embeds, _, pos = G.build_embeds(model, ids, extra)
        lp = G.target_logps(model, embeds, mask, ids, n_target, 1.0)
        labels = torch.full_like(ids, -100)
        labels[0, -n_target:] = ids[0, -n_target:]
        ntp = model(input_ids=ids, attention_mask=mask, labels=labels, **extra).ntp_loss
    assert torch.allclose(-lp.mean().float(), ntp.float(), atol=1e-5), (-lp.mean(), ntp)
    assert ids[0, pos] == THOUGHT


def test_broadcast_gradient_equals_duplicated_rows(model, row):
    """One thought broadcast to G samples == R3's G identical rows (values and gradients)."""
    ids, mask, n_target, extra = G.prompt_tensors(row, PAD, "cpu", 8)
    noise = torch.randn(4, D, dtype=torch.float64)
    noise[0] = 0
    m1, m2 = copy.deepcopy(model), copy.deepcopy(model)
    e1, _, pos = G.build_embeds(m1, ids, extra)
    lp1 = G.target_logps(m1, G.perturb(e1, pos, noise), mask, ids, n_target, 1.0)
    w = torch.randn_like(lp1)
    (lp1 * w).sum().backward()
    ids4 = ids.expand(4, -1)
    e2, _, _ = G.build_embeds(m2, ids4, {k: v.expand(4, *v.shape[1:]) for k, v in extra.items()})
    e2 = e2.clone()
    e2[:, pos] = e2[:, pos] + noise
    lp2 = G.target_logps(m2, e2, mask, ids4, n_target, 1.0)
    (lp2 * w).sum().backward()
    assert torch.allclose(lp1, lp2, atol=1e-6)
    for (n, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        if p1.grad is not None or p2.grad is not None:
            assert torch.allclose(p1.grad, p2.grad, atol=1e-5, rtol=1e-4), n


def swap_adjacent_blocks(x, k):  # verbatim from R3
    original_shape = x.shape
    x_2d = x.view(-1, k)
    n = x_2d.size(0)
    indices = torch.arange(n).view(-1, 2).flip(1).reshape(-1)
    return x_2d[indices].view(original_shape)


def r3_rollout_and_loss(policy, ref_model, rows, noises, g, beta=0.01, temperature=1.0):
    """R3's _generate_and_score_completions + compute_loss for 2 'GPUs' (one prompt x g rows each), with
    the collator R3 uses (left padding to a multiple of 8). Returns per-GPU rewards, advantages, old / ref
    logps and losses (loss before the Trainer's /grad_accum and DDP mean)."""
    from transformers import DataCollatorForSeq2Seq

    class Tok:  # minimal tokenizer for DataCollatorForSeq2Seq.pad
        pad_token_id, padding_side, model_input_names = PAD, "left", ["input_ids", "attention_mask"]

        def pad(self, features, padding, max_length, pad_to_multiple_of, return_tensors):
            L = max(len(f["input_ids"]) for f in features)
            L = (L + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
            out = {"input_ids": [], "attention_mask": [], "labels": []}
            for f in features:
                n = L - len(f["input_ids"])
                out["input_ids"].append([PAD] * n + list(f["input_ids"]))
                out["attention_mask"].append([0] * n + list(f["attention_mask"]))
                out["labels"].append([-100] * n + list(f["labels"]))
            return {k: torch.tensor(v) for k, v in out.items()}

    def generate_embs(model, ids, supports):  # R3: model.generate_embs(prompt_completion_ids, mask)
        return model.generate_embs(ids, support_tokens=supports)[0]

    def get_logps(model, input_ids, inputs_embeds, attention_mask, logits_to_keep):  # R3 my_get_per_token_logps
        logits = G.Qwen2ForCausalLM.forward(model, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                            use_cache=False).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        logits = logits / temperature
        return selective_log_softmax(logits, input_ids)

    per_gpu = []
    for row, noise in zip(rows, noises):
        start = int(row["attention_mask"].argmax())  # R3 rows are unpadded
        feat = {k: row[k][start:].tolist() for k in ("input_ids", "attention_mask", "labels")}
        batch = Tok().pad([feat] * g, True, None, 8, "pt")
        ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        supports = torch.as_tensor(row["support_tokens"])[None].expand(g, -1, -1)
        labels_length = (labels[0] != -100).sum(dim=-1)
        logits_to_keep = int(labels_length)
        with torch.no_grad():
            original_embeds = generate_embs(policy, ids, supports)
            where = torch.nonzero(ids == THOUGHT)
            original_embeds[torch.arange(g), where[:, 1]] += noise
            old = get_logps(policy, ids, original_embeds, mask, logits_to_keep)
            ref = get_logps(ref_model, ids, original_embeds, mask, logits_to_keep)
        reward = -(-old.clone().sum(dim=-1) / labels_length).exp().to(torch.float32)
        per_gpu.append(dict(ids=ids, mask=mask, supports=supports, noise=noise, old=old, ref=ref, reward=reward,
                            logits_to_keep=logits_to_keep))
    rewards = torch.cat([p["reward"] for p in per_gpu])  # gather
    temp_rewards = rewards.clone().view(-1, g)
    xx = temp_rewards[:, 0].unsqueeze(1).expand_as(temp_rewards).reshape(-1)
    xx = swap_adjacent_blocks(xx, g)
    advantages = rewards - xx.mean()
    advantages = advantages / (torch.norm(advantages) + 1e-6)
    for i, p in enumerate(per_gpu):
        p["adv"] = advantages[i * g:(i + 1) * g]
        embeds = generate_embs(policy, p["ids"], p["supports"])
        where = torch.nonzero(p["ids"] == THOUGHT)
        embeds[torch.arange(g), where[:, 1]] += p["noise"]
        lp = get_logps(policy, p["ids"], embeds, p["mask"], p["logits_to_keep"])
        per_token_kl = torch.exp(p["ref"] - lp) - (p["ref"] - lp) - 1
        old = lp.detach()
        coef_1 = torch.exp(lp - old)
        coef_2 = torch.clamp(coef_1, 1 - 0.2, 1 + 0.28)
        per_token_loss = -torch.min(coef_1 * p["adv"].unsqueeze(1), coef_2 * p["adv"].unsqueeze(1))
        per_token_loss = per_token_loss + beta * per_token_kl
        completion_mask = p["mask"][:, -p["logits_to_keep"]:]
        p["loss"] = (per_token_loss * completion_mask).sum() / completion_mask.sum()
    return per_gpu


def test_matches_r3(rows):
    """bf16 model: rewards, advantages, old/ref logps, loss and gradients equal R3's code on the same noise."""
    torch.manual_seed(1)
    base = tiny_model(torch.bfloat16)
    pol_r3, ref_r3 = copy.deepcopy(base), copy.deepcopy(base).eval()
    pol, ref = copy.deepcopy(base), copy.deepcopy(base).eval()
    with torch.no_grad():  # make the reference differ from the policy so the KL term is non-zero
        for m in (ref_r3, ref):
            m.injectnet.W_v_2.weight.mul_(0.5)
    g = 8
    a = args(num_generations=g, noise_std=1.5)
    gen = torch.Generator().manual_seed(3)
    ours = G.rollout(pol, ref, rows[:2], a, gen, PAD)
    r3 = r3_rollout_and_loss(pol_r3, ref_r3, rows[:2], [o["noise"] for o in ours], g)
    for o, r in zip(ours, r3):
        assert o["ids"].shape == r["ids"][:1].shape and torch.equal(o["ids"][0], r["ids"][0]), "padding"
        assert torch.equal(o["old"], r["old"]), "old logps (bf16)"
        assert torch.equal(o["ref"], r["ref"]), "ref logps (policy rollout embeds)"
        assert torch.equal(o["reward"], r["reward"]), "reward"
        assert torch.allclose(o["adv"], r["adv"]), "advantage"
    pol.zero_grad()
    pol_r3.zero_grad()
    for o, r in zip(ours, r3):
        loss = G.group_loss(pol, o, a)[0]
        assert torch.allclose(loss.float(), r["loss"].float(), atol=1e-6), (loss, r["loss"])
        (loss / 2).backward()
        (r["loss"] / 2).backward()  # DDP mean over the 2 GPUs
    for (n, p1), (_, p2) in zip(pol.named_parameters(), pol_r3.named_parameters()):
        if p2.grad is not None:
            assert p1.grad is not None, n
            # bf16: the broadcast thought sums 8 identical row gradients once, R3 sums them through 8 rows
            assert torch.allclose(p1.grad.float(), p2.grad.float(), atol=2e-2, rtol=5e-2), n


def test_step(model, rows, lr_llm):
    policy = copy.deepcopy(model)
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    groups, n_gen, n_llm = G.param_groups(policy, 1e-2, lr_llm)
    assert n_gen > 0 and n_llm > 0
    opt = torch.optim.AdamW(groups)
    a = args()
    gen = torch.Generator().manual_seed(0)
    gs = G.rollout(policy, ref, rows, a, gen, PAD)
    assert len(gs) == 4 and all(g["adv"].shape == (4,) for g in gs)
    for g in gs:
        assert g["noise"][0].abs().sum() == 0 and g["noise"][1:].abs().sum() > 0
        assert torch.all(g["reward"] < 0)
    unit = torch.cat([gs[0]["adv"], gs[1]["adv"]])
    assert abs(unit.norm().item() - 1) < 1e-4
    loss, kl, clipped = G.group_loss(policy, gs[0], a)  # identical policy and reference => KL 0
    assert kl.abs().item() < 1e-6 and clipped.item() == 0.0
    before = {n: p.detach().clone() for n, p in policy.named_parameters()}
    opt.zero_grad()
    for g in gs:
        (G.group_loss(policy, g, a)[0] / len(gs)).backward()
    opt.step()
    changed = {n for n, p in policy.named_parameters() if not torch.equal(p, before[n])}
    assert any(n.startswith("injectnet.") for n in changed), "generator must update"
    assert any(n.startswith("model.") for n in changed) == (lr_llm > 0), lr_llm
    assert G.group_loss(policy, gs[0], a)[1].item() > 0  # KL to the frozen reference after the update


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = tiny_model()
    rows = [make_row(rng, n_pad=int(rng.integers(0, 4)), n_prompt=int(rng.integers(5, 12)),
                     n_target=int(rng.integers(2, 6))) for _ in range(4)]
    test_padding_and_logps_match_ntp(model, rows[0])
    test_broadcast_gradient_equals_duplicated_rows(model, rows[1])
    test_matches_r3(rows)
    test_step(model, rows, lr_llm=0.0)
    test_step(model, rows, lr_llm=1e-2)
    print("ok: pad-to-8 logps == NTP, broadcast == duplicated rows, rollout/advantages/loss/grads == R3 code (bf16), "
          "GRPO step (LLM frozen / trained)")


if __name__ == "__main__":
    main()
