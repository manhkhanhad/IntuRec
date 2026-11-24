"""Multi-thought MotiveRec: CollabRecF2 (text mode, inject readout) with S = reason_step <|Thought|> slots that
are each built from retrieved motives, instead of thought 1 from motives and thoughts 2..S copied hidden states.

Thoughts are written one after another; each step is a full LLM pass. For thought t the query is the last-layer
hidden state at slot t with thoughts 1..t-1 already in place (thought 1: CollabRecF2's history query).
config.thought_support:
  split   the K retrieved motives (retriever rank order) are divided into S contiguous groups and thought t reads
          group t; K = S gives one motive per thought. With a single motive the readout does not use the query
          (softmax over one support), so the thought is the pooled motive passed through InjectNet.
  shared  every thought reads all K motives, with its own query.
  first   only thought 1 reads the K motives; thoughts 2..S are the last-layer hidden state at the previous
          thought slot (CollabRecF2 / IntuRec's own reason_step > 1 recurrence, "copy").
  first_norm  as first, but each copied state is rescaled per row to the norm of thought 1 (last-layer states are
          ~60x larger than injected thoughts / input embeddings).
  refine  "broad, then refine": thought 1 reads all K motives with the MotiveRec reader (in practice their mean:
          its attention is uniform). Thought t >= 2 selects among the same K motives with a sharp attention whose
          query is the last-layer state at the previous thought slot (history + thoughts 1..t-1):
            a = softmax(scale * cos(refine_q(h_{t-1}), refine_k(p_j))),  thought_t = sum_j a_j W_v_2(p_j)
          p_j = pooled motive j (InjectNet pooling), W_v_2 = InjectNet's value map (same space as thought 1),
          scale = exp(refine_log_scale), learnable, initial 10. Meant for a larger pool (support_k 10).
Progressive step loss (config.prl_weight > 0; PRL of ReaRec, arXiv 2503.22675, Eq. 9-10, without its contrastive
term): besides the usual NTP on the full model (all S thoughts, temperature 1), the passes that already build the
thoughts are also trained to predict the target: pass k (k = 0..S-1) has thoughts 1..k in place and the raw
<|Thought|> embedding in the later slots, and gets CE(logits / tau_k) with tau_k = prl_alpha ** (S - k).
total = NTP + prl_weight * sum_k L_k. Train/valid only; reported in the "Support Loss" log column.
The reader (InjectNet / pooling) is shared by all thoughts. Objective: NTP only (beta = gamma = 0), as MotiveRec.
The motive head (motive_head_weight) and train-time motive insertion work unchanged; the gated readout does not.
"""
import inspect
import math

import torch

from model.CollabRecF2 import CollabRecF2

THOUGHT_SUPPORT = ("split", "shared", "first", "first_norm", "refine")


class MotiveRecMS(CollabRecF2):
    def __init__(self, config):
        super().__init__(config)
        self.thought_support = getattr(config, "thought_support", "split")
        assert self.thought_support in THOUGHT_SUPPORT, self.thought_support
        assert self.support_mode == "text" and self.readout == "inject", "MotiveRecMS: text mode, inject readout"
        self.prl_weight = float(getattr(config, "prl_weight", 0.0) or 0.0)
        self.prl_alpha = float(getattr(config, "prl_alpha", 1.5) or 1.5)
        self._prl_labels, self._prl_loss = None, None
        if self.thought_support == "refine":
            rdim = getattr(config, "refine_dim", 256)
            self.refine_q = torch.nn.Linear(config.hidden_size, rdim)
            self.refine_k = torch.nn.Linear(config.hidden_size, rdim)
            self.refine_log_scale = torch.nn.Parameter(torch.tensor(math.log(10.0)))
        self.last_refine_attn = None
        self._refine_stats, self._refine_calls = [0.0, 0.0], 0

    def refine(self, state, pooled):
        """state [bsz, dim], pooled motives [bsz, K, dim] -> (thought [bsz, dim], attention [bsz, K])."""
        q = torch.nn.functional.normalize(self.refine_q(state).float(), dim=-1)
        k = torch.nn.functional.normalize(self.refine_k(pooled).float(), dim=-1)
        attn = torch.softmax((k @ q.unsqueeze(-1)).squeeze(-1) * self.refine_log_scale.float().exp(), dim=-1)
        values = self.injectnet.W_v_2(pooled)
        return (attn.to(values.dtype).unsqueeze(-1) * values).sum(1), attn

    def step_loss(self, mid_state, labels, tau):
        """Temperature-scaled CE of the target tokens from one pass's last-layer states."""
        keep = labels[:, 1:] != -100  # position i predicts token i + 1
        logits = self.lm_head(mid_state[:, :-1][keep]).float() / tau
        return torch.nn.functional.cross_entropy(logits, labels[:, 1:][keep])

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        self._prl_labels = labels if self.prl_weight > 0 else None
        self._prl_loss = None
        out = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        if self._prl_loss is not None and getattr(out, "loss", None) is not None:
            out.loss = out.loss + self.prl_weight * self._prl_loss
            out.support_loss = self._prl_loss.detach()
        self._prl_labels = None
        return out

    def step_supports(self, support_tokens, t, S):
        """support_tokens [bsz, K, len] -> the supports thought t (0-based) reads."""
        if self.thought_support in ("shared", "first", "first_norm", "refine"):
            return support_tokens
        K = support_tokens.shape[1]
        assert K >= S, f"split needs support_k >= reason_step ({K} < {S})"
        return support_tokens[:, t * K // S:(t + 1) * K // S]

    def generate_embs(self, input_ids, support_emb=None, own_rationale=None, support_tokens=None, own_tokens=None,
                      neg_tokens=None, rand_support_tokens=None, anchor_tokens=None, support_is_true=None):
        assert self.beta == 0 and self.gamma == 0, "MotiveRecMS is NTP only (beta = gamma = 0)"
        assert support_tokens is not None, "MotiveRecMS needs supports (support_k > 0)"
        S = int(self.reason_step)
        inputs_embeds = self.model.embed_tokens(input_ids)
        attention_mask = (input_ids != self.config.pad_token_id).int()
        thought_id = self.model.embed_tokens.num_embeddings - 3
        idxs = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        first = torch.where(input_ids == thought_id, idxs, input_ids.new_full(idxs.shape, input_ids.shape[1])).min(1).values
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        pooled = None
        if self.thought_support == "refine" and S > 1:
            bsz, K, length = support_tokens.shape
            pooled = self.pool_rationales(support_tokens.reshape(-1, length)).reshape(bsz, K, -1)
        for t in range(S):
            now = first + t
            mid_state = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
            if self._prl_labels is not None:
                step = self.step_loss(mid_state, self._prl_labels, self.prl_alpha ** (S - t))
                self._prl_loss = step if self._prl_loss is None else self._prl_loss + step
            if t > 0 and self.thought_support in ("first", "first_norm"):
                thought = mid_state[rows, now - 1]
                if self.thought_support == "first_norm":
                    thought = thought * (ref_norm / thought.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)).to(thought.dtype)
            elif t > 0 and self.thought_support == "refine":
                thought, attn = self.refine(mid_state[rows, now - 1], pooled)
                self.last_refine_attn = attn.detach()
                if self.training:
                    ent = -(attn * attn.clamp_min(1e-12).log()).sum(-1).mean()
                    self._refine_stats[0] += float(ent); self._refine_stats[1] += float(attn.max(-1).values.mean())
                    self._refine_calls += 1
                    if self._refine_calls % 1280 == 0:
                        print(f"[MotiveRecMS refine] last 1280 selections: entropy {self._refine_stats[0] / 1280:.3f} "
                              f"(uniform {math.log(attn.shape[-1]):.3f}), max weight {self._refine_stats[1] / 1280:.3f}, "
                              f"scale {float(self.refine_log_scale.float().exp()):.2f}", flush=True)
                        self._refine_stats = [0.0, 0.0]
            else:
                query = mid_state[rows, now].unsqueeze(1)  # [bsz, 1, dim]
                thought = self.inject(query, self.step_supports(support_tokens, t, S))  # [bsz, dim]
            if t == 0:
                ref_norm = thought.float().norm(dim=-1, keepdim=True)  # [bsz, 1], thought 1's norm
            inputs_embeds = inputs_embeds.index_put((rows, now), thought.to(inputs_embeds.dtype))  # out of place
        zero = torch.tensor(0.0, device=inputs_embeds.device)
        return inputs_embeds, zero, zero


# generate() validates its kwargs (support_tokens, ...) against forward's signature: expose CollabRecF2's.
MotiveRecMS.forward.__signature__ = inspect.signature(CollabRecF2.forward)
