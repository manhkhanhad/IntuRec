"""Multi-step IntuRec: IntuRecF2 with a choice of how thoughts 2..S are produced.

Thought 1 is IntuRec's own: InjectNet(history query, intuitions), and the intuition loss is applied to
thought 1 only, as in IntuRecF2. Steps 2..S (config.step_mode):
  copy  IntuRecF2's recurrence: thought t = last-layer hidden state at thought position t-1 (R3-style).
  ratt  VRec's RATT (Verifiable-Rec/code/src/finetune/layers.py, SelfAttentionLayer): the hidden state
        just before thought t queries, with RoPE, all valid positions before thought t (history + earlier
        thoughts); thought t = the attention-weighted value. One RATT module shared by all steps.
IntuRec's own files are not modified; this subclasses IntuRecF2.
"""
import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from model.IntuRecF2 import IntuRecF2
from model.Layers import VanillaRoPE

STEP_MODES = ("copy", "ratt")


class RATT(nn.Module):
    """VRec's SelfAttentionLayer (end_k = -1), vectorised: one query per row at index idx - 1,
    keys/values over positions < idx that the attention mask keeps."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.rope = VanillaRoPE(hidden_size)

    def forward(self, hidden_states, attention_mask, thought_idx):
        # hidden_states [bsz, seqlen, dim], attention_mask [bsz, seqlen], thought_idx [bsz] -> [bsz, dim]
        bsz, seqlen, _ = hidden_states.shape
        hidden_states = hidden_states.to(self.query.weight.dtype)
        pos = torch.arange(seqlen, device=hidden_states.device)
        cos, sin = self.rope(pos.unsqueeze(0))  # IntuRec's VanillaRoPE (same base 1000 as VRec's)
        cos, sin = cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)
        q, k = apply_rotary_pos_emb(self.query(hidden_states), self.key(hidden_states),
                                    cos.squeeze(0), sin.squeeze(0), unsqueeze_dim=0)
        v = self.value(hidden_states)
        rows = torch.arange(bsz, device=hidden_states.device)
        q_sel = q[rows, thought_idx - 1].unsqueeze(1)  # [bsz, 1, dim]
        scores = (q_sel @ k.transpose(-2, -1)).squeeze(1).float() / self.hidden_size ** 0.5  # [bsz, seqlen]
        keep = attention_mask.bool() & (pos.unsqueeze(0) < thought_idx.unsqueeze(1))
        scores = scores.masked_fill(~keep, float("-inf"))
        attn = torch.softmax(scores, dim=-1).to(v.dtype)
        return (attn.unsqueeze(1) @ v).squeeze(1)


class IntuRecMS(IntuRecF2):
    def __init__(self, config):
        super().__init__(config)
        self.step_mode = getattr(config, "step_mode", "copy")
        assert self.step_mode in STEP_MODES, self.step_mode
        if self.step_mode == "ratt":
            self.ratt = RATT(config.hidden_size)

    def generate_embs(self, input_ids, intuition_tokens, random_tokens, tgt_tokens, tgt_ids):
        # Same as IntuRecF2.generate_embs except the step >= 2 update (see module docstring).
        inputs_embeds = self.model.embed_tokens(input_ids)  # (bsz, seqlen, dim)
        attention_mask = (input_ids != self.config.pad_token_id).int()
        thought_id = self.model.embed_tokens.num_embeddings - 3  # thought id flag
        idxs = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        masked = torch.where((input_ids == thought_id), idxs, input_ids.new_full(idxs.shape, input_ids.shape[1]))
        thought_ids_first = masked.min(dim=1).values  # (bsz)

        mid_state = super(IntuRecF2, self).forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                                   output_hidden_states=True)['hidden_states'][-1]
        mid_indices = thought_ids_first.unsqueeze(1).unsqueeze(2).expand(-1, 1, inputs_embeds.shape[-1])
        his_query = torch.gather(mid_state, 1, mid_indices)  # (bsz, 1, dim)
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)

        inject_embeds = self.model.embed_tokens(intuition_tokens)
        inject_mask = (intuition_tokens != self.config.pad_token_id).int()
        inject_emb = self.injectnet(his_query, inject_embeds, inject_mask).to(inputs_embeds.dtype)  # [bsz, dim]
        inputs_embeds[batch_indices, thought_ids_first] = inject_emb

        for rs in range(1, int(self.reason_step)):
            thought_ids_pre = thought_ids_first + rs - 1
            thought_ids_now = thought_ids_first + rs
            mid_state = super(IntuRecF2, self).forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                                       output_hidden_states=True)['hidden_states'][-1]
            if self.step_mode == "ratt":
                rs_emb = self.ratt(mid_state, attention_mask, thought_ids_now)
            else:
                rs_emb = mid_state[batch_indices, thought_ids_pre]
            inputs_embeds[batch_indices, thought_ids_now] = rs_emb.to(inputs_embeds.dtype)

        loss = torch.tensor(0.0, device=inputs_embeds.device)
        if tgt_ids is not None:  # intuition loss on thought 1 only (IntuRecF2, unchanged)
            assert random_tokens is not None and tgt_tokens is not None
            random_embeds = self.model.embed_tokens(random_tokens)
            random_mask = (random_tokens != self.config.pad_token_id).int()
            random_emb = self.injectnet(his_query, random_embeds, random_mask).to(inputs_embeds.dtype)
            tgt_embeds = self.model.embed_tokens(tgt_tokens)
            tgt_mask = (tgt_tokens != self.config.pad_token_id).int()
            tgt_emb = self.injectnet.convertseq2emb(tgt_embeds, tgt_mask).to(inputs_embeds.dtype)
            pos_logits = torch.sum(inject_emb * tgt_emb, dim=-1)
            neg_logits = torch.sum(random_emb * tgt_emb, dim=-1)
            loss = torch.mean(torch.nn.functional.softplus(neg_logits - pos_logits))
        return inputs_embeds, loss
