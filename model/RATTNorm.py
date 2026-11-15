"""RATT with normalised inputs (fix for the saturated attention of model/IntuRecMS.RATT).

Problem found on run 366074 (CDs, IntuRecMS ratt rs2): Qwen's last-layer hidden states have norm ~166, and
RATT divides query.key only by sqrt(dim), so the softmax is near one-hot from the first step (entropy 0.48
against 5.08 uniform at initialisation). After one epoch the attention sat on one template token for every
user, thought 2 was the same vector for all users (cross-user cosine 0.997, norm 129 against 2.3 for
thought 1), and masking it left the loss unchanged. Changes here:
  - hidden states are RMS-normalised before query / key / value, so the scores start small and the softmax
    starts diffuse (entropy 4.89 at initialisation) and passes gradient to query and key
  - a learnable temperature exp(log_temp) on the scores (starts at 1) lets the attention sharpen
  - residual (optional, default on): the normalised query state (hidden state just before the thought slot,
    the "copy" recurrence) is added to the pooled value, so the thought is user-specific from the first step
    (the diffuse pooled value alone starts as a near mean-pool, cross-user cosine 0.97)
  - the output is rescaled to ref_norm * exp(log_gain): ref_norm = the norm of the previous thought
    (detached), so a later thought enters the LLM at the same scale as the one before it
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from model.Layers import VanillaRoPE


class RATTNorm(nn.Module):
    def __init__(self, hidden_size, residual=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.residual = bool(residual)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.log_temp = nn.Parameter(torch.zeros(()))
        self.log_gain = nn.Parameter(torch.zeros(()))
        self.rope = VanillaRoPE(hidden_size)

    def attention(self, hidden_states, attention_mask, thought_idx):
        """-> attn [bsz, seqlen] (float32), x = normalised hidden states."""
        bsz, seqlen, _ = hidden_states.shape
        x = hidden_states.float()
        x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).to(self.query.weight.dtype)
        pos = torch.arange(seqlen, device=x.device)
        cos, sin = self.rope(pos.unsqueeze(0))
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        q, k = apply_rotary_pos_emb(self.query(x), self.key(x), cos.squeeze(0), sin.squeeze(0), unsqueeze_dim=0)
        rows = torch.arange(bsz, device=x.device)
        q_sel = q[rows, thought_idx - 1].unsqueeze(1)
        scores = (q_sel @ k.transpose(-2, -1)).squeeze(1).float() / self.hidden_size ** 0.5
        scores = scores * self.log_temp.float().exp()
        keep = attention_mask.bool() & (pos.unsqueeze(0) < thought_idx.unsqueeze(1))
        return torch.softmax(scores.masked_fill(~keep, float("-inf")), dim=-1), x

    def forward(self, hidden_states, attention_mask, thought_idx, ref_norm):
        # hidden_states [bsz, seqlen, dim], thought_idx [bsz], ref_norm [bsz] -> [bsz, dim] (float32)
        attn, x = self.attention(hidden_states, attention_mask, thought_idx)
        out = (attn.to(x.dtype).unsqueeze(1) @ self.value(x)).squeeze(1)
        if self.residual:
            out = out + x[torch.arange(x.shape[0], device=x.device), thought_idx - 1]
        out = F.normalize(out.float(), dim=-1)
        return out * (ref_norm.detach().float() * self.log_gain.float().exp()).unsqueeze(-1)
