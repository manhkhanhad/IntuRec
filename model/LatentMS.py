"""Multi-step naive latent recommender (no intuitions) with a prefix-decoding monotonicity loss.

Thoughts (S = reason_step <|Thought|> slots, built one after another; each step is a full LLM pass):
  thought 1    config.first_thought: "last" = last-layer hidden state just before slot 1 (R3-style readout),
               "ratt" = RATT pooling as for later steps
  thought t>1  RATT (model/IntuRecMS.py, VRec's SelfAttentionLayer): the hidden state just before slot t
               queries all valid earlier positions (history + thoughts 1..t-1). One RATT shared by all steps.
Training loss (labels given):
  L_t      per-row NTP loss of the target title when the decoder attends to thoughts 1..t only
           (slots t+1..S masked out of the attention mask; positions unchanged). L_S = the normal NTP loss.
  L        = NTP(L_S, token mean, as IntuRecF2) + prefix_weight * sum_{t<S} mean_rows(L_t)
             + mono_weight * mean_rows sum_{t=2..S} relu(L_t - stopgrad(L_{t-1}) + mono_margin)
  stopgrad keeps the penalty from being met by making an earlier prefix worse; the margin keeps it from
  being met by a later thought that just repeats the previous one. Without labels (decoding) the model
  writes all S thoughts and decodes normally.
Accepts and ignores IntuRec's intuition inputs, so run_inturec_f2.py's train/eval loop runs unchanged.
Logged as: ntp_loss = L_S; intuition_loss = the unweighted monotonicity term.
"""
import torch
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM

from model.IntuRecMS import RATT

FIRST_THOUGHT = ("last", "ratt")


class LatentMS(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.first_thought = getattr(config, "first_thought", "last")
        assert self.first_thought in FIRST_THOUGHT, self.first_thought
        self.ratt = RATT(config.hidden_size)
        self.reason_step = 1
        self.alpha = 1.0  # set by run_inturec_f2.initial_model; unused
        self.prefix_weight = getattr(config, "prefix_weight", 1.0)
        self.mono_weight = getattr(config, "mono_weight", 1.0)
        self.mono_margin = getattr(config, "mono_margin", 0.02)

    def _thought_first(self, input_ids):
        thought_id = self.model.embed_tokens.num_embeddings - 3  # thought id flag (as IntuRecF2)
        idxs = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        masked = torch.where(input_ids == thought_id, idxs, input_ids.new_full(idxs.shape, input_ids.shape[1]))
        return masked.min(dim=1).values  # (bsz)

    def generate_embs(self, input_ids):
        inputs_embeds = self.model.embed_tokens(input_ids)
        attention_mask = (input_ids != self.config.pad_token_id).int()
        first = self._thought_first(input_ids)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        for rs in range(int(self.reason_step)):
            now = first + rs
            mid_state = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
            if rs == 0 and self.first_thought == "last":
                emb = mid_state[rows, now - 1]
            else:
                emb = self.ratt(mid_state, attention_mask, now)
            inputs_embeds[rows, now] = emb.to(inputs_embeds.dtype)
        return inputs_embeds, first

    def _row_nll(self, hidden, labels):
        """Per-row mean NLL of the label tokens. hidden [bsz, L, dim], labels [bsz, L] (-100 = ignore)."""
        tgt = labels[:, 1:]
        sel = tgt != -100
        logits = self.lm_head(hidden[:, :-1][sel]).float()  # only the answer positions
        nll = F.cross_entropy(logits, tgt[sel], reduction="none")
        row = torch.arange(labels.shape[0], device=labels.device).unsqueeze(1).expand_as(tgt)[sel]
        tot = torch.zeros(labels.shape[0], device=labels.device).index_add_(0, row, nll)
        cnt = torch.zeros(labels.shape[0], device=labels.device).index_add_(0, row, torch.ones_like(nll))
        return tot / cnt.clamp(min=1), nll.sum(), cnt.sum()

    def forward(self, input_ids=None, attention_mask=None, intuition_tokens=None, random_tokens=None,
                tgt_tokens=None, tgt_ids=None, labels=None, inputs_embeds=None, **kwargs):
        first_call = input_ids is not None and inputs_embeds is None and input_ids.size() == attention_mask.size()
        if not first_call:  # decoding steps after the prompt (KV cache)
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds,
                                   labels=labels, **kwargs)
        inputs_embeds, first = self.generate_embs(input_ids)
        if labels is None:  # generation: prompt pass
            return super().forward(attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

        S = int(self.reason_step)
        hidden = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
        row_L, tok_sum, tok_cnt = self._row_nll(hidden, labels)
        ntp_loss = tok_sum / tok_cnt  # token mean over the batch = HF causal-LM loss
        row_losses = [None] * (S - 1) + [row_L]
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        for t in range(1, S):  # prefix: thoughts 1..t visible
            mask_t = attention_mask.clone()
            for s in range(t, S):
                mask_t[rows, first + s] = 0
            h_t = self.model(inputs_embeds=inputs_embeds, attention_mask=mask_t).last_hidden_state
            row_losses[t - 1] = self._row_nll(h_t, labels)[0]
        prefix_loss = sum(l.mean() for l in row_losses[:-1]) if S > 1 else ntp_loss.new_zeros(())
        mono = ntp_loss.new_zeros(())
        for t in range(1, S):
            mono = mono + F.relu(row_losses[t] - row_losses[t - 1].detach() + self.mono_margin).mean()
        loss = ntp_loss + self.prefix_weight * prefix_loss + self.mono_weight * mono
        self.last_prefix_losses = [float(l.mean()) for l in row_losses]
        # per-step log (the train loop only prints ntp / "intuition" = mono): mean L_1..L_S every 1280 calls
        acc = getattr(self, "_log_acc", None)
        self._log_acc = acc = acc if acc is not None and len(acc) == S else [0.0] * S
        self._log_acc = [a + float(l.mean()) for a, l in zip(acc, row_losses)]
        self._log_n = getattr(self, "_log_n", 0) + 1
        if self._log_n % 1280 == 0:
            print(f"[LatentMS {'train' if self.training else 'eval'}] mean prefix NLL L_1..L_S over last 1280 calls: "
                  + " ".join(f"{a / 1280:.4f}" for a in self._log_acc), flush=True)
            self._log_acc = [0.0] * S
        out = type("Out", (), {})()
        out.loss, out.ntp_loss, out.intuition_loss = loss, ntp_loss, mono.detach()
        return out
