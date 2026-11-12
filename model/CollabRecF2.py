from typing import Optional, Tuple, Union
import torch
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn as nn
from model.IntuRecF2 import InjectNet


class SupportAttention(nn.Module):
    """Builds the thought vector from the history query and the support rationales.

    thought = W_r(query) + softmax(W_q(query) W_k(S)^T / sqrt(d)) W_v(S), where S are the
    projected support rationale embeddings. Without supports only the residual remains.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_r = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, supports=None):
        # query: [bsz, 1, dim], supports: [bsz, M, dim]
        res = self.W_r(query).squeeze(1)  # [bsz, dim]
        if supports is None:
            return res
        attn_scores = self.W_q(query) @ self.W_k(supports).transpose(-2, -1) / (self.hidden_size ** 0.5)  # [bsz, 1, M]
        attn_probs = torch.softmax(attn_scores, dim=-1)
        return res + torch.matmul(attn_probs, self.W_v(supports)).squeeze(1)  # [bsz, dim]


class CollabRecF2(Qwen2ForCausalLM):
    """Latent collaborative reasoning recommender.

    The <|Thought|> embedding is built from the history hidden state attending over the
    rationales of collaboratively retrieved support samples, and is aligned with the row's
    own rationale by an InfoNCE loss (weight beta).

    config.support_mode selects how rationales enter the model:
      "text": rationale tokens are embedded with the LLM's own embed_tokens and pooled by
              IntuRec's InjectNet (as IntuRec does for intuition titles); the aux target is the
              own rationale pooled by injectnet.convertseq2emb (as IntuRec's tgt_emb), against
              random train rationales pooled the same way. No external embedding space.
      "emb":  precomputed sentence-embedding vectors with a learned projection (first version).
    """

    def __init__(self, config):
        super().__init__(config)
        self.support_mode = getattr(config, "support_mode", "emb")
        if self.support_mode == "text":
            self.injectnet = InjectNet(config.hidden_size)
            self.no_support_proj = nn.Linear(config.hidden_size, config.hidden_size)  # SUPPORT_K=0 ablation
        else:
            rationale_dim = getattr(config, "rationale_dim", 768)
            self.support_proj = nn.Linear(rationale_dim, config.hidden_size)
            self.support_attn = SupportAttention(config.hidden_size)
            self.rationale_head = nn.Linear(config.hidden_size, rationale_dim)
        self.reason_step = 1
        self.beta = 0.0
        # Contrastive support loss (IntuRec's intuition-loss analog), weight gamma: the thought built
        # from the retrieved supports must score the anchor (target title or ground-truth motives,
        # training only) higher than a thought built from random supports.
        self.gamma = 0.0
        self.aux_temperature = 0.05
        self.aux_negatives = 1024
        self.rationale_bank = None  # [n_train, rationale_dim], set by the training script

    def pool_rationales(self, tokens):
        # tokens: [n, len] left-padded rationale token ids -> [n, dim] (IntuRec's convertseq2emb)
        return self.injectnet.convertseq2emb(self.model.embed_tokens(tokens),
                                             (tokens != self.config.pad_token_id).int())

    def generate_embs(self, input_ids, support_emb=None, own_rationale=None,
                      support_tokens=None, own_tokens=None, neg_tokens=None,
                      rand_support_tokens=None, anchor_tokens=None):
        # input_ids: [bsz, seqlen]
        # "emb" mode:  support_emb [bsz, M, rationale_dim], own_rationale [bsz, rationale_dim]
        # "text" mode: support_tokens [bsz, M, len], own_tokens [bsz, len], neg_tokens [bsz, N, len]
        # (own_* / neg_* only in train/valid: aux-loss targets, never inputs)
        # rand_support_tokens [bsz, M, len]: random supports; anchor_tokens [bsz, A, len]: target title (A=1)
        # or ground-truth motive texts (A=2); both train/valid only, for the contrastive support loss
        inputs_embeds = self.model.embed_tokens(input_ids)  # (bsz, seqlen, dim)
        thought_id = self.model.embed_tokens.num_embeddings - 3  # thought id flag
        idxs = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        masked = torch.where((input_ids == thought_id), idxs, input_ids.new_full(idxs.shape, input_ids.shape[1]))
        thought_ids_first = masked.min(dim=1).values  # (bsz)

        mid_state = super().forward(inputs_embeds=inputs_embeds, attention_mask=(input_ids != self.config.pad_token_id).int(), output_hidden_states=True)['hidden_states'][-1]  # (bsz, seqlen, dim)
        mid_indices = thought_ids_first.unsqueeze(1).unsqueeze(2).expand(-1, 1, inputs_embeds.shape[-1])
        his_query = torch.gather(mid_state, 1, mid_indices)  # (bsz, 1, dim)

        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)

        if self.support_mode == "text":
            if support_tokens is not None:
                support_embeds = self.model.embed_tokens(support_tokens)  # [bsz, M, len, dim]
                support_mask = (support_tokens != self.config.pad_token_id).int()
                thought = self.injectnet(his_query, support_embeds, support_mask)  # [bsz, dim]
            else:
                thought = self.no_support_proj(his_query).squeeze(1)
        else:
            supports = None
            if support_emb is not None:
                supports = self.support_proj(support_emb.to(self.support_proj.weight.dtype))  # [bsz, M, dim]
            thought = self.support_attn(his_query, supports)  # [bsz, dim]
        thought = thought.to(inputs_embeds.dtype)
        inputs_embeds[batch_indices, thought_ids_first] = thought

        if self.reason_step > 1:
            for rs in range(1, self.reason_step):
                thought_ids_pre = thought_ids_first + rs - 1
                thought_ids_now = thought_ids_first + rs
                mid_state = super().forward(inputs_embeds=inputs_embeds, attention_mask=(input_ids != self.config.pad_token_id).int(), output_hidden_states=True)['hidden_states'][-1]
                inputs_embeds[batch_indices, thought_ids_now] = mid_state[batch_indices, thought_ids_pre]

        support_loss = torch.tensor(0.0, device=inputs_embeds.device)
        if (self.support_mode == "text" and self.gamma > 0 and support_tokens is not None
                and rand_support_tokens is not None and anchor_tokens is not None):
            rand_thought = self.injectnet(his_query, self.model.embed_tokens(rand_support_tokens),
                                          (rand_support_tokens != self.config.pad_token_id).int())  # [bsz, dim]
            bsz, n_anchor, length = anchor_tokens.shape
            anchor = self.pool_rationales(anchor_tokens.reshape(-1, length)).reshape(bsz, n_anchor, -1).mean(1)
            pos = torch.sum(thought.float() * anchor.float(), dim=-1)
            neg = torch.sum(rand_thought.float() * anchor.float(), dim=-1)
            support_loss = torch.mean(torch.nn.functional.softplus(neg - pos))

        rationale_loss = torch.tensor(0.0, device=inputs_embeds.device)
        if self.support_mode == "text" and own_tokens is not None and neg_tokens is not None and self.beta > 0:
            bsz, n_neg, length = neg_tokens.shape
            own_emb = self.pool_rationales(own_tokens)  # [bsz, dim]
            neg_emb = self.pool_rationales(neg_tokens.reshape(-1, length)).reshape(bsz, n_neg, -1)  # [bsz, N, dim]
            z = torch.nn.functional.normalize(thought.float(), dim=-1)
            pos = torch.sum(z * torch.nn.functional.normalize(own_emb.float(), dim=-1), dim=-1, keepdim=True)
            neg = torch.einsum("bd,bnd->bn", z, torch.nn.functional.normalize(neg_emb.float(), dim=-1))
            logits = torch.cat([pos, neg], dim=-1) / self.aux_temperature
            rationale_loss = torch.nn.functional.cross_entropy(
                logits, torch.zeros(bsz, dtype=torch.long, device=z.device))
        elif own_rationale is not None and self.beta > 0 and self.rationale_bank is not None:
            z = torch.nn.functional.normalize(self.rationale_head(thought).float(), dim=-1)  # [bsz, rdim]
            pos = torch.sum(z * own_rationale.float(), dim=-1, keepdim=True)  # [bsz, 1]
            neg_idx = torch.randint(0, self.rationale_bank.shape[0], (self.aux_negatives,), device=z.device)
            neg = z @ self.rationale_bank[neg_idx].float().T  # [bsz, n_neg]
            logits = torch.cat([pos, neg], dim=-1) / self.aux_temperature
            rationale_loss = torch.nn.functional.cross_entropy(
                logits, torch.zeros(len(logits), dtype=torch.long, device=z.device))

        return inputs_embeds, rationale_loss, support_loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        support_emb: Optional[torch.FloatTensor] = None,
        own_rationale: Optional[torch.FloatTensor] = None,
        support_tokens: Optional[torch.LongTensor] = None,
        own_tokens: Optional[torch.LongTensor] = None,
        neg_tokens: Optional[torch.LongTensor] = None,
        rand_support_tokens: Optional[torch.LongTensor] = None,
        anchor_tokens: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # The thought embedding is built only on the first (full-prompt) step; later
        # generation steps run on cached keys/values as usual.
        if input_ids is not None and inputs_embeds is None and input_ids.size() == attention_mask.size():
            inputs_embeds, loss_rationale, loss_support = self.generate_embs(
                input_ids, support_emb, own_rationale, support_tokens, own_tokens, neg_tokens,
                rand_support_tokens, anchor_tokens)
            outputs = super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
            ntp_loss = outputs.loss if hasattr(outputs, "loss") else None
            outputs.loss = (ntp_loss + self.beta * loss_rationale + self.gamma * loss_support
                            if ntp_loss is not None else None)
            outputs.ntp_loss = ntp_loss
            outputs.rationale_loss = loss_rationale
            outputs.support_loss = loss_support
            return outputs
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
