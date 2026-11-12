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
        # How the thought reads the (text) supports:
        #   "inject": IntuRec's InjectNet (softmax over supports; cannot say "none of these").
        #   "gated":  base(query) + sum_s sigmoid(r_s) * attn_s * v_s, where r_s is a relevance logit
        #             per support, supervised by BCE against "is a true motive" (weight sel_weight).
        #             Wrong motives can be switched off; with all gates closed the thought is O0's.
        #   "soft":   base(query) + sum_s softmax(attn_s + log pi_s) * v_s, pi = retriever probabilities
        #             of the supports (probability-weighted motive mixture, no hard selection).
        # Pooling (convertseq2emb) and attention (W_*_2) reuse InjectNet's weights; IntuRec is unmodified.
        self.readout = getattr(config, "readout", "inject")
        self.sel_weight = 0.0
        if self.support_mode == "text":
            self.injectnet = InjectNet(config.hidden_size)
            self.no_support_proj = nn.Linear(config.hidden_size, config.hidden_size)  # SUPPORT_K=0 ablation
            if self.readout == "gated":
                rel_dim = getattr(config, "relevance_dim", 256)
                self.rel_q = nn.Linear(config.hidden_size, rel_dim)
                self.rel_k = nn.Linear(config.hidden_size, rel_dim)
                self.rel_bias = nn.Parameter(torch.zeros(()))
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
        # Collaborative InfoNCE over the refined thought (R3 Attempt 10 applied after motive injection):
        # config.collab_queue = {"bank_size", "projection_dim", "relations", ...} or None (off).
        self.collab = getattr(config, "collab_queue", None) or {}
        if self.collab:
            self.collab_weights = {"user": 0.0, "target": 0.0, "motive": 0.0}
            self.collab_heads = nn.ModuleDict({
                rel: nn.Sequential(nn.LayerNorm(config.hidden_size),
                                   nn.Linear(config.hidden_size, self.collab["projection_dim"], bias=False))
                for rel in self.collab["relations"]})
            size = self.collab["bank_size"]
            # The queue is training-only state: never saved, so a reloaded model starts empty.
            self.register_buffer("bank_r", torch.zeros(size, config.hidden_size, dtype=torch.float16), persistent=False)
            self.register_buffer("bank_user", torch.zeros(size, dtype=torch.int64), persistent=False)
            self.register_buffer("bank_target", torch.zeros(size, dtype=torch.int64), persistent=False)
            self.register_buffer("bank_sample", torch.zeros(size, dtype=torch.int64), persistent=False)
            self.register_buffer("bank_motive", torch.full((size, 2), -1, dtype=torch.int64), persistent=False)
            self.register_buffer("bank_pointer", torch.zeros((), dtype=torch.int64), persistent=False)
            self.register_buffer("bank_count", torch.zeros((), dtype=torch.int64), persistent=False)

    def push_bank(self, thought, user, target, sample, motive):
        """FIFO insert of the admitted rows' refined thoughts and their labels."""
        n = thought.size(0)
        if n == 0:
            return
        size = self.bank_r.size(0)
        slots = (self.bank_pointer + torch.arange(n, device=thought.device)) % size
        self.bank_r[slots] = thought.detach().to(self.bank_r.dtype)
        self.bank_user[slots] = user
        self.bank_target[slots] = target
        self.bank_sample[slots] = sample
        self.bank_motive[slots] = motive
        self.bank_pointer.copy_((self.bank_pointer + n) % size)
        self.bank_count.copy_(torch.clamp(self.bank_count + n, max=size))

    def collab_losses(self, thought, user, target, sample, motive, admit):
        """Multi-positive InfoNCE per relation; candidates = queue + admitted rows of this micro-batch.

        Positives: same user / same target / a shared motive prototype (one projection head each).
        Negatives (all heads): share no user, no target and no motive with the anchor.
        Returns {relation: loss} and {relation: share of anchors with a positive}.
        """
        n = int(self.bank_count)
        cur = torch.nonzero(admit, as_tuple=False).flatten()
        cand_r = torch.cat([self.bank_r[:n].to(thought.device), thought.detach()[cur]], 0)
        cand_user = torch.cat([self.bank_user[:n], user[cur]], 0)
        cand_target = torch.cat([self.bank_target[:n], target[cur]], 0)
        cand_sample = torch.cat([self.bank_sample[:n], sample[cur]], 0)
        cand_motive = torch.cat([self.bank_motive[:n], motive[cur]], 0)
        not_self = sample[:, None] != cand_sample[None, :]
        same = {
            "user": user[:, None] == cand_user[None, :],
            "target": target[:, None] == cand_target[None, :],
            # shared motive id; -1 marks an empty slot (row with fewer than 2 motives)
            "motive": ((motive[:, :, None, None] == cand_motive[None, None, :, :])
                       & (motive[:, :, None, None] >= 0)).any(3).any(1),  # [bsz, 2, n, 2] -> [bsz, n]
        }
        negatives = not_self & ~same["user"] & ~same["target"] & ~same["motive"]
        losses, rates = {}, {}
        for rel, head in self.collab_heads.items():
            positives = not_self & same[rel]
            dtype = head[1].weight.dtype
            anchors = nn.functional.normalize(head(thought.to(dtype)).float(), dim=-1)
            cands = nn.functional.normalize(head(cand_r.to(dtype)).float(), dim=-1)
            per_anchor = []
            for i in range(thought.size(0)):
                pos = torch.nonzero(positives[i], as_tuple=False).flatten()
                neg = torch.nonzero(negatives[i], as_tuple=False).flatten()
                if pos.numel() == 0 or neg.numel() == 0:
                    continue
                pos = pos[torch.randperm(pos.numel(), device=pos.device)[:self.collab["max_positives"]]]
                neg = neg[torch.randperm(neg.numel(), device=neg.device)[:self.collab["num_negatives"]]]
                pos_s = anchors[i] @ cands[pos].T / self.collab["temperature"]
                neg_s = anchors[i] @ cands[neg].T / self.collab["temperature"]
                per_anchor.append(torch.logsumexp(torch.cat([pos_s, neg_s]), 0) - torch.logsumexp(pos_s, 0))
            losses[rel] = torch.stack(per_anchor).mean() if per_anchor else thought.sum() * 0.0
            rates[rel] = positives.any(1).float().mean().item()
        return losses, rates

    def motive_readout(self, his_query, support_tokens, support_logp=None, support_is_true=None):
        """Gated / soft readout over the supports. Returns (thought [bsz, dim], selection BCE, gates)."""
        bsz, n_sup, length = support_tokens.shape
        inj = self.injectnet
        pooled = self.pool_rationales(support_tokens.reshape(-1, length)).reshape(bsz, n_sup, -1)  # [bsz, N, dim]
        attn = (inj.W_q_2(his_query) @ inj.W_k_2(pooled).transpose(-2, -1)).squeeze(1).float()
        attn = attn / (pooled.shape[-1] ** 0.5)  # [bsz, N]
        values = inj.W_v_2(pooled)  # [bsz, N, dim]
        base = self.no_support_proj(his_query).squeeze(1)  # [bsz, dim]
        selection_loss = torch.zeros((), device=his_query.device)
        gates = None
        if self.readout == "soft":
            assert support_logp is not None, "readout 'soft' needs support_logp (retriever log-probabilities)"
            weights = torch.softmax(attn + support_logp.float(), dim=-1)
        else:  # gated
            rel = (self.rel_q(his_query) * self.rel_k(pooled)).sum(-1).float()  # [bsz, N]
            rel = rel / (self.rel_q.out_features ** 0.5) + self.rel_bias.float()
            gates = torch.sigmoid(rel)
            weights = torch.softmax(attn, dim=-1) * gates
            if support_is_true is not None:
                selection_loss = nn.functional.binary_cross_entropy_with_logits(rel, support_is_true.float())
        thought = base + (weights.to(values.dtype).unsqueeze(-1) * values).sum(1)
        return thought, selection_loss, gates

    def pool_rationales(self, tokens):
        # tokens: [n, len] left-padded rationale token ids -> [n, dim] (IntuRec's convertseq2emb)
        return self.injectnet.convertseq2emb(self.model.embed_tokens(tokens),
                                             (tokens != self.config.pad_token_id).int())

    def generate_embs(self, input_ids, support_emb=None, own_rationale=None,
                      support_tokens=None, own_tokens=None, neg_tokens=None,
                      rand_support_tokens=None, anchor_tokens=None, thought_override=None,
                      support_logp=None, support_is_true=None):
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

        selection_loss = torch.zeros((), device=inputs_embeds.device)
        self.last_gates = None
        if self.support_mode == "text":
            if support_tokens is not None and self.readout in ("gated", "soft"):
                thought, selection_loss, gates = self.motive_readout(his_query, support_tokens, support_logp,
                                                                     support_is_true)
                self.last_gates = None if gates is None else gates.detach()
            elif support_tokens is not None:
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
        # Analysis only (collab/latent_diagnostics.py): keep the pre-injection history query, and
        # optionally replace the thought written into the <|Thought|> slot (mask / mean / shuffle).
        self.last_his_query = his_query.squeeze(1).detach()
        if thought_override is not None:
            thought = thought_override.to(inputs_embeds.dtype)
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

        return inputs_embeds, rationale_loss, support_loss, thought, selection_loss

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
        cq_user: Optional[torch.LongTensor] = None,
        cq_target: Optional[torch.LongTensor] = None,
        cq_sample: Optional[torch.LongTensor] = None,
        cq_motives: Optional[torch.LongTensor] = None,
        thought_override: Optional[torch.FloatTensor] = None,
        support_logp: Optional[torch.FloatTensor] = None,
        support_is_true: Optional[torch.FloatTensor] = None,
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
            inputs_embeds, loss_rationale, loss_support, thought, loss_selection = self.generate_embs(
                input_ids, support_emb, own_rationale, support_tokens, own_tokens, neg_tokens,
                rand_support_tokens, anchor_tokens, thought_override, support_logp, support_is_true)
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
                            + self.sel_weight * loss_selection
                            if ntp_loss is not None else None)
            outputs.ntp_loss = ntp_loss
            outputs.rationale_loss = loss_rationale
            outputs.support_loss = loss_support
            outputs.selection_loss = loss_selection
            # Collaborative InfoNCE on the refined thought: training only (valid NTP stays comparable).
            outputs.collab_losses, outputs.collab_rates = {}, {}
            if self.collab and self.training and cq_user is not None and labels is not None:
                with torch.no_grad():  # R3's gate: share of target tokens predicted correctly
                    preds = outputs.logits[:, :-1].argmax(-1)
                    tgt = labels[:, 1:]
                    mask = tgt != -100
                    acc = ((preds == tgt) & mask).sum(1).float() / mask.sum(1).clamp(min=1)
                    admit = acc >= self.collab["min_token_accuracy"]
                losses, rates = self.collab_losses(thought, cq_user, cq_target, cq_sample, cq_motives, admit)
                outputs.loss = outputs.loss + sum(self.collab_weights[r] * losses[r] for r in losses)
                outputs.collab_losses, outputs.collab_rates = losses, rates
                outputs.collab_rates["admit"] = admit.float().mean().item()
                outputs.collab_rates["queue"] = int(self.bank_count)
                self.push_bank(thought[admit], cq_user[admit], cq_target[admit], cq_sample[admit], cq_motives[admit])
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
