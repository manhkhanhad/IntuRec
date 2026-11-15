"""IntuRecMS step_mode=ratt (model/IntuRecMS.py) with thoughts 2..S built by model/RATTNorm.RATTNorm instead
of the saturating RATT. Thought 1 (InjectNet) and the intuition loss on thought 1 are unchanged.
config.ratt_residual (default 1) = add the copy-recurrence state to the pooled value."""
import torch

from model.IntuRecF2 import IntuRecF2
from model.RATTNorm import RATTNorm


class IntuRecMS2(IntuRecF2):
    def __init__(self, config):
        super().__init__(config)
        self.ratt = RATTNorm(config.hidden_size, residual=getattr(config, "ratt_residual", 1))

    def generate_embs(self, input_ids, intuition_tokens, random_tokens, tgt_tokens, tgt_ids):
        # IntuRecMS.generate_embs with the step >= 2 update replaced.
        inputs_embeds = self.model.embed_tokens(input_ids)
        attention_mask = (input_ids != self.config.pad_token_id).int()
        thought_id = self.model.embed_tokens.num_embeddings - 3
        idxs = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        masked = torch.where((input_ids == thought_id), idxs, input_ids.new_full(idxs.shape, input_ids.shape[1]))
        first = masked.min(dim=1).values

        mid_state = super(IntuRecF2, self).forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                                   output_hidden_states=True)['hidden_states'][-1]
        his_query = torch.gather(mid_state, 1, first.unsqueeze(1).unsqueeze(2).expand(-1, 1, inputs_embeds.shape[-1]))
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        inject_embeds = self.model.embed_tokens(intuition_tokens)
        inject_mask = (intuition_tokens != self.config.pad_token_id).int()
        inject_emb = self.injectnet(his_query, inject_embeds, inject_mask).to(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.index_put((rows, first), inject_emb)  # out of place

        for rs in range(1, int(self.reason_step)):
            now = first + rs
            mid_state = super(IntuRecF2, self).forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                                       output_hidden_states=True)['hidden_states'][-1]
            emb = self.ratt(mid_state, attention_mask, now, inputs_embeds[rows, now - 1].float().norm(dim=-1))
            inputs_embeds = inputs_embeds.index_put((rows, now), emb.to(inputs_embeds.dtype))  # out of place

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
