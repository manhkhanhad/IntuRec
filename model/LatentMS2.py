"""LatentMS (model/LatentMS.py: loss, logging, decoding unchanged) with thoughts 2..S built by
model/RATTNorm.RATTNorm instead of the saturating RATT. Thought 1 = hidden state before slot 1 (as
first_thought=last). config.ratt_residual (default 1) = add the copy-recurrence state to the pooled value."""
import torch

from model.LatentMS import LatentMS
from model.RATTNorm import RATTNorm


class LatentMS2(LatentMS):
    def __init__(self, config):
        super().__init__(config)
        assert self.first_thought == "last", "LatentMS2 builds thought 1 from the last hidden state"
        self.ratt = RATTNorm(config.hidden_size, residual=getattr(config, "ratt_residual", 1))

    def generate_embs(self, input_ids):
        inputs_embeds = self.model.embed_tokens(input_ids)
        attention_mask = (input_ids != self.config.pad_token_id).int()
        first = self._thought_first(input_ids)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        for rs in range(int(self.reason_step)):
            now = first + rs
            mid_state = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
            if rs == 0:
                emb = mid_state[rows, now - 1]
            else:
                emb = self.ratt(mid_state, attention_mask, now, inputs_embeds[rows, now - 1].float().norm(dim=-1))
            inputs_embeds = inputs_embeds.index_put((rows, now), emb.to(inputs_embeds.dtype))  # out of place
        return inputs_embeds, first
