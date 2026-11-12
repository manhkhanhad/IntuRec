import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from typing import Optional, Tuple, Union
import torch
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn as nn
from model.Layers import VanillaRoPE


class InjectNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.rope = VanillaRoPE(hidden_size)
        self.W_q_2 = nn.Linear(hidden_size, hidden_size)
        self.W_k_2 = nn.Linear(hidden_size, hidden_size)
        self.W_v_2 = nn.Linear(hidden_size, hidden_size)

    def apply_rotary_pos_emb(self, Q, K, costheta, sintheta):
        # Q: [bsz, seqlen, dim], K: [bsz, seqlen, dim], costheta: [bsz, seqlen, seqlen], sintheta: [bsz, seqlen, seqlen]
        Q_negrotary = torch.cat([
                -Q[..., Q.shape[-1] // 2:],
                Q[..., :Q.shape[-1] // 2]
            ],
            dim=-1
        )
        K_negrotary = torch.cat([
                -K[..., K.shape[-1] // 2:],
                K[..., :K.shape[-1] // 2]
            ],
            dim=-1
        )
        Q = Q * costheta + Q_negrotary * sintheta
        K = K * costheta + K_negrotary * sintheta
        return Q, K
    
    def forward(self, query, x, attention_mask):
        # query: [bsz, 1, dim], x: [bsz, N, seqlen, dim], attention_mask: [bsz, N, seqlen]
        bsz, N, seqlen, dim = x.shape
        
        x_ = x.reshape(-1, seqlen, dim)
        attention_mask_ = attention_mask.reshape(-1, seqlen)

        arange = torch.arange(seqlen, device=attention_mask.device).unsqueeze(0)  # [1, seqlen]

        Q = self.W_q(x_)
        K = self.W_k(x_)
        V = self.W_v(x_)

        costheta, sintheta = self.rope(
            torch.arange(0, seqlen, device=x.device)[None, :].float().expand(bsz*N, -1)
        )
        costheta = costheta.to(Q.dtype)
        sintheta = sintheta.to(Q.dtype)
        Q, K = self.apply_rotary_pos_emb(Q, K, costheta, sintheta)

        Q_selected = Q[:, -1:, :] # [bsz*N, 1, dim]

        attn_scores = Q_selected @ K.transpose(-2, -1) / (dim ** 0.5) # [bsz*N, 1, seqlen]
        attn_scores = attn_scores.masked_fill(attention_mask_.unsqueeze(1) == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1) # [bsz*N, 1, seqlen]

        output_selected = torch.matmul(attn_probs, V)  # [bsz*N, 1, dim]
        output = output_selected.squeeze(1).reshape(-1, N, dim) # [bsz, N, dim]

        Q_ = self.W_q_2(query)  # [bsz, 1, dim]
        K_ = self.W_k_2(output) # [bsz, N, dim]
        V_ = self.W_v_2(output) # [bsz, N, dim]

        attn_scores_ = Q_ @ K_.transpose(-2, -1) / (dim ** 0.5) # [bsz, 1, N]
        attn_probs_ = torch.softmax(attn_scores_, dim=-1) # [bsz, 1, N]
        res = torch.matmul(attn_probs_, V_).squeeze(1)  # [bsz, dim]

        return res

    def convertseq2emb(self,x, attention_mask):
        # x: [bsz, seqlen, dim], attention_mask: [bsz, seqlen]
        bsz, seqlen, dim = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        costheta, sintheta = self.rope(
            torch.arange(0, seqlen, device=x.device)[None, :].float().expand(bsz, -1)
        )
        costheta = costheta.to(Q.dtype)
        sintheta = sintheta.to(Q.dtype)
        Q, K = self.apply_rotary_pos_emb(Q, K, costheta, sintheta)

        Q_selected = Q[:, -1:, :] # [bsz, 1, dim]

        attn_scores = Q_selected @ K.transpose(-2, -1) / (dim ** 0.5) # [bsz, 1, seqlen]
        attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1) # [bsz, 1, seqlen]

        output_selected = torch.matmul(attn_probs, V)  # [bsz, 1, dim]
        output = output_selected.squeeze(1) # [bsz, dim]
        return output

class IntuRecF2(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.injectnet = InjectNet(config.hidden_size)
        self.alpha = 1.0
        self.reason_step = 1.0

    def generate_embs(self, input_ids, intuition_tokens, random_tokens, tgt_tokens, tgt_ids):
        # input_ids: [bsz, seqlen]
        # intuition_tokens: [bsz, N, intulen] 
        # random_tokens: [bsz, N, ranlen] 
        # tgt_tokens: [bsz, tgtlen]
        # tgt_ids: [bsz]

        inputs_embeds = self.model.embed_tokens(input_ids) # (bsz, seqlen, dim)
        thought_id = self.model.embed_tokens.num_embeddings - 3  # thought id flag
        idxs = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)  # (bsz, seqlen)
        masked = torch.where((input_ids == thought_id), idxs, input_ids.new_full(idxs.shape, input_ids.shape[1]))
        thought_ids_first = masked.min(dim=1).values  # (bsz)

        mid_state = super().forward(inputs_embeds=inputs_embeds, attention_mask=(input_ids != self.config.pad_token_id).int(), output_hidden_states=True)['hidden_states'][-1] # (bsz, seqlen, dim)
        mid_indices = thought_ids_first.unsqueeze(1).unsqueeze(2).expand(-1, 1, inputs_embeds.shape[-1]) # (bsz, 1, dim)
        his_query = torch.gather(mid_state, 1, mid_indices) # (bsz, 1, dim)

        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)

        inject_embeds = self.model.embed_tokens(intuition_tokens)
        inject_mask = (intuition_tokens != self.config.pad_token_id).int()
        inject_emb = self.injectnet(his_query, inject_embeds, inject_mask).to(inputs_embeds.dtype) # [bsz, dim]

        inputs_embeds[batch_indices, thought_ids_first] = inject_emb

        if self.reason_step > 1:
            for rs in range(1, self.reason_step):
                thought_ids_pre = thought_ids_first + rs - 1
                thought_ids_now = thought_ids_first + rs
                mid_state = super().forward(inputs_embeds=inputs_embeds, attention_mask=(input_ids != self.config.pad_token_id).int(), output_hidden_states=True)['hidden_states'][-1] # (bsz, seqlen, dim)
                rs_emb = mid_state[batch_indices, thought_ids_pre]
                inputs_embeds[batch_indices, thought_ids_now] = rs_emb


        loss = torch.tensor(0.0, device=inputs_embeds.device)
        if tgt_ids is not None:
            assert random_tokens is not None
            assert tgt_tokens is not None

            random_embeds = self.model.embed_tokens(random_tokens)
            random_mask = (random_tokens != self.config.pad_token_id).int()
            random_emb = self.injectnet(his_query, random_embeds, random_mask).to(inputs_embeds.dtype) # [bsz, dim]

            tgt_embeds = self.model.embed_tokens(tgt_tokens)
            tgt_mask = (tgt_tokens != self.config.pad_token_id).int()
            tgt_emb = self.injectnet.convertseq2emb(tgt_embeds, tgt_mask).to(inputs_embeds.dtype) # [bsz, dim]

            pos_logits = torch.sum(inject_emb * tgt_emb, dim=-1)  # [bsz]
            neg_logits = torch.sum(random_emb * tgt_emb, dim=-1)  # [bsz]

            loss = torch.mean(torch.nn.functional.softplus(neg_logits - pos_logits))

        return inputs_embeds, loss


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        intuition_tokens: torch.LongTensor = None,
        random_tokens: torch.LongTensor = None,
        tgt_tokens: torch.LongTensor = None,
        tgt_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
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
        if input_ids is not None and inputs_embeds is None and input_ids.size() == attention_mask.size():            
            inputs_embeds, loss_intu = self.generate_embs(input_ids, intuition_tokens, random_tokens, tgt_tokens, tgt_ids) # [bsz, seqlen, dim]
            input_ids = None

            outputs = super().forward(
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
            loss = outputs.loss if hasattr(outputs, "loss") else None

            if loss is not None and loss_intu is not None:
                combined_loss = loss + self.alpha * loss_intu
            elif loss is not None:
                combined_loss = loss
            elif loss_intu is not None:
                combined_loss = loss_intu
            else:
                combined_loss = None
            outputs.loss=combined_loss
            outputs.intuition_loss=loss_intu
            outputs.ntp_loss=loss
            return outputs
        else:
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