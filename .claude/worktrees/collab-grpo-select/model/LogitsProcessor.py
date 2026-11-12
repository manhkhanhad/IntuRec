import torch
from typing import Callable, List
from transformers.generation import LogitsProcessor
from transformers.utils import add_start_docstrings

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""


class IntuRecLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        tokenizer,
        model,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        guide_token="<|StartIntuition|>",
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.model = model
        self._num_beams = num_beams
        self.tokenizer = tokenizer
        self.count=0
        self.guide_token = tokenizer.encode(guide_token, add_special_tokens=False)[0]
    
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, -float("inf"))
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                if self.count == 0:
                    hash_key = [self.guide_token]
                else:
                    hash_key=sent[-self.count:].tolist()
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key)

                if len(prefix_allowed_tokens) == 0:
                    continue
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0
                
        self.count += 1

    
        return scores + mask
