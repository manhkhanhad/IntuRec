import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from transformers import Qwen2ForCausalLM

class IntuRecF1(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)