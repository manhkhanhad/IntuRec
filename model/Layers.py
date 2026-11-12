import torch
import torch.nn as nn

class VanillaRoPE(nn.Module):
    def __init__(self, hidden_size, device=None):
        super(VanillaRoPE, self).__init__()
        self.base = 1000
        theta = 1.0 / (self.base ** (torch.arange(0, hidden_size, 2, dtype=torch.int64).float().to(device) / hidden_size))
        self.register_buffer("theta", theta, persistent=False)
    
    @torch.no_grad()
    def forward(self, pos):
        # pos: [bsz, seqlen]
        theta = self.theta[None, :, None].float().expand(pos.shape[0], -1, 1).to(pos.device) # [bsz, seqlen/2, 1]
        pos = pos[:, None, :].float() # [bsz, 1, seqlen]
        freqs = (theta @ pos).permute(0, 2, 1) # [bsz, seqlen, seqlen/2]
        freqs = torch.cat([freqs, freqs], dim=-1) # [bsz, seqlen, seqlen]
        costheta = freqs.cos()
        sintheta = freqs.sin()
        return costheta, sintheta
