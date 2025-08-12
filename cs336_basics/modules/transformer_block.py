import torch
from torch import Tensor
from jaxtyping import Float
from einops import einsum

from cs336_basics.modules.multihead_attention import MultiHeadSelfAttention
from cs336_basics.modules.rmsnorm import RMSNorm
from cs336_basics.modules.swiglu import SwiGLU

class TransformerBlock(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int | None = None,
                 theta: float | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        d_k = d_model // num_heads

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            d_k=d_k,
            max_sequence_length=max_seq_len,
            device=device,
            dtype=dtype)
        # Pre-Norm RMSNorm
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_hidden=d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(self,
                in_features: Float[Tensor, " ... sequence_length d_in"]) -> Float[Tensor, "... sequence_length d_out"]:
        # First half: Pre-norm + attention
        out_features = self.ln1(in_features)
        out_features = self.attn(out_features)
        out_features = out_features + in_features

        # Second half: Pre-norm + feed-forward
        intermediate_features = out_features
        out_features = self.ln2(out_features)
        out_features = self.ffn(out_features)
        out_features = out_features + intermediate_features

        return out_features