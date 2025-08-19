import torch
from torch import Tensor
from jaxtyping import Float, Bool, Int
from einops import einsum, rearrange

from cs336_basics.functional.attention import attention
from cs336_basics.modules.linear import Linear
from cs336_basics.modules.rope import RotaryPositionalEmbedding

class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 theta: float | None = None,
                 d_k: int | None = None,
                 max_sequence_length: int | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        h_dk = h_dv = d_model
        self.q_proj = Linear(in_features=d_model, out_features=h_dk, device=device, dtype=dtype)
        self.k_proj = Linear(in_features=d_model, out_features=h_dk, device=device, dtype=dtype)
        self.v_proj = Linear(in_features=d_model, out_features=h_dk, device=device, dtype=dtype)
        self.output_proj = Linear(in_features=h_dv, out_features=d_model, device=device, dtype=dtype)

        if (theta is not None
            and d_k is not None
            and max_sequence_length is not None):
            self.rope = RotaryPositionalEmbedding(theta=theta,
                                                  d_k=d_k,
                                                  max_sequence_length=max_sequence_length,
                                                  device=device)
        else:
            self.rope = None

    def forward(
            self,
            in_features: Float[Tensor, " ... sequence_length d_in"],
            token_positions: Int[Tensor, " ... sequence_length"] | None = None,
        ) -> Float[Tensor, "... sequence_length d_out"]:
        # input q_proj is (64, 64); model_size is 64 and num_heads is 4, so each head's dimension is (16, 64)
        *_, seq_len, _ = in_features.shape

        query:  Float[Tensor, "... seq_len hd_k"] = self.q_proj(in_features)
        key:    Float[Tensor, "... seq_len hd_k"] = self.k_proj(in_features)
        value:  Float[Tensor, "... seq_len hd_v"] = self.v_proj(in_features)

        query = rearrange(query, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        key =   rearrange(key,   "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        value = rearrange(value, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)

        if self.rope:
            if token_positions is None:
                # Generate a token positions tensor if not provided
                token_positions = torch.arange(seq_len, device=in_features.device, dtype=torch.int64)

            # Apply RoPE to query and key
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)

        # Mask is over the sequence length dimension
        # mask = torch.tril(torch.ones((seq_len, seq_len), device=in_features.device, dtype=torch.bool))

        # Equivalent mask computation with broadcasted index comparison
        indices = torch.arange(seq_len, device=in_features.device)
        mask: Bool[Tensor, "... seq_len seq_len"] = (indices <= indices.unsqueeze(-1))

        attention_output: Float[Tensor, "... num_heads seq_len d_v"] = attention(
            query, key, value, mask=mask
        )

        # Reshape back
        attention_output = rearrange(attention_output, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)",
                                     num_heads=self.num_heads)
        return self.output_proj(attention_output)