import torch
from torch import Tensor
from jaxtyping import Float, Bool
from einops import einsum

from cs336_basics.modules.softmax import softmax

class ScaledDotProductAttention(torch.nn.Module):
    """
    Scaled Dot-Product Attention module.
    """

    def __init__(self):
        super().__init__()

    def forward(self,
                query: Float[Tensor, "b ... queries d_k"],
                key: Float[Tensor, "b ... keys d_k"],
                value: Float[Tensor, "b ... keys d_v"],
                mask: Float[Tensor, " ... queries keys"] | None = None,
            ) -> Float[Tensor, "b ... queries d_v"]:
        *_, d_k = query.shape
        multipled = einsum(query, key, "... queries d_k, ... keys d_k -> ... queries keys") / d_k**0.5
        # Apply mask
        if mask is not None:
            multipled = multipled.masked_fill(mask == 0, float('-inf'))

        multipled = softmax(multipled, dim=-1)
        multipled = einsum(multipled, value, "... queries keys, ... keys d_v -> ... queries d_v")

        return multipled