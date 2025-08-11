import torch
from torch import Tensor
from jaxtyping import Float, Bool
from einops import einsum

from cs336_basics.modules.softmax import softmax

def attention(query: Float[Tensor, "b ... seq_len d_k"],
              key: Float[Tensor, "b ... seq_len d_k"],
              value: Float[Tensor, "b ... seq_len d_v"],
              mask: Bool[Tensor, " ... seq_len seq_len"] | None = None,
              ) -> Float[Tensor, "b ... seq_len d_v"]:
    *_, d_k = query.shape
    multipled = einsum(query, key, "... queries d_k, ... keys d_k -> ... queries keys") / d_k**0.5

    if mask is not None:
        multipled = multipled.masked_fill(mask == False, float('-inf'))

    multipled = softmax(multipled, dim=-1)
    multipled = einsum(multipled, value, "... queries keys, ... keys d_v -> ... queries d_v")

    return multipled