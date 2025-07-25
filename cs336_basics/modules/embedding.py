import torch
from torch import Tensor
from jaxtyping import Int, Float


class Embedding(torch.nn.Module):
    def __init__(self,
                 num_embeddings: int,   # Vocabulary size
                 embedding_dim: int,    # d_model
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()

        init_tensor: Float[Tensor, " vocab_size d_model"] = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(init_tensor, mean=0, std=1, a=-3, b=3)
        self.embedding = torch.nn.Parameter(init_tensor)


    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.embedding[token_ids]