import torch
from torch import Tensor
from jaxtyping import Float, Int

from cs336_basics.modules.embedding import Embedding
from cs336_basics.modules.transformer_block import TransformerBlock
from cs336_basics.modules.rmsnorm import RMSNorm
from cs336_basics.modules.linear import Linear
from cs336_basics.modules.softmax import softmax

class TransformerLM(torch.nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 theta: float | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()

        self.embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype)

        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                device=device,
                dtype=dtype)
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)


    def forward(self,
                in_tokens: Int[Tensor, "batch_size sequence_length"]) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
        out_features: Float[Tensor, "batch_size sequence_length d_model"] = self.embedding(in_tokens)

        for layer in self.layers:
            out_features = layer(out_features)

        out_features = self.ln_final(out_features)

        token_probs: Float[Tensor, "batch_size sequence_length vocab_size"] = self.lm_head(out_features)

        # Returning token probabilities directly, not logits.
        return token_probs