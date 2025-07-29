import torch
from torch import Tensor
from jaxtyping import Float
from einops import einsum


class RotaryPositionalEncoding(torch.nn.Module):
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_sequence_length: int,
                 device: torch.device | None = None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE requires d_k to be an even number.")
        self.theta = theta
        self.d_k = d_k
        self.max_sequence_length = max_sequence_length
        self.device = device

        frequencies = self._compute_frequencies()
        self.register_buffer('cos_cached', frequencies.cos(), persistent=False)
        self.register_buffer('sin_cached', frequencies.sin(), persistent=False)


    def _compute_frequencies(self) -> Tensor:
        angles = 1 / (self.theta ** (torch.arange(0, self.d_k, step=2).float() / self.d_k))

        # seq_len * 1 vector
        positions = torch.arange(0, self.max_sequence_length, step=1, device=self.device).float()

        # Outer product: gives us a (seq_len, d_k // 2) matrix for i * angles
        freqs = einsum(positions, angles, "seq_len, d -> seq_len d")
        return freqs


    def _apply_rotary_embedding(self,
                                x: Float[Tensor, "... seq_len d_k"],
                                cos: Float[Tensor, "seq_len d_k/2"],
                                sin: Float[Tensor, "seq_len d_k/2"],
                                ) -> Float[Tensor, "... seq_len d_k"]:
        """
        Apply rotary embedding to input tensor x.

        For each pair of dimensions (x1, x2), apply rotation:
        x1' = x1 * cos - x2 * sin
        x2' = x1 * sin + x2 * cos
        """
        # Split the input into odds and evens
        x1 = x[..., 0::2]   # (..., d_k // 2) where d_k = 0, 2, 4, ...
        x2 = x[..., 1::2]   # (..., d_k // 2) where d_k = 1, 3, 5, ...

        # Rotate
        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x1 * sin + x2 * cos

        # Restack
        return torch.stack([x1_rotated, x2_rotated], dim=-1).reshape(x.shape)


    def forward(self,
                x: Float[Tensor, "... seq_len d_k"],
                token_positions: Float[Tensor, "... seq_len"],
                ) -> Float[Tensor, "... seq_len d_k"]:
        assert isinstance(self.sin_cached, torch.Tensor)
        assert isinstance(self.cos_cached, torch.Tensor)

        token_positions_long = token_positions.long()

        sin = self.sin_cached[token_positions_long]
        cos = self.cos_cached[token_positions_long]

        return self._apply_rotary_embedding(x, cos, sin)
