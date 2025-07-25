import torch
from torch import Tensor

from cs336_basics.modules.linear import Linear

class SwiGLU(torch.nn.Module):
    def __init__(self,
                 d_model: int,  # Both input and output dimensions
                 d_hidden: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        # Canonically d_hidden = 8/3 * d_model
        self.linear1 = Linear(d_model, d_hidden, device=device, dtype=dtype)
        self.linear2 = Linear(d_hidden, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_hidden, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x_silu = self.linear1(x)
        x_silu = x_silu * torch.sigmoid(x_silu)

        x_linear = self.linear3(x)
        x_elementwise = x_silu * x_linear

        x_out = self.linear2(x_elementwise)
        return x_out