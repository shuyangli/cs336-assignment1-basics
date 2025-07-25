import torch
from torch import Tensor
from jaxtyping import Float
from einops import einsum

class RMSNorm(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))


    def forward(self, x: Float[Tensor, "... d_model"]) -> Tensor:
        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)

        # RMSNorm(a_i) = a_i * RMS(a) * gain_i
        # where RMS(a) = sqrt(sum(a_i^2) / d_model + eps)
        result = torch.sum(x ** 2, dim=-1, keepdim=True) / self.d_model + self.eps
        result = torch.rsqrt(result)
        result = x * result * self.gain

        return result.to(dtype=in_dtype)