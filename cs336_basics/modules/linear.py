import torch
import einops


class Linear(torch.nn.Module):
    # Does NOT include a bias term, following most modern LLMs!
    # y = x @ W^T
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # construct and store weights as W (not W^T) for memory ordering reasons
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(x, self.weight, "... in, out in -> ... out")