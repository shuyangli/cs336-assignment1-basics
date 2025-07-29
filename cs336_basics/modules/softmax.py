import torch
from torch import Tensor

def softmax(x: Tensor, dim: int) -> Tensor:
    x_normalized = x - torch.max(x, dim=dim, keepdim=True).values
    return x_normalized.exp() / torch.sum(x_normalized.exp(), dim=dim, keepdim=True)