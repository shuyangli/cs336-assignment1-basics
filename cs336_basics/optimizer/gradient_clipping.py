from collections.abc import Iterable
import torch

def clip_gradient(
    params: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
):
    all_grads = torch.cat([p.grad for p in params if p.grad is not None])
    l2_norm = (all_grads ** 2).sum().sqrt()
    if l2_norm <= max_l2_norm:
        return

    clip = max_l2_norm / (l2_norm + eps)
    for p in params:
        if p.grad is not None:
            p.grad *= clip