import torch
from torch import Tensor
from jaxtyping import Float
import math
from collections.abc import Callable, Iterable


class AdamW(torch.optim.Optimizer):
    def __init__(self, params,
                 lr: float = 1e-3,  # alpha
                 betas: tuple[float, float] = (0.9, 0.999),
                 weight_decay: float = 0.01,    # lambda
                 eps: float = 1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        if beta1 < 0.0 or beta1 >= 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if beta2 < 0.0 or beta2 >= 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "weight_decay": weight_decay,
            "eps": eps
        }
        super().__init__(params, defaults=defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float:
        if closure:
            loss = closure()
        else:
            loss = None

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                t = state.get("t", 1)
                first_moment = state.get("m", torch.zeros_like(grad))
                second_moment = state.get("v", torch.zeros_like(grad))

                first_moment = beta1 * first_moment + (1 - beta1) * grad
                second_moment = beta2 * second_moment + (1 - beta2) * grad ** 2
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                p.data -= lr_t * first_moment / (torch.sqrt(second_moment) + eps)
                p.data *= (1 - lr * weight_decay)

                # Update state
                state["t"] = t + 1
                state["m"]  = first_moment
                state["v"]  = second_moment

        return loss