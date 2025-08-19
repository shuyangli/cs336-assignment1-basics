from collections.abc import Callable, Iterable
import torch
from torch import Tensor
from jaxtyping import Float
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = { "lr": lr }
        super().__init__(params, defaults=defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float:
        if closure:
            loss = closure()
        else:
            loss = None

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                print(p)
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.

        return loss