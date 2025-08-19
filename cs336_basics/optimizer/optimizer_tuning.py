import torch
import matplotlib.pyplot as plt
from collections import defaultdict

from cs336_basics.optimizer.sgd import SGD


if __name__ == "__main__":
    NUM_STEPS = 10
    weights_per_lr = {}
    optimizers = {}

    for learning_rate in [1e0, 1e1, 1e2]: #, 1e3]:
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        weights_per_lr[learning_rate] = weights
        optimizers[learning_rate] = SGD([weights], lr=learning_rate)

    steps = list(range(NUM_STEPS))
    losses = defaultdict(lambda: list())

    for t in range(NUM_STEPS):
        for lr, opt in optimizers.items():
            opt.zero_grad() # Reset the gradients for all learnable parameters.
            weights = weights_per_lr[lr]
            loss = (weights**2).mean() # Compute a scalar loss value.
            losses[lr].append(loss.cpu().item())
            loss.backward() # Run backward pass, which computes gradients.
            opt.step() # Run optimizer step.

    plt.figure(figsize=(8, 5))

    for learning_rate, loss_series in losses.items():
        plt.plot(steps, loss_series, label=f"Learning rate = {learning_rate}")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss by step")
    plt.legend()
    plt.grid(True)
    plt.show()