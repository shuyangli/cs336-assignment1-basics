import torch
import matplotlib.pyplot as plt
from collections import defaultdict

from cs336_basics.optimizer.lr_cosine_schedule import get_cosine_annealing_learning_rate_schedule


if __name__ == "__main__":
    MAX_LR = 1e-1
    MIN_LR = 1e-4
    NUM_WARMUP_ITERATIONS = 10
    NUM_ANNEALING_ITERATIONS = 300

    TOTAL_ITERATIONS = 350

    t_list = list(range(TOTAL_ITERATIONS))
    lr_list = [
        get_cosine_annealing_learning_rate_schedule(t, MAX_LR, MIN_LR, NUM_WARMUP_ITERATIONS, NUM_ANNEALING_ITERATIONS)
        for t in t_list
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(t_list, lr_list, label=f"Learning rate schedule")

    plt.xlabel("Step")
    plt.ylabel("Learng Rate")
    plt.title("Cosine annealing LR schedule")
    plt.legend()
    plt.grid(True)
    plt.show()