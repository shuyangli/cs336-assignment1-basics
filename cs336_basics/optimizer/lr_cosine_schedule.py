import math

def get_cosine_annealing_learning_rate_schedule(
    t: int,
    max_lr: float,
    min_lr: float,
    num_warmup_iterations: int,
    cosine_annealing_iterations: int,
):
    if t < num_warmup_iterations:
        # Pre-annealing
        return max_lr * (t / num_warmup_iterations)
    if t > cosine_annealing_iterations:
        # Post-annealing
        return min_lr

    return min_lr + (1 + math.cos((t - num_warmup_iterations) / (cosine_annealing_iterations - num_warmup_iterations) * math.pi)) * 0.5 * (max_lr - min_lr)