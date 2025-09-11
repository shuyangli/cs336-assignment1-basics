import typer
from typing_extensions import Annotated
import numpy as np
import numpy.typing as npt
import os
import matplotlib.pyplot as plt
import torch
import time

import wandb

from cs336_basics.modules.transformer_lm import TransformerLM
from cs336_basics.optimizer.adamw import AdamW
from cs336_basics.optimizer.gradient_clipping import clip_gradient
from cs336_basics.functional.cross_entropy import cross_entropy_loss
from cs336_basics.training.checkpointing import save_checkpoint
from cs336_basics.training.get_batch import get_batch

def main(
    train_dataset_path: Annotated[str, typer.Option("--train-dataset", help="Path to the training dataset")] = "data/TinystoriesV2-train.npy",
    val_dataset_path: Annotated[str, typer.Option("--val-dataset", help="Path to the validation dataset")] = "data/TinystoriesV2-valid.npy",

    # Model hyperparameters
    context_length: Annotated[int, typer.Option("--context-length", "-c", help="Context length for the model")] = 128,
    vocab_size: Annotated[int, typer.Option("--vocab-size", "-v", help="Vocabulary size")] = 10000,
    num_layers: Annotated[int, typer.Option("--num-layers", "-l", help="Number of layers in the model")] = 2,
    d_model: Annotated[int, typer.Option("--d-model", "-d", help="Dimensionality of model embeddings")] = 128,
    num_heads: Annotated[int, typer.Option("--num-heads", "-a", help="Number of attention heads")] = 4,
    d_ff: Annotated[int, typer.Option("--d-ff", "-f", help="Dimensionality of feedforward network")] = 512,
    rope_theta: Annotated[float | None, typer.Option("--rope-theta", "-t", help="Theta parameter for RoPE")] = None,

    # Optimization hyperparameters
    learning_rate: Annotated[float, typer.Option("--learning-rate", "-r", help="Learning rate for optimizer")] = 1e-3,
    beta1: Annotated[float, typer.Option("--beta1", help="Beta1 for AdamW optimizer")] = 0.9,
    beta2: Annotated[float, typer.Option("--beta2", help="Beta2 for AdamW optimizer")] = 0.999,
    weight_decay: Annotated[float, typer.Option("--weight-decay", "-w", help="Weight decay for AdamW optimizer")] = 0.01,
    epsilon: Annotated[float, typer.Option("--epsilon", help="Epsilon for AdamW optimizer")] = 1e-8,

    max_l2_norm: Annotated[float | None, typer.Option("--max-l2-norm", help="Max L2 norm for gradient clipping")] = None,

    min_learning_rate: Annotated[float | None, typer.Option("--min-learning-rate", help="Minimum learning rate after cosine annealing")] = None,
    num_warmup_iterations: Annotated[int | None, typer.Option("--num-warmup-iterations", help="Number of warmup iterations for learning rate schedule")] = None,
    cosine_annealing_iterations: Annotated[int | None, typer.Option("--cosine-annealing-iterations", help="Number of iterations for cosine annealing")] = None,

    # Training hyperparameters
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", help="Batch size for training")] = 32,
    epochs: Annotated[int, typer.Option("--epochs", "-e", help="Number of training epochs")] = 100,
    device: Annotated[str, typer.Option("--device", help="Device to use for training (e.g., 'cpu', 'cuda')")] = "cpu",
    save_every: Annotated[int, typer.Option("--save-every", help="Save model every N epochs")] = 20,
    save_path: Annotated[str, typer.Option("--save-path", help="Path to save the model")] = "checkpoints",
    validate_every: Annotated[int, typer.Option("--validate-every", help="Save model every N epochs")] = 100,
):
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Weights and Biases logging
    run = wandb.init(
        entity="shuyangli-personal",
        project="cs336-basics-assignment1",
        config={
            "context_length": context_length,
            "vocab_size": vocab_size,
            "num_layers": num_layers,
            "d_model": d_model,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "rope_theta": rope_theta,

            "learning_rate": learning_rate,
            "beta1": beta1,
            "beta2": beta2,
            "weight_decay": weight_decay,
            "epsilon": epsilon,
            "max_l2_norm": max_l2_norm,

            "min_learning_rate": min_learning_rate,
            "num_warmup_iterations": num_warmup_iterations,
            "cosine_annealing_iterations": cosine_annealing_iterations,

            "batch_size": batch_size,
            "epochs": epochs,
        }
    )

    # Initialize a new model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=rope_theta,
        device=torch.device(device),
    )

    # Initialize the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay, eps=epsilon)

    # Load data corpus
    train_set: npt.NDArray = np.load(train_dataset_path, mmap_mode="r")
    train_set = train_set.astype(np.int64)

    val_set: npt.NDArray = np.load(val_dataset_path, mmap_mode="r")
    val_set = val_set.astype(np.int64)

    loss_iters = []
    losses = []

    validation_iters = []
    val_losses = []

    # Actual training loop!
    print("Training started...")
    overall_start_time = time.perf_counter()
    start_time = time.perf_counter()

    for iteration in range(1, epochs + 1):
        optimizer.zero_grad()

        # Forward pass
        xs, ys = get_batch(dataset=train_set, batch_size=batch_size, context_size=context_length, device=device)
        logits = model(xs)

        loss = cross_entropy_loss(logits[:, -1, :], ys[:, -1])

        # Backward pass
        loss.backward()

        # Apply gradient clipping before optimizer step
        if max_l2_norm is not None:
            clip_gradient(model.parameters(), max_l2_norm)

        # TODO: Update learning rate based on schedule

        optimizer.step()

        stop_time = time.perf_counter()
        step_time = stop_time - start_time
        print(f"Step time: {step_time}")
        start_time = stop_time

        run.log({"train-loss": loss.item(), "step_time": step_time}, step=iteration)

        # Save checkpoint periodically
        if iteration % save_every == 0:
            checkpoint_path = f"{save_path}/epoch-{iteration}.pt"
            save_checkpoint(model, optimizer, iteration, checkpoint_path)

        # Also log validation periodically
        if iteration % validate_every == 0:
            print(f"Epoch {iteration} / {epochs}: training loss: {loss.item():.4f}")

            loss_iters.append(iteration)
            losses.append(loss.item())

            val_xs, val_ys = get_batch(dataset=val_set, batch_size=batch_size, context_size=context_length, device=device)
            val_logits = model(val_xs)
            val_loss = cross_entropy_loss(val_logits[:, -1, :], val_ys[:, -1])

            validation_iters.append(iteration)
            val_losses.append(val_loss.item())
            print(f"validation loss: {val_loss.item():.4f}")

            run.log({"val-loss": val_loss.item()}, step=iteration)


    overall_end_time = time.perf_counter()
    print("Overall training time: ", overall_end_time - overall_start_time)

    run.finish()

    # Print loss chart
    plt.figure(figsize=(8, 5))
    plt.plot(loss_iters, losses, label=f"Training loss")
    plt.plot(loss_iters, val_losses, label=f"Validation loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Cross-entropy loss over training")
    plt.legend()
    plt.grid(True)
    plt.show()


# uv run ./cs336_basics/training/training_loop.py --train-dataset data/TinystoriesV2-train.npy --val-dataset data/TinystoriesV2-valid.npy --num-layers 2 --d-model 128 --num-heads 4 --d-ff 512 --learning-rate 0.001 --weight-decay 0.01 --batch-size 32 --epochs 1000 --device cpu --save-every 100 --save-path ./checkpoints
if __name__ == "__main__":
    typer.run(main)