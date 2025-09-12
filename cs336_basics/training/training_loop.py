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
from cs336_basics.optimizer.lr_cosine_schedule import get_cosine_annealing_learning_rate_schedule
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

    max_l2_norm: Annotated[float | None, typer.Option("--max-l2-norm", help="Max L2 norm for gradient clipping")] = None,

    min_learning_rate: Annotated[float | None, typer.Option("--min-learning-rate", help="Minimum learning rate after cosine annealing")] = None,
    num_warmup_iterations: Annotated[int | None, typer.Option("--num-warmup-iterations", help="Number of warmup iterations for learning rate schedule")] = None,
    cosine_annealing_iterations: Annotated[int | None, typer.Option("--cosine-annealing-iterations", help="Number of iterations for cosine annealing")] = None,

    # Training hyperparameters
    batch_size: Annotated[int, typer.Option("--batch-size", help="Batch size for training")] = 32,
    epochs: Annotated[int | None, typer.Option("--epochs", help="Number of training epochs")] = None,
    device: Annotated[str, typer.Option("--device", help="Device to use for training (e.g., 'cpu', 'cuda')")] = "cpu",
    save_every: Annotated[int, typer.Option("--save-every", help="Save model every N epochs")] = 20,
    save_path: Annotated[str, typer.Option("--save-path", help="Path to save the model")] = "checkpoints",
    total_tokens_processed: Annotated[int | None, typer.Option("--total-tokens", help="Total tokens we want to process, ignored if --epochs is provided")] = None,

    # Misc
    enable_wandb: Annotated[bool, typer.Option("--enable-wandb/--disable-wandb", help="Enable Weights and Biases logging")] = True,
):
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Compute epoch targets
    if epochs is not None:
        print(f"Training for {epochs} epochs")
    elif total_tokens_processed is not None:
        if total_tokens_processed <= 0:
            raise ValueError(f"Total tokens to process must be positive, got {total_tokens_processed}")
        epochs = total_tokens_processed // (batch_size * context_length)
        print(f"Training for {epochs} epochs to process approximately {total_tokens_processed} tokens")
    else:
        raise ValueError("Either --epochs or --total-tokens must be provided")

    if num_warmup_iterations is not None and cosine_annealing_iterations is None:
        cosine_annealing_iterations = epochs

    training_config = {
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
        "max_l2_norm": max_l2_norm,

        "min_learning_rate": min_learning_rate,
        "num_warmup_iterations": num_warmup_iterations,
        "cosine_annealing_iterations": cosine_annealing_iterations,

        "batch_size": batch_size,
        "epochs": epochs,
    }
    print("Training configuration: ", training_config)

    # Weights and Biases logging
    run = wandb.init(
        entity="shuyangli-personal",
        project="cs336-basics-assignment1",
        config=training_config,
        mode="online" if enable_wandb else "disabled",
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
    # Compiling model to speed up training
    model.compile(backend="aot_eager")

    # Initialize the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay, eps=1e-8)

    # Load data corpus
    train_set: npt.NDArray = np.load(train_dataset_path, mmap_mode="r")
    train_set = train_set.astype(np.int64)

    val_set: npt.NDArray = np.load(val_dataset_path, mmap_mode="r")
    val_set = val_set.astype(np.int64)

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

        # Cosine annealing, if applicable
        if num_warmup_iterations is not None and cosine_annealing_iterations is not None:
            scheduled_lr = get_cosine_annealing_learning_rate_schedule(iteration - 1, learning_rate, min_learning_rate if min_learning_rate is not None else 0.0, num_warmup_iterations, cosine_annealing_iterations)
            for group in optimizer.param_groups:
                group['lr'] = scheduled_lr

        optimizer.step()

        stop_time = time.perf_counter()
        step_time = stop_time - start_time
        print(f"{step_time=}")
        start_time = stop_time

        all_gradients = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
        gradient_norm = torch.linalg.norm(all_gradients)
        all_weights = torch.cat([p.data.view(-1) for p in model.parameters()])
        weight_norm = torch.linalg.norm(all_weights)


        # Save checkpoint periodically
        if iteration % save_every == 0:
            checkpoint_path = f"{save_path}/epoch-{iteration}.pt"
            save_checkpoint(model, optimizer, iteration, checkpoint_path)

        # Also log validation
        val_xs, val_ys = get_batch(dataset=val_set, batch_size=batch_size, context_size=context_length, device=device)
        val_logits = model(val_xs)
        val_loss = cross_entropy_loss(val_logits[:, -1, :], val_ys[:, -1])

        print(f"Epoch {iteration} / {epochs}: training loss: {loss.item():.4f}, validation loss: {val_loss.item():.4f}")


        run.log({
            "train-loss": loss.item(),
            "val-loss": val_loss.item(),
            "step-time": step_time,
            "gradient-l2-norm": gradient_norm,
            "weight-l2-norm": weight_norm,

            # TODO: Maybe make this more robust?
            "learning-rate": optimizer.param_groups[0]['lr'],
        },  step=iteration)

        del val_xs, val_ys, val_logits, val_loss, xs, ys, logits, loss


    overall_end_time = time.perf_counter()
    print("Overall training time: ", overall_end_time - overall_start_time)

    run.finish()

    # Save final checkpoint
    checkpoint_path = f"{save_path}/{run.name}-final.pt"
    save_checkpoint(model, optimizer, epochs, checkpoint_path)


# uv run ./cs336_basics/training/training_loop.py --train-dataset data/TinystoriesV2-train.npy --val-dataset data/TinystoriesV2-valid.npy --num-layers 2 --d-model 128 --num-heads 4 --d-ff 512 --learning-rate 0.001 --weight-decay 0.01 --batch-size 32 --epochs 1000 --device cpu --save-every 100 --save-path ./checkpoints
if __name__ == "__main__":
    typer.run(main)