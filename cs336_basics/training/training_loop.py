import typer
from typing_extensions import Annotated
import numpy as np
import numpy.typing as npt
import os
import matplotlib.pyplot as plt

# import wandb

from cs336_basics.modules.transformer_lm import TransformerLM
from cs336_basics.optimizer.adamw import AdamW
from cs336_basics.functional.cross_entropy import cross_entropy_loss
from cs336_basics.training.checkpointing import save_checkpoint
from cs336_basics.training.get_batch import get_batch

def main(
    dataset_path: Annotated[str, typer.Option("--dataset-path", help="Path to the training dataset")] = "data/TinystoriesV2-tokenized.npy",

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

    # Training hyperparameters
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", help="Batch size for training")] = 32,
    epochs: Annotated[int, typer.Option("--epochs", "-e", help="Number of training epochs")] = 100,
    device: Annotated[str, typer.Option("--device", help="Device to use for training (e.g., 'cpu', 'cuda')")] = "cpu",
    save_every: Annotated[int, typer.Option("--save-every", "-s", help="Save model every N epochs")] = 20,
    save_path: Annotated[str, typer.Option("--save-path", "-p", help="Path to save the model")] = "checkpoints",
):
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Initialize a new model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=rope_theta,
    )

    # Initialize the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay, eps=epsilon)

    # Load data corpus
    data_corpus: npt.NDArray = np.load(dataset_path, mmap_mode="r")
    # Convert this to int64 for CPU compatibility
    data_corpus = data_corpus.astype(np.int64)

    losses = []

    # Actual training loop!
    for iteration in range(1, epochs + 1):
        optimizer.zero_grad()

        # Forward pass
        xs, ys = get_batch(dataset=data_corpus, batch_size=batch_size, context_size=context_length, device=device)
        logits = model(xs)

        loss = cross_entropy_loss(logits[:, -1, :], ys[:, -1])
        losses.append(loss.item())

        # Backward pass
        loss.backward()
        optimizer.step()

        # Save checkpoint periodically
        if (iteration) % save_every == 0:
            print(f"Epoch {iteration} / {epochs}: loss: {loss.item():.4f}")
            checkpoint_path = f"{save_path}/epoch-{iteration}.pt"
            save_checkpoint(model, optimizer, iteration, checkpoint_path)

    # TODO: add W&B logging
    # training_run = wandb.init(
    #     entity="shuyangli-personal"
    # )

    # Print loss chart
    t_list = list(range(1, epochs + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(t_list, losses, label=f"Loss")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Cross-entropy loss over training")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    typer.run(main)