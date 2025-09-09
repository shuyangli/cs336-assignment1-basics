import typer
from typing_extensions import Annotated
import numpy as np
import torch
from torch import Tensor

# import wandb

from cs336_basics.modules.transformer_lm import TransformerLM
from cs336_basics.tokenizer.bpe import BpeTokenizer
from cs336_basics.functional.softmax import softmax
from cs336_basics.training.checkpointing import load_checkpoint

def main(
    checkpoint_path: Annotated[str, typer.Option("--checkpoint", help="Path to the checkpoint")],

    # Model hyperparameters
    context_length: Annotated[int, typer.Option("--context-length", "-c", help="Context length for the model")] = 128,
    vocab_size: Annotated[int, typer.Option("--vocab-size", "-v", help="Vocabulary size")] = 10000,
    num_layers: Annotated[int, typer.Option("--num-layers", "-l", help="Number of layers in the model")] = 2,
    d_model: Annotated[int, typer.Option("--d-model", "-d", help="Dimensionality of model embeddings")] = 128,
    num_heads: Annotated[int, typer.Option("--num-heads", "-a", help="Number of attention heads")] = 4,
    d_ff: Annotated[int, typer.Option("--d-ff", "-f", help="Dimensionality of feedforward network")] = 512,
    rope_theta: Annotated[float | None, typer.Option("--rope-theta", "-t", help="Theta parameter for RoPE")] = None,
    device: Annotated[str, typer.Option("--device", help="Device to use (e.g., 'cpu', 'cuda')")] = "cpu",

    # Tokenizer
    special_tokens: Annotated[str, typer.Option("--special-tokens", "-s", help="Comma-separated list of special tokens")] = "<|endoftext|>",
    vocab_path: Annotated[str, typer.Option("--vocab-path", "-v", help="Path for the trained tokenizer vocabulary")] = "data/tinystories-tokenizer.json",
    merges_path: Annotated[str, typer.Option("--merges-path", "-m", help="Path for the trained tokenizer merges")] = "data/tinystories-merges.txt",

    # Sampling
    temperature: Annotated[float, typer.Option("--temperature", "-r", help="Sampling temperature")] = 1.0,
    max_length: Annotated[int, typer.Option("--max-length", "-b", help="Maximum generation length")] = 100,
):
    # Initialize a model with the same architecture
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=rope_theta,
    )
    model.to(device)

    # Load the checkpoint
    load_checkpoint(checkpoint_path, model, optimizer=None)

    # Load tokenizer
    special_tokens_list = [token.strip() for token in special_tokens.split(",")]
    tokenizer = BpeTokenizer.from_files(vocab_path, merges_path, special_tokens_list)

    print("Model loaded successfully.")

    # For sampling
    eps = 1e-6

    while True:
        prompt = input("\nEnter a prompt (Ctrl+C to quit): ")

        # Tokenize the prompt
        input_tokens = tokenizer.encode(prompt)
        print("Input tokens: ", input_tokens)

        input_tensor = torch.tensor(input_tokens, dtype=torch.int64)

        num_generations = 0

        while num_generations < max_length:
            num_generations += 1

            output = model(input_tensor)
            output_logits = output[-1, :]  # Get logits for the last token
            output_probs = softmax(output_logits / (temperature + eps), -1)

            output_sample = torch.multinomial(output_probs, 1)
            output_token = tokenizer.decode(output_sample.tolist())

            print(output_token, end="", flush=True)
            if output_token == "<|endoftext|>":
                break
        if num_generations >= max_length:
            print("\n[Reached max generation length]")


# uv run ./cs336_basics/inference/inference_main.py --checkpoint checkpoints/epoch-1000.pt --num-layers 2 --d-model 128 --num-heads 4 --d-ff 512 --device cpu --temperature 1.0 --max-length 100
if __name__ == "__main__":
    typer.run(main)