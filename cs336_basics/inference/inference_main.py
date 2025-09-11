import typer
from typing_extensions import Annotated
import torch
from torch import Tensor

from cs336_basics.modules.transformer_lm import TransformerLM
from cs336_basics.tokenizer.bpe import BpeTokenizer
from cs336_basics.functional.softmax import softmax
from cs336_basics.training.checkpointing import load_checkpoint

def sample_from_distribution(
    model: torch.nn.Module,
    input_tensor: Tensor,
    temperature: float = 1.0,
    eps: float = 1e-6,
    top_p: float | None = None,
    num_samples: int = 1,
) -> Tensor:
    # Returns the index of token sampled from the distribution.
    output = model(input_tensor)
    output_logits = output[-1, :]  # Get logits for the last token
    output_probs = softmax(output_logits / (temperature + eps), -1)

    if top_p is not None:
        # Manipulate the output_probs to only keep the top_p cumulative probability mass
        sorted_probs, sorted_indices = torch.sort(output_probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        output_probs[indices_to_remove] = 0.0

    output_sample = torch.multinomial(output_probs, num_samples, replacement=True)
    return output_sample

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
    temperature: Annotated[float, typer.Option("--temperature", help="Sampling temperature")] = 1.0,
    top_p: Annotated[float | None, typer.Option("--top-p", help="Top-p sampling threshold")] = None,
    max_length: Annotated[int, typer.Option("--max-length", help="Maximum generation length")] = 100,
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
        device=torch.device(device),
    )

    # Load the checkpoint
    load_checkpoint(checkpoint_path, model, optimizer=None)

    # Load tokenizer
    special_tokens_list = [token.strip() for token in special_tokens.split(",")]
    tokenizer = BpeTokenizer.from_files(vocab_path, merges_path, special_tokens_list)

    print("Model loaded successfully.")

    while True:
        prompt = input("\nEnter a prompt (Ctrl+C to quit): ")

        # Tokenize the prompt
        input_tokens = tokenizer.encode(prompt)
        print("Input tokens: ", input_tokens)

        input_tensor = torch.tensor(input_tokens, dtype=torch.int64, device=device)

        num_generations = 0

        while num_generations < max_length:
            num_generations += 1

            output_sample = sample_from_distribution(
                model,
                input_tensor,
                temperature=temperature,
                eps=1e-6,
                top_p=top_p,
                num_samples=1,
            )
            output_token = tokenizer.decode(output_sample.tolist())
            print(output_token, end="", flush=True)
            if output_token == "<|endoftext|>":
                break

            # Append the sampled token to the input tensor for the next iteration
            input_tensor = torch.cat([input_tensor, output_sample], dim=0)

        if num_generations >= max_length:
            print("\n[Reached max generation length]")


# uv run ./cs336_basics/inference/inference_main.py --checkpoint checkpoints/epoch-10000.pt --num-layers 10 --d-model 128 --num-heads 8 --d-ff 512 --device cpu --temperature 1.0 --max-length 100
if __name__ == "__main__":
    typer.run(main)