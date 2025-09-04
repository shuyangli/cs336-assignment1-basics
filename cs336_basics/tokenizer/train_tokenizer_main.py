import typer
import time
from typing_extensions import Annotated

from cs336_basics.tokenizer.bpe_training import train_bpe

def train_tokenizer(
    corpus_path: Annotated[str, typer.Option("--corpus-path", "-c", help="Path to the training corpus text file")] = "data/TinyStoriesV2-GPT4-valid.txt",
    vocab_size: Annotated[int, typer.Option("--vocab-size", "-v", help="Vocabulary size for the tokenizer")] = 10000,
    special_tokens: Annotated[str, typer.Option("--special-tokens", "-s", help="Comma-separated list of special tokens")] = "<|endoftext|>",
    output_path: Annotated[str, typer.Option("--output-path", "-o", help="Path to save the trained tokenizer vocabulary")] = "tokenizer.json",
    merges_path: Annotated[str, typer.Option("--merges-path", "-m", help="Path to save the trained tokenizer merges")] = "merges.txt",
    num_processes: Annotated[int, typer.Option("--num_processes", "-p", help="Number of processes to use for parallelism")] = 8,
):
    special_tokens_list = [token.strip() for token in special_tokens.split(",")]
    tokenizer = train_bpe(corpus_path, vocab_size, special_tokens_list, num_processes)

    tokenizer.save_to_file(output_path, merges_path)
    print(f"Tokenizer saved to {output_path=} and {merges_path=}")

if __name__ == "__main__":
    print("Starting tokenizer training...")
    start_time = time.perf_counter()
    print(f"Start time: {start_time:.4f} seconds")

    typer.run(train_tokenizer)

    end_time = time.perf_counter()
    print(f"End time: {end_time:.4f} seconds")
    print(f"Total training time: {end_time - start_time:.4f} seconds")