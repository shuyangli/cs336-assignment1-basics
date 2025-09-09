import typer
import time
import numpy as np
from typing_extensions import Annotated

from cs336_basics.tokenizer.bpe import BpeTokenizer

def tokenize_corpus(
    corpus_path: Annotated[str, typer.Option("--corpus-path", "-c", help="Path to the training corpus text file")] = "data/TinyStoriesV2-GPT4-valid.txt",
    output_path: Annotated[str, typer.Option("--output-path", "-o", help="Path to save the tokenized corpus")] = "tokenized_corpus.npy",
    special_tokens: Annotated[str, typer.Option("--special-tokens", "-s", help="Comma-separated list of special tokens")] = "<|endoftext|>",
    vocab_path: Annotated[str, typer.Option("--vocab-path", "-v", help="Path for the trained tokenizer vocabulary")] = "data/tinystories-tokenizer.json",
    merges_path: Annotated[str, typer.Option("--merges-path", "-m", help="Path for the trained tokenizer merges")] = "data/tinystories-merges.txt",
):
    special_tokens_list = [token.strip() for token in special_tokens.split(",")]
    print(f"Starting tokenizing {corpus_path}")

    tokenizer = BpeTokenizer.from_files(vocab_path, merges_path, special_tokens_list)

    # Read file as iterable
    print("Tokenizing corpus...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        tokens = tokenizer.encode_iterable(f)
        print(f"Finished reading iterable")
        token_array = np.fromiter(tokens, dtype=np.uint16)
        print(f"Total tokens: {len(token_array)}")

    np.save(output_path, token_array)
    print(f"Tokenized corpus saved to {output_path}")


if __name__ == "__main__":
    start_time = time.perf_counter()
    print(f"Start time: {start_time:.4f} seconds")

    typer.run(tokenize_corpus)

    end_time = time.perf_counter()
    print(f"End time: {end_time:.4f} seconds")
    print(f"Total training time: {end_time - start_time:.4f} seconds")