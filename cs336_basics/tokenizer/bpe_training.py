import regex as re
import heapq
import os
from multiprocessing import Pool

from collections.abc import Sequence
from collections import defaultdict

from cs336_basics.tokenizer.bpe import Vocabulary, BpeTokenizer
from cs336_basics.tokenizer.pretokenization_example import find_chunk_boundaries


# This is a hot spot:
def compute_next_merge(tokens: dict[Sequence[bytes], int]) -> tuple[bytes, bytes]:
    # Compute token statistics
    token_stats: dict[tuple[bytes, bytes], int] = defaultdict(int)

    # Still room to improve here, since across two runs only one pair changes.
    for word, count in tokens.items():
        for i in range(len(word) - 1):
            token_stats[(word[i], word[i + 1])] += count

    # Merge tokens
    highest_count = -1
    highest_tokens: tuple[bytes, bytes] = (b"", b"")

    for pair, count in token_stats.items():
        if count > highest_count or count == highest_count and highest_tokens < pair:
            highest_count = count
            highest_tokens = pair
    return highest_tokens


def update_words(pretokenized_words_to_count: dict[Sequence[bytes], int], merge: tuple[bytes, bytes]) -> tuple[dict[Sequence[bytes], int], list[tuple[Sequence[bytes], Sequence[bytes], int]]]:
    """Apply a merge to words and return updated map and list of changes.

    Returns:
        - new_pretokenized_words_to_count: updated mapping word->count
        - changes: list of (old_word, new_word, count) for words that changed
    """
    new_pretokenized_words_to_count: dict[Sequence[bytes], int] = {}
    changes: list[tuple[Sequence[bytes], Sequence[bytes], int]] = []

    for word, count in pretokenized_words_to_count.items():
        new_word: list[bytes] = []
        i = 0

        # If either component isn't present, the word cannot change
        if merge[0] not in word or merge[1] not in word:
            new_pretokenized_words_to_count[word] = new_pretokenized_words_to_count.get(word, 0) + count
            continue

        word_len = len(word)
        while i <= word_len - 1:
            if i == word_len - 1:
                new_word.append(word[i])
                i += 1
            elif (word[i], word[i + 1]) == merge:
                new_word.append(word[i] + word[i + 1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        new_word_tuple = tuple(new_word)

        # Aggregate counts for identical new words
        new_pretokenized_words_to_count[new_word_tuple] = new_pretokenized_words_to_count.get(new_word_tuple, 0) + count

        # Record change only if the word actually changed
        if new_word_tuple != word:
            changes.append((word, new_word_tuple, count))

    return new_pretokenized_words_to_count, changes


def _worker_pretokenize_and_count(
    args: tuple[str | os.PathLike, int, int, list[str]]
) -> tuple[dict[Sequence[bytes], int], dict[tuple[bytes, bytes], int]]:
    """Worker function for pre-tokenization and counting."""
    input_path, start, end, special_tokens = args
    pretokenized_words_to_count: dict[Sequence[bytes], int] = defaultdict(int)
    token_stats: dict[tuple[bytes, bytes], int] = defaultdict(int)

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    for word in BpeTokenizer.corpus_to_pretokenized_words(chunk, special_tokens):
        word_key = tuple([bytes([c]) for c in word])
        pretokenized_words_to_count[word_key] += 1
        for i in range(len(word_key) - 1):
            token_stats[(word_key[i], word_key[i + 1])] += 1
    return pretokenized_words_to_count, token_stats


def train_bpe_from_pretokenized(
    pretokenized_words_to_count: dict[Sequence[bytes], int],
    token_stats: dict[tuple[bytes, bytes], int],
    vocab_size: int,
    special_tokens: list[str],
) -> Vocabulary:
    """Trains BPE from pre-tokenized and counted words."""
    assert vocab_size >= 256 + len(special_tokens)

    vocab = Vocabulary(vocab_size=vocab_size)
    vocab.init_for_training()

    target_vocab_size = vocab_size - len(special_tokens)

    # Build a max-heap for token pairs
    heap = [(count, pair) for pair, count in token_stats.items()]
    heapq._heapify_max(heap)

    total_iterations = target_vocab_size - len(vocab.token_to_bytes)
    it = 0

    while len(vocab.token_to_bytes) < target_vocab_size and heap:
        it += 1
        if it % 100 == 0 or it == 1:
            print(f"Iteration {it} / {total_iterations}")
        # Get the most frequent pair
        _, next_merge = heapq.heappop(heap)
        vocab.merge_tokens(next_merge)
        pretokenized_words_to_count, changes = update_words(pretokenized_words_to_count, next_merge)

        # Update token_stats only for affected pairs
        for old_word, new_word, count in changes:
            # Decrement counts for pairs in the old word
            for i in range(len(old_word) - 1):
                pair = (old_word[i], old_word[i + 1])
                token_stats[pair] -= count
                if token_stats[pair] == 0:
                    # Keep dict tidy by removing zeros
                    del token_stats[pair]

            # Increment counts for pairs in the new word
            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                token_stats[pair] += count

        # Rebuild the heap from updated token_stats
        heap = [(cnt, pair) for pair, cnt in token_stats.items()]
        heapq._heapify_max(heap)

    vocab.add_special_tokens(special_tokens)

    return vocab


def train_bpe_with_text(corpus: str, vocab_size: int, special_tokens: list[str]):
    # We only need to pretokenize each word once, then multiply by count
    pretokenized_words_to_count: dict[Sequence[bytes], int] = defaultdict(int)
    token_stats: dict[tuple[bytes, bytes], int] = defaultdict(int)

    for word in BpeTokenizer.corpus_to_pretokenized_words(corpus, special_tokens):
        word_key = tuple([bytes([c]) for c in word])
        pretokenized_words_to_count[word_key] += 1
        for i in range(len(word_key) - 1):
            token_stats[(word_key[i], word_key[i + 1])] += 1

    return train_bpe_from_pretokenized(pretokenized_words_to_count, token_stats, vocab_size, special_tokens)


def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], num_processes: int = 4) -> Vocabulary:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

    boundaries_per_process = zip(boundaries[:-1], boundaries[1:])
    pool_args = [(input_path, start, end, special_tokens) for start, end in boundaries_per_process]

    with Pool(processes=num_processes) as pool:
        results = pool.map(_worker_pretokenize_and_count, pool_args)

    # Aggregate results from all processes
    aggregated_counts: dict[Sequence[bytes], int] = defaultdict(int)
    aggregated_token_stats: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for result in results:
        pretokenized_words_to_count, token_stats = result
        for word, count in pretokenized_words_to_count.items():
            aggregated_counts[word] += count
        for pair, count in token_stats.items():
            aggregated_token_stats[pair] += count

    return train_bpe_from_pretokenized(aggregated_counts, aggregated_token_stats, vocab_size, special_tokens)
