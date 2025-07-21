import regex as re
import os

from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Sequence

from .pretokenization_example import find_chunk_boundaries


# GPT-2 pre-tokenization pattern
PRETOKENIZATION_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

@dataclass
class Vocabulary:
    token_to_bytes: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    vocab_size: int

    def __init__(self, vocab_size: int, special_tokens: Sequence[str] = []):
        self.token_to_bytes = {}
        self.merges = []
        self.vocab_size = vocab_size

        assert vocab_size >= 256 + len(special_tokens)

        for i in range(256):
            self._add_token(bytes([i]))

        # TODO: this isn't correct, add after merging
        for special_token in special_tokens:
            self._add_token(special_token.encode("utf-8"))

    def _add_token(self, token: bytes) -> int:
        token_idx = len(self.token_to_bytes)
        self.token_to_bytes[token_idx] = token
        return token_idx

    def decode_token(self, token: int) -> bytes:
        return self.token_to_bytes.get(token, b"<UNK>")

    def merge_tokens(self, tokens: Sequence[bytes]) -> int:
        assert len(tokens) == 2
        new_bytes_tuple = (tokens[0], tokens[1])
        self.merges.append(new_bytes_tuple)

        new_bytes = tokens[0] + tokens[1]
        return self._add_token(new_bytes)


# This is a hot spot:
def compute_next_merge(tokens: dict[Sequence[bytes], int]) -> tuple[bytes, bytes]:
    # Compute token statistics
    token_stats: dict[tuple[bytes, bytes], int] = defaultdict(lambda: 0)

    for word, count in tokens.items():
        for i in range(len(word) - 1):
            token_stats[(word[i], word[i + 1])] += count

    # Merge tokens
    stats_by_count: list[tuple[int, tuple[bytes, bytes]]] = [(v, k) for k, v in token_stats.items()]
    stats_by_count.sort(reverse=True)
    return stats_by_count[0][1]

def train_bpe_with_text(corpus: str, vocab_size: int, special_tokens: list[str]):
    vocab = Vocabulary(vocab_size=vocab_size, special_tokens=special_tokens)

    # First split the corpus with special tokens
    split_special_token_pattern = "|".join([re.escape(token) for token in special_tokens])

    # We only need to pretokenize each word once, then multiply by count
    pretokenized_words_to_count: dict[Sequence[bytes], int] = defaultdict(int)

    for corpus_chunk in re.split(split_special_token_pattern, corpus):
        # Pretokenize
        for word in re.finditer(PRETOKENIZATION_PATTERN, corpus_chunk):
            # TODO: why is this so annoying to use?
            word_key = tuple([bytes([c]) for c in word.group(0).encode("utf-8")])
            pretokenized_words_to_count[word_key] += 1

    # BPE training loop
    while len(vocab.token_to_bytes) < vocab_size:
        next_merge = compute_next_merge(pretokenized_words_to_count)
        vocab.merge_tokens(next_merge)

        new_pretokenized_words_to_count = {}
        for word, count in pretokenized_words_to_count.items():
            new_word = []
            i = 0

            # TODO this is wrong: we are not separating the tokens correctly
            while i <= len(word) - 1:
                if i == len(word) - 1:
                    new_word.append(word[i])
                    i += 1
                elif (word[i], word[i + 1]) == next_merge:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_pretokenized_words_to_count[tuple(new_word)] = count

        pretokenized_words_to_count = new_pretokenized_words_to_count
        del new_pretokenized_words_to_count
    return vocab

def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], num_processes: int = 4) -> Vocabulary:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token



    with open(input_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    return train_bpe_with_text(corpus, vocab_size, special_tokens)


class BytePairTokenizer:
    vocab: Vocabulary

    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

    def encode(self, text: str) -> list[int]:
        text_bytes = text.encode("utf-8")
        output_tokens = []
        # TODO
        raise NotImplementedError

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.vocab.decode_token(t).decode("utf-8") for t in tokens)
