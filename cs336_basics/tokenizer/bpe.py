import regex as re
import os

from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Sequence
from typing import Iterable

from .pretokenization_example import find_chunk_boundaries


# GPT-2 pre-tokenization pattern
PRETOKENIZATION_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

@dataclass
class Vocabulary:
    token_to_bytes: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    vocab_size: int

    def __init__(self,
                 vocab_size: int,
                 vocab: dict[int, bytes] | None = None,
                 merges: list[tuple[bytes, bytes]] | None = None):
        if vocab:
            self.token_to_bytes = vocab
        else:
            self.token_to_bytes = {}

        if self.token_to_bytes:
            self.bytes_to_token = {v: k for k, v in self.token_to_bytes.items()}
        else:
            self.bytes_to_token = {}

        if merges is None:
            self.merges = []
        else:
            self.merges = merges

        self.vocab_size = vocab_size

    def init_for_training(self) -> None:
        for i in range(256):
            self._add_token(bytes([i]))

    def add_special_tokens(self, special_tokens: Sequence[str]) -> None:
        for token in special_tokens:
            self._add_token(token.encode("utf-8"))

    def _add_token(self, token: bytes) -> int:
        token_idx = len(self.token_to_bytes)
        self.token_to_bytes[token_idx] = token
        self.bytes_to_token[token] = token_idx
        return token_idx

    def merge_tokens(self, tokens: Sequence[bytes]) -> int:
        new_bytes_tuple = (tokens[0], tokens[1])
        self.merges.append(new_bytes_tuple)

        new_bytes = tokens[0] + tokens[1]
        return self._add_token(new_bytes)

    def lookup_tokens(self, input_bytes: list[bytes]) -> list[int]:
        # TODO: Figure out an "UNKNOWN" token
        return [self.bytes_to_token.get(b, -1) for b in input_bytes]


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


def update_words(pretokenized_words_to_count: dict[Sequence[bytes], int], merge: tuple[bytes, bytes]) -> dict[Sequence[bytes], int]:
    new_pretokenized_words_to_count = {}
    for word, count in pretokenized_words_to_count.items():
        new_word = []
        i = 0

        if merge[0] not in word or merge[1] not in word:
            new_pretokenized_words_to_count[word] = count
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
        new_pretokenized_words_to_count[tuple(new_word)] = count
    return new_pretokenized_words_to_count


def train_bpe_with_text(corpus: str, vocab_size: int, special_tokens: list[str]):
    assert vocab_size >= 256 + len(special_tokens)

    vocab = Vocabulary(vocab_size=vocab_size)
    vocab.init_for_training()

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
    target_vocab_size = vocab_size - len(special_tokens)
    while len(vocab.token_to_bytes) < target_vocab_size:
        next_merge = compute_next_merge(pretokenized_words_to_count)
        vocab.merge_tokens(next_merge)
        pretokenized_words_to_count = update_words(pretokenized_words_to_count, next_merge)

    vocab.add_special_tokens(special_tokens)

    return vocab


class BpeTokenizer:
    vocab: Vocabulary
    special_tokens: set[str]

    def __init__(self, vocab: Vocabulary, special_tokens: list[str] | None = None):
        self.vocab = vocab

        if special_tokens:
            self.special_tokens = set(special_tokens)
        else:
            self.special_tokens = set()

    @classmethod
    def from_files(cls, vocab_path: str | os.PathLike, merges_path: str | os.PathLike, special_tokens: list[str] | None = None) -> "BpeTokenizer":
        import json

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        with open(merges_path, "r", encoding="utf-8") as f:
            merges_data = [tuple(line.strip().split(" ")) for line in f]

        vocab = Vocabulary(
            vocab_size=len(vocab_data),
            vocab={int(k): bytes(v, "utf-8") for k, v in vocab_data.items()},
            merges=[(bytes(m1, "utf-8"), bytes(m2, "utf-8")) for m1, m2 in merges_data]
        )

        return cls(vocab, special_tokens=special_tokens)

    def _apply_merges(self, word: str, merges: list[tuple[bytes, bytes]]) -> list[bytes]:
        word_bytes = [bytes([b]) for b in word.encode("utf-8")]
        for merge in merges:
            new_word_bytes = []
            i = 0
            while i < len(word_bytes):
                if i < len(word_bytes) - 1 and (word_bytes[i], word_bytes[i + 1]) == merge:
                    new_word_bytes.append(merge[0] + merge[1])
                    i += 2
                else:
                    new_word_bytes.append(word_bytes[i])
                    i += 1
            word_bytes = new_word_bytes

        return word_bytes

    def __split_with_special_tokens(self, text: str) -> Iterable[str]:
        if not self.special_tokens:
            yield text
            return

        # First split the corpus with special tokens
        split_special_token_pattern = "(" + "|".join([f"{re.escape(token)}" for token in self.special_tokens]) + ")"
        yield from re.splititer(split_special_token_pattern, text)


    def encode(self, text: str) -> list[int]:
        text_bytes = []

        for text_chunk in self.__split_with_special_tokens(text):
            if text_chunk in self.special_tokens:
                text_bytes.append(text_chunk.encode("utf-8"))
                continue

            for word in re.finditer(PRETOKENIZATION_PATTERN, text_chunk):
                word_bytes = self._apply_merges(word.group(0), self.vocab.merges)
                text_bytes.extend(word_bytes)

        return self.vocab.lookup_tokens(text_bytes)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        bytes_list = [self.vocab.token_to_bytes[token_id] for token_id in token_ids if token_id in self.vocab.token_to_bytes]
        return b"".join(bytes_list).decode("utf-8", errors="ignore")


def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], num_processes: int = 4) -> Vocabulary:
    # TODO: Do this when we need to train on TinyStories.
    #
    # with open(input_path, "rb") as f:
    #     boundaries = find_chunk_boundaries(
    #         f, num_processes, "<|endoftext|>".encode("utf-8"))

    #     # The following is a serial implementation, but you can parallelize this
    #     # by sending each start/end pair to a set of processes.
    #     boundaries = zip(boundaries[:-1], boundaries[1:])
    #     for start, end in boundaries:
    #         # Read the chunk and decode it
    #         f.seek(start)
    #         chunk = f.read(end - start).decode("utf-8", errors="ignore")
    #         # Run pre-tokenization on your chunk and store the counts for each pre-token

    with open(input_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    return train_bpe_with_text(corpus, vocab_size, special_tokens)