import regex as re
import os
import json

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Iterable, Generator


# GPT-2 pre-tokenization pattern
PRETOKENIZATION_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

@dataclass
class Vocabulary:
    token_to_bytes: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    merge_ranks: dict[tuple[bytes, bytes], int]
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
            # Create a dictionary mapping merge pairs to their rank (order in the merges list).
            # Lower rank means it was learned earlier and should be prioritized.
            self.merge_ranks = {merge: i for i, merge in enumerate(self.merges)}

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
        self.merge_ranks[new_bytes_tuple] = len(self.merges)

        new_bytes = tokens[0] + tokens[1]
        return self._add_token(new_bytes)

    def lookup_tokens(self, input_bytes: list[bytes]) -> list[int]:
        # TODO: Figure out an "UNKNOWN" token
        lst = []
        for b in input_bytes:
            if b in self.bytes_to_token:
                lst.append(self.bytes_to_token[b])
            else:
                print(f"Unknown byte {str(b)}!")
                lst.append(-1)
        return lst
        return [self.bytes_to_token.get(b, -1) for b in input_bytes]

    def save_to_file(self, vocab_path: str | os.PathLike, merges_path: str | os.PathLike) -> None:
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v.hex() for k, v in self.token_to_bytes.items()}, f, ensure_ascii=False, indent=0)

        with open(merges_path, "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a.hex()} {b.hex()}\n")


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
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        vocab_converted = {int(k): bytes.fromhex(v) for k, v in vocab_data.items()}

        with open(merges_path, "r", encoding="utf-8") as f:
            merges_data = []
            for line in f:
                s = line.strip()
                if not s:
                    continue
                # We deliberately split on a single space; hex never contains spaces.
                lhs_hex, rhs_hex = s.split(" ", 1)
                merges_data.append((bytes.fromhex(lhs_hex), bytes.fromhex(rhs_hex)))

        vocab = Vocabulary(
            vocab_size=len(vocab_converted),
            vocab=vocab_converted,
            merges=merges_data
        )

        return cls(vocab, special_tokens=special_tokens)

    @classmethod
    def corpus_to_pretokenized_words(cls, corpus: str, special_tokens: list[str]) -> Generator[bytes]:
        split_special_token_pattern = "|".join([re.escape(token) for token in special_tokens])
        for corpus_chunk in re.split(split_special_token_pattern, corpus):
            # Pretokenize
            for word in re.finditer(PRETOKENIZATION_PATTERN, corpus_chunk):
                yield word.group(0).encode("utf-8")


    def _apply_merges(self, word: str, vocab: Vocabulary) -> list[bytes]:
        # Initially, the word is a sequence of single-byte bytes objects.
        parts = [bytes([b]) for b in word.encode("utf-8")]

        while True:
            # Find the next pair to merge.
            # We look for the pair with the lowest rank in merge_ranks.
            min_rank = float("inf")
            best_pair_index = -1

            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                if pair in vocab.merge_ranks:
                    rank = vocab.merge_ranks[pair]
                    if rank < min_rank:
                        min_rank = rank
                        best_pair_index = i

            # If no mergeable pairs are found, we are done.
            if best_pair_index == -1:
                break

            # Merge the best pair.
            i = best_pair_index
            merged_part = parts[i] + parts[i + 1]
            parts = parts[:i] + [merged_part] + parts[i + 2:]

        return parts

    def __split_with_special_tokens(self, text: str) -> Iterable[str]:
        if not self.special_tokens:
            yield text
            return

        # First split the corpus with special tokens
        sorted_special_tokens = sorted(list(self.special_tokens), key=len, reverse=True)
        split_special_token_pattern = "(" + "|".join([f"{re.escape(token)}" for token in sorted_special_tokens]) + ")"
        yield from re.splititer(split_special_token_pattern, text)


    def encode(self, text: str) -> list[int]:
        text_bytes = []

        for text_chunk in self.__split_with_special_tokens(text):
            if text_chunk in self.special_tokens:
                text_bytes.append(text_chunk.encode("utf-8"))
                continue

            for word in re.finditer(PRETOKENIZATION_PATTERN, text_chunk):
                word_bytes = self._apply_merges(word.group(0), self.vocab)
                text_bytes.extend(word_bytes)

        return self.vocab.lookup_tokens(text_bytes)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        bytes_list = [self.vocab.token_to_bytes[token_id] for token_id in token_ids if token_id in self.vocab.token_to_bytes]
        return b"".join(bytes_list).decode("utf-8", errors="ignore")
