import regex as re
import os

from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Sequence


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


def compute_next_merge(tokens: Sequence[Sequence[bytes]]) -> tuple[bytes, bytes]:
    # Compute token statistics
    token_stats: dict[tuple[bytes, bytes], int] = defaultdict(lambda: 0)

    for word in tokens:
        for i in range(len(word) - 1):
            token_stats[(word[i], word[i + 1])] += 1

    # Merge tokens
    stats_by_count: list[tuple[tuple[bytes, bytes], int]] = [(k, v) for k, v in token_stats.items()]
    stats_by_count.sort(key=lambda t: (t[1], t[0]), reverse=True)
    return stats_by_count[0][0]

def train_bpe_with_text(corpus: str, vocab_size: int, special_tokens: list[str]):
    vocab = Vocabulary(vocab_size=vocab_size, special_tokens=special_tokens)

    all_tokens: list[list[bytes]] = []

    # First split the corpus with special tokens
    split_special_token_pattern = "|".join([re.escape(token) for token in special_tokens])

    for corpus_chunk in re.split(split_special_token_pattern, corpus):
        # Pretokenize
        for word in re.finditer(PRETOKENIZATION_PATTERN, corpus_chunk):
            word_tokens = []
            word_bytes = word.group(0).encode("utf-8")
            for b in word_bytes:
                word_tokens.append(bytes([b]))
            all_tokens.append(word_tokens)

    # BPE training loop
    while len(vocab.token_to_bytes) < vocab_size:
        next_merge = compute_next_merge(all_tokens)
        vocab.merge_tokens(next_merge)

        new_all_tokens = []
        for word in all_tokens:
            word_tokens = []
            i = 0
            while i <= len(word) - 1:
                if i == len(word) - 1:
                    word_tokens.append(word[i])
                    i += 1
                elif (word[i], word[i + 1]) == next_merge:
                    word_tokens.append(word[i] + word[i + 1])
                    i += 2
                else:
                    word_tokens.append(word[i])
                    i += 1
            new_all_tokens.append(word_tokens)
        all_tokens = new_all_tokens
        del new_all_tokens
    return vocab

def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]) -> Vocabulary:
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