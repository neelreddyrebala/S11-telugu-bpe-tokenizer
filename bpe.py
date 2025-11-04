#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Sennrich-style BPE tokenizer for Unicode text.
- Trains merges on whitespace-separated tokens.
- Starts from character vocab (with </w> word-end marker).
- Exports merges and vocab to JSON.
- Provides encode/decode utilities and compression ratio calculation.
"""

from collections import Counter, defaultdict
import json
import re
from typing import List, Tuple, Dict

WORD_END = "</w>"
UNK = "<unk>"
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"

def normalize(text: str) -> str:
    # Keep simple normalization — collapse consecutive whitespace
    text = re.sub(r"\s+", " ", text.strip())
    return text

def word_to_symbols(word: str) -> Tuple[str, ...]:
    # Represent a word as tuple of characters + end marker
    # If word is empty, return just the end marker
    if not word:
        return (WORD_END,)
    return tuple(list(word)) + (WORD_END,)

def get_stats(tokenized_words: List[Tuple[str, ...]]) -> Counter:
    """Count frequency of adjacent symbol pairs across corpus."""
    pairs = Counter()
    for word in tokenized_words:
        for i in range(len(word)-1):
            pairs[(word[i], word[i+1])] += 1
    return pairs

def merge_pair(pair: Tuple[str, str], tokenized_words: List[Tuple[str, ...]]) -> List[Tuple[str, ...]]:
    """Merge a given pair across all words."""
    a, b = pair
    pattern = re.compile(r"(?<!\S)" + re.escape(a) + r" " + re.escape(b) + r"(?!\S)")
    # Instead of regex across strings, operate on tuples for clarity
    new_words = []
    for word in tokenized_words:
        i = 0
        merged = []
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i+1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(word[i])
                i += 1
        new_words.append(tuple(merged))
    return new_words

class BPETokenizer:
    def __init__(self):
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.special_tokens = [PAD, UNK, BOS, EOS]

    def train(self, texts: List[str], vocab_size_limit: int = 5000, min_pair_freq: int = 2, progress: bool = True):
        """
        Train BPE merges from a list of texts.
        - vocab_size_limit includes special tokens + initial chars + </w> + merges.
        - min_pair_freq: stop merging pairs that occur fewer than this many times.
        """
        # Prepare corpus words
        words = []
        for t in texts:
            t = normalize(t)
            words.extend(t.split(" "))

        # Build initial tokenized words (tuples of symbols)
        tokenized_words = [word_to_symbols(w) for w in words]

        # Initialize vocab with all individual symbols present
        symbol_counter = Counter()
        for w in tokenized_words:
            symbol_counter.update(w)
        symbols = sorted(symbol_counter.keys())

        # Start vocab with special tokens then symbols
        vocab_list = self.special_tokens + symbols
        self.vocab = {tok: i for i, tok in enumerate(vocab_list)}
        vocab_size = len(self.vocab)

        if progress:
            print(f"Initial symbols: {len(symbols)}; initial vocab (incl specials): {vocab_size}")

        # Greedy merging until vocab size limit or no frequent pairs
        while True:
            if vocab_size >= vocab_size_limit:
                if progress:
                    print(f"Reached vocab size limit: {vocab_size}")
                break

            stats = get_stats(tokenized_words)
            if not stats:
                if progress:
                    print("No more pairs to merge.")
                break

            (best_a, best_b), freq = stats.most_common(1)[0]
            if freq < min_pair_freq:
                if progress:
                    print(f"Stopping: best pair freq {freq} < min_pair_freq {min_pair_freq}")
                break

            # Add merge to list and update tokenization
            self.merges.append((best_a, best_b))
            tokenized_words = merge_pair((best_a, best_b), tokenized_words)

            # Update vocab with new symbol
            new_symbol = best_a + best_b
            if new_symbol not in self.vocab:
                self.vocab[new_symbol] = len(self.vocab)
                vocab_size = len(self.vocab)

            if progress and len(self.merges) % 50 == 0:
                print(f"Merges: {len(self.merges)}, vocab size: {vocab_size}, best pair: {(best_a, best_b)} freq {freq}")

        # Build reverse map
        self.id_to_token = {i: t for t, i in self.vocab.items()}

    def encode_word(self, word: str) -> List[str]:
        # Greedy application of learned merges to a single word
        if not self.merges:
            # Fallback: character tokens + WORD_END
            return list(word) + [WORD_END]

        symbols = list(word) + [WORD_END]
        # Apply merges in training order greedily until no change
        merges_lookup = set(self.merges)
        merged = True
        while merged:
            merged = False
            i = 0
            new_syms = []
            while i < len(symbols):
                if i < len(symbols)-1 and (symbols[i], symbols[i+1]) in merges_lookup:
                    new_syms.append(symbols[i] + symbols[i+1])
                    i += 2
                    merged = True
                else:
                    new_syms.append(symbols[i])
                    i += 1
            symbols = new_syms
        return symbols

    def encode(self, text: str) -> List[int]:
        text = normalize(text)
        tokens = []
        for w in text.split(" "):
            syms = self.encode_word(w)
            for s in syms:
                tok_id = self.vocab.get(s, self.vocab.get(UNK))
                tokens.append(tok_id if tok_id is not None else self.vocab[UNK])
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        syms = [self.id_to_token.get(t, UNK) for t in token_ids]
        # Reconstruct words by splitting on WORD_END
        words = []
        acc = []
        for s in syms:
            if s == WORD_END:
                words.append("".join(acc))
                acc = []
            elif s in self.special_tokens:
                # Skip specials in surface form
                continue
            else:
                acc.append(s)
        if acc:
            words.append("".join(acc))
        return " ".join(words)

    def save(self, path_prefix: str):
        data = {
            "merges": self.merges,
            "vocab": self.vocab,
            "special_tokens": self.special_tokens,
            "word_end": WORD_END,
        }
        with open(path_prefix + "_tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path_prefix: str):
        with open(path_prefix + "_tokenizer.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls()
        obj.merges = [tuple(m) for m in data["merges"]]
        obj.vocab = {k: int(v) for k, v in data["vocab"].items()}
        obj.id_to_token = {i: t for t, i in obj.vocab.items()}
        obj.special_tokens = data.get("special_tokens", [PAD, UNK, BOS, EOS])
        return obj

def compression_ratio(raw_text: str, token_ids: List[int]) -> float:
    # Define compression as: (raw character count including spaces) / (# tokens)
    # Higher is better (each token represents more characters).
    raw_chars = len(raw_text)
    n_tokens = max(1, len(token_ids))
    return raw_chars / n_tokens
