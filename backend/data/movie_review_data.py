"""IMDB preprocessing with fixed 70-20-10 train/val/test split."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass
class MovieDataConfig:
    batch_size: int = 64
    max_vocab_size: int = 25000
    max_seq_len: int = 256
    min_freq: int = 2
    seed: int = 42
    train_limit: int = 0
    val_limit: int = 0
    test_limit: int = 0


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def build_vocab(
    tokenized_texts: List[List[str]], max_vocab_size: int, min_freq: int
) -> Dict[str, int]:
    counts = Counter()
    for tokens in tokenized_texts:
        counts.update(tokens)

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, freq in counts.most_common(max_vocab_size):
        if freq < min_freq:
            continue
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def encode_tokens(
    tokens: List[str], vocab: Dict[str, int], max_seq_len: int
) -> List[int]:
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens][:max_seq_len]
    if len(ids) < max_seq_len:
        ids.extend([vocab[PAD_TOKEN]] * (max_seq_len - len(ids)))
    return ids


class ReviewDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Dict[str, int],
        max_seq_len: int,
    ) -> None:
        self.labels = labels
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.tokenized = [tokenize(t) for t in texts]
        self.encoded = [
            encode_tokens(tokens, vocab, max_seq_len) for tokens in self.tokenized
        ]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        tokens = self.tokenized[idx][: self.max_seq_len]
        length = min(len(tokens), self.max_seq_len)
        return {
            "input_ids": torch.tensor(self.encoded[idx], dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "tokens": tokens,
        }


def _collate(batch: List[dict]) -> dict:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "lengths": torch.stack([item["length"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
        "tokens": [item["tokens"] for item in batch],
    }


def _split_70_20_10(texts: List[str], labels: List[int]) -> dict:
    total = len(texts)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.2)

    return {
        "train_texts": texts[:train_end],
        "train_labels": labels[:train_end],
        "val_texts": texts[train_end:val_end],
        "val_labels": labels[train_end:val_end],
        "test_texts": texts[val_end:],
        "test_labels": labels[val_end:],
    }


def _apply_limit(values: List, limit: int) -> List:
    if limit and limit > 0:
        return values[:limit]
    return values


@lru_cache(maxsize=8)
def _load_imdb_combined(seed: int) -> tuple[List[str], List[int]]:
    dataset = load_dataset("stanfordnlp/imdb")
    all_texts = list(dataset["train"]["text"]) + list(dataset["test"]["text"])
    all_labels = list(dataset["train"]["label"]) + list(dataset["test"]["label"])

    shuffled_idx = torch.randperm(
        len(all_texts), generator=torch.Generator().manual_seed(seed)
    )
    texts = [all_texts[i] for i in shuffled_idx.tolist()]
    labels = [all_labels[i] for i in shuffled_idx.tolist()]
    return texts, labels


def build_imdb_dataloaders(config: MovieDataConfig) -> dict:
    texts, labels = _load_imdb_combined(config.seed)

    splits = _split_70_20_10(texts, labels)
    splits["train_texts"] = _apply_limit(splits["train_texts"], config.train_limit)
    splits["train_labels"] = _apply_limit(splits["train_labels"], config.train_limit)
    splits["val_texts"] = _apply_limit(splits["val_texts"], config.val_limit)
    splits["val_labels"] = _apply_limit(splits["val_labels"], config.val_limit)
    splits["test_texts"] = _apply_limit(splits["test_texts"], config.test_limit)
    splits["test_labels"] = _apply_limit(splits["test_labels"], config.test_limit)

    train_tokenized = [tokenize(t) for t in splits["train_texts"]]
    vocab = build_vocab(train_tokenized, config.max_vocab_size, config.min_freq)
    idx_to_word = {idx: word for word, idx in vocab.items()}

    train_ds = ReviewDataset(
        splits["train_texts"], splits["train_labels"], vocab, config.max_seq_len
    )
    val_ds = ReviewDataset(
        splits["val_texts"], splits["val_labels"], vocab, config.max_seq_len
    )
    test_ds = ReviewDataset(
        splits["test_texts"], splits["test_labels"], vocab, config.max_seq_len
    )

    return {
        "train_loader": DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=_collate
        ),
        "val_loader": DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=_collate
        ),
        "test_loader": DataLoader(
            test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=_collate
        ),
        "vocab": vocab,
        "idx_to_word": idx_to_word,
        "split_sizes": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
    }


def load_imdb_split_samples(config: MovieDataConfig, limit: int = 200) -> List[dict]:
    """Return a subset of test samples for frontend inference selection."""
    texts, labels = _load_imdb_combined(config.seed)
    splits = _split_70_20_10(texts, labels)
    test_texts = _apply_limit(splits["test_texts"], config.test_limit)
    test_labels = _apply_limit(splits["test_labels"], config.test_limit)

    samples: List[dict] = []
    for text, label in zip(test_texts, test_labels):
        tokens = tokenize(text)[: config.max_seq_len]
        samples.append(
            {
                "text": " ".join(tokens),
                "tokens": tokens,
                "label": int(label),
                "length": len(tokens),
            }
        )
        if len(samples) >= limit:
            break
    return samples
