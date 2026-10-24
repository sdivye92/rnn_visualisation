"""Word-level preprocessing for Alice text generation."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass
class TextDataConfig:
    data_path: str
    seq_len: int = 30
    batch_size: int = 64
    val_ratio: float = 0.2
    seed: int = 42
    min_freq: int = 2
    max_vocab_size: int = 10000


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def build_word_vocab(
    words: List[str], max_vocab_size: int, min_freq: int
) -> Tuple[Dict[str, int], Dict[int, str]]:
    counts = Counter(words)
    word_to_idx: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, freq in counts.most_common(max_vocab_size):
        if freq < min_freq:
            continue
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word


def encode_text(words: List[str], word_to_idx: Dict[str, int]) -> List[int]:
    unk = word_to_idx[UNK_TOKEN]
    return [word_to_idx.get(w, unk) for w in words]


def decode_ids(ids: List[int], idx_to_word: Dict[int, str]) -> str:
    return " ".join(idx_to_word.get(i, UNK_TOKEN) for i in ids)


def _make_windows(
    encoded: List[int], seq_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(encoded) <= seq_len + 1:
        raise ValueError("Text is too short for the selected sequence length.")

    x_data = []
    y_data = []
    for start in range(0, len(encoded) - seq_len):
        x_data.append(encoded[start : start + seq_len])
        y_data.append(encoded[start + 1 : start + seq_len + 1])
    return torch.tensor(x_data, dtype=torch.long), torch.tensor(
        y_data, dtype=torch.long
    )


def build_text_gen_dataloaders(config: TextDataConfig) -> dict:
    text = Path(config.data_path).read_text(encoding="utf-8")
    words = tokenize(text)
    word_to_idx, idx_to_word = build_word_vocab(
        words, config.max_vocab_size, config.min_freq
    )
    encoded = encode_text(words, word_to_idx)

    x_tensor, y_tensor = _make_windows(encoded, config.seq_len)
    dataset = TensorDataset(x_tensor, y_tensor)

    val_size = int(len(dataset) * config.val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "vocab_size": len(word_to_idx),
        "word_to_idx": word_to_idx,
        "idx_to_word": idx_to_word,
        "raw_text_preview": text[:500],
    }
