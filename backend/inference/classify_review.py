"""Inference helpers for trained movie review classifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch

from backend.data.movie_review_data import PAD_TOKEN, tokenize
from backend.models.classifier import ClassifierConfig, ReviewClassifier


LABEL_MAP = {0: "negative", 1: "positive"}


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_classifier_model(model_dir: str) -> Dict:
    model_path = Path(model_dir)
    metadata = json.loads((model_path / "metadata.json").read_text(encoding="utf-8"))
    cfg = metadata["config"]
    vocab = metadata["vocab"]
    idx_to_word = {int(k): v for k, v in metadata["idx_to_word"].items()}

    model_cfg = ClassifierConfig(
        vocab_size=cfg["vocab_size"],
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        rnn_type=metadata["model_type"],
        pad_idx=vocab[PAD_TOKEN],
    )
    model = ReviewClassifier(model_cfg)
    model.load_state_dict(torch.load(model_path / "model.pt", map_location="cpu"))
    model.eval()

    return {
        "model": model,
        "metadata": metadata,
        "vocab": vocab,
        "idx_to_word": idx_to_word,
    }


def _encode_review(review: str, vocab: Dict[str, int], max_seq_len: int) -> Dict:
    tokens = tokenize(review)
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens][:max_seq_len]
    length = len(ids)
    if length < max_seq_len:
        ids.extend([vocab["<PAD>"]] * (max_seq_len - length))
    return {"input_ids": ids, "length": max(length, 1), "tokens": tokens[:max_seq_len]}


def classify_review(model_bundle: Dict, review: str) -> Dict:
    model: ReviewClassifier = model_bundle["model"]
    metadata = model_bundle["metadata"]
    vocab = model_bundle["vocab"]
    max_seq_len = metadata["config"]["max_seq_len"]
    encoded = _encode_review(review, vocab, max_seq_len)

    device = _device()
    model = model.to(device)
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    lengths = torch.tensor([encoded["length"]], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(input_ids, lengths=lengths)
        logits = out["logits"][0]
        probs = torch.softmax(logits, dim=-1)
        pred_idx = int(torch.argmax(probs).item())

    return {
        "label_idx": pred_idx,
        "label": LABEL_MAP[pred_idx],
        "confidence": float(probs[pred_idx].item()),
        "probabilities": {LABEL_MAP[i]: float(probs[i].item()) for i in range(2)},
        "tokens": encoded["tokens"],
        "input_ids": encoded["input_ids"],
        "length": encoded["length"],
    }

