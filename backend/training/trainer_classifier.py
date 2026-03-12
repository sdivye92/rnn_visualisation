"""Training loop for movie review sentiment classification."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from backend.data.movie_review_data import MovieDataConfig, build_imdb_dataloaders
from backend.models.classifier import ClassifierConfig, ReviewClassifier


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    max_batches: int | None = None,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_acc = 0.0
    batches = 0

    for batch_idx, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        if train_mode:
            optimizer.zero_grad()

        out = model(input_ids, lengths=lengths)
        logits = out["logits"]
        loss = criterion(logits, labels)

        if train_mode:
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
        total_loss += float(loss.item())
        total_acc += acc
        batches += 1
        if max_batches and batch_idx >= max_batches:
            break

    return {
        "loss": total_loss / max(batches, 1),
        "accuracy": total_acc / max(batches, 1),
    }


def train_movie_classifier(
    *,
    model_type: str,
    epochs: int,
    learning_rate: float,
    embedding_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    max_vocab_size: int,
    max_seq_len: int,
    min_freq: int = 2,
    train_limit: int = 0,
    val_limit: int = 0,
    test_limit: int = 0,
    max_train_batches_per_epoch: int | None = None,
    max_val_batches_per_epoch: int | None = None,
    save_dir: str = "saved_models",
    model_name: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    device = _device()
    data_cfg = MovieDataConfig(
        batch_size=batch_size,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
        min_freq=min_freq,
        train_limit=train_limit,
        val_limit=val_limit,
        test_limit=test_limit,
    )
    artifacts = build_imdb_dataloaders(data_cfg)

    model_cfg = ClassifierConfig(
        vocab_size=len(artifacts["vocab"]),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        rnn_type=model_type,
        pad_idx=artifacts["vocab"]["<PAD>"],
    )
    model = ReviewClassifier(model_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    for epoch in tqdm(range(1, epochs + 1), desc="Training classifier"):
        train_metrics = _run_epoch(
            model,
            artifacts["train_loader"],
            criterion,
            optimizer,
            device,
            max_batches=max_train_batches_per_epoch,
        )
        val_metrics = _run_epoch(
            model,
            artifacts["val_loader"],
            criterion,
            None,
            device,
            max_batches=max_val_batches_per_epoch,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        epoch_payload = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "history": history,
        }
        if progress_callback:
            progress_callback(epoch_payload)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = model_name or f"movie_cls_{model_type.lower()}_{timestamp}"
    run_dir = Path(save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), run_dir / "model.pt")
    metadata = {
        "task": "movie_classification",
        "model_type": model_type.upper(),
        "timestamp": timestamp,
        "split_sizes": artifacts["split_sizes"],
        "config": {
            "learning_rate": learning_rate,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "max_vocab_size": max_vocab_size,
            "max_seq_len": max_seq_len,
            "min_freq": min_freq,
            "vocab_size": len(artifacts["vocab"]),
            "train_limit": train_limit,
            "val_limit": val_limit,
            "test_limit": test_limit,
            "max_train_batches_per_epoch": max_train_batches_per_epoch,
            "max_val_batches_per_epoch": max_val_batches_per_epoch,
        },
        "vocab": artifacts["vocab"],
        "idx_to_word": {str(k): v for k, v in artifacts["idx_to_word"].items()},
        "history": history,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model_dir": str(run_dir),
        "metadata": metadata,
        "history": history,
        "device": str(device),
    }

