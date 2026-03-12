"""Training loop for word-level text generation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from backend.data.text_gen_data import TextDataConfig, build_text_gen_dataloaders
from backend.models.text_gen import TextGenConfig, TextGenerationModel


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / max(total, 1)


def _run_epoch(
    model: TextGenerationModel,
    loader,
    criterion,
    optimizer: Optional[optim.Optimizer],
    device: torch.device,
    max_batches: int | None = None,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_acc = 0.0
    batches = 0

    for batch_idx, (x_batch, y_batch) in enumerate(loader, start=1):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if train_mode:
            optimizer.zero_grad()

        out = model(x_batch)
        logits = out["logits"]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), y_batch.reshape(-1))

        if train_mode:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_acc += _accuracy_from_logits(logits, y_batch)
        batches += 1
        if max_batches and batch_idx >= max_batches:
            break

    return {
        "loss": total_loss / max(batches, 1),
        "accuracy": total_acc / max(batches, 1),
    }


def train_text_generator(
    *,
    data_path: str,
    model_type: str,
    epochs: int,
    learning_rate: float,
    embedding_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    seq_len: int,
    batch_size: int,
    val_ratio: float = 0.2,
    max_train_batches_per_epoch: int | None = None,
    max_val_batches_per_epoch: int | None = None,
    save_dir: str = "saved_models",
    model_name: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    device = _device()
    data_cfg = TextDataConfig(
        data_path=data_path,
        seq_len=seq_len,
        batch_size=batch_size,
        val_ratio=val_ratio,
    )
    data_artifacts = build_text_gen_dataloaders(data_cfg)

    model_cfg = TextGenConfig(
        vocab_size=data_artifacts["vocab_size"],
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        rnn_type=model_type,
    )
    model = TextGenerationModel(model_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    for epoch in tqdm(range(1, epochs + 1), desc="Training text generator"):
        train_metrics = _run_epoch(
            model,
            data_artifacts["train_loader"],
            criterion,
            optimizer,
            device,
            max_batches=max_train_batches_per_epoch,
        )
        val_metrics = _run_epoch(
            model,
            data_artifacts["val_loader"],
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
    run_name = model_name or f"text_gen_{model_type.lower()}_{timestamp}"
    run_dir = Path(save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), run_dir / "model.pt")
    metadata = {
        "task": "text_generation",
        "model_type": model_type.upper(),
        "timestamp": timestamp,
        "config": {
            "learning_rate": learning_rate,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "val_ratio": val_ratio,
            "vocab_size": data_artifacts["vocab_size"],
            "max_train_batches_per_epoch": max_train_batches_per_epoch,
            "max_val_batches_per_epoch": max_val_batches_per_epoch,
        },
        "word_to_idx": data_artifacts["word_to_idx"],
        "idx_to_word": {str(k): v for k, v in data_artifacts["idx_to_word"].items()},
        "history": history,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return {
        "model_dir": str(run_dir),
        "metadata": metadata,
        "history": history,
        "device": str(device),
    }
