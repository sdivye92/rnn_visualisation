"""Word-level RNN/LSTM model for sentiment classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ClassifierConfig:
    vocab_size: int
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.1
    rnn_type: str = "LSTM"
    pad_idx: int = 0
    num_classes: int = 2


class ReviewClassifier(nn.Module):
    """Sequence classifier using final hidden representation."""

    def __init__(self, config: ClassifierConfig) -> None:
        super().__init__()
        rnn_type = config.rnn_type.upper()
        if rnn_type not in {"RNN", "LSTM"}:
            raise ValueError("rnn_type must be one of: RNN, LSTM")

        self.config = config
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(
            config.vocab_size, config.embedding_dim, padding_idx=config.pad_idx
        )
        dropout = config.dropout if config.num_layers > 1 else 0.0

        if rnn_type == "RNN":
            self.rnn = nn.RNN(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=dropout,
            )

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, ...]] = None,
        return_embeddings: bool = False,
        return_hidden_states: bool = False,
    ) -> dict:
        embeddings = self.embedding(x)
        hidden_states, hidden = self.rnn(embeddings, hidden)

        if lengths is not None:
            batch_idx = torch.arange(x.shape[0], device=x.device)
            last_timestep = torch.clamp(lengths - 1, min=0)
            final_hidden = hidden_states[batch_idx, last_timestep]
        else:
            final_hidden = hidden_states[:, -1, :]

        logits = self.fc(self.dropout(final_hidden))

        out = {"logits": logits, "hidden": hidden}
        if return_embeddings:
            out["embeddings"] = embeddings
        if return_hidden_states:
            out["hidden_states"] = hidden_states
        return out

    def output_weights(self) -> torch.Tensor:
        """Return classifier output projection weights (classes, hidden_dim)."""
        return self.fc.weight

