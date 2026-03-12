"""Character-level RNN/LSTM language model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class TextGenConfig:
    vocab_size: int
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.0
    rnn_type: str = "LSTM"


class TextGenerationModel(nn.Module):
    """Simple char-level model supporting vanilla RNN and LSTM."""

    def __init__(self, config: TextGenConfig) -> None:
        super().__init__()
        rnn_type = config.rnn_type.upper()
        if rnn_type not in {"RNN", "LSTM"}:
            raise ValueError("rnn_type must be one of: RNN, LSTM")

        self.config = config
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
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

        self.fc = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, ...]] = None,
        return_embeddings: bool = False,
        return_hidden_states: bool = False,
    ) -> dict:
        embeddings = self.embedding(x)
        hidden_states, hidden = self.rnn(embeddings, hidden)
        logits = self.fc(hidden_states)

        out = {"logits": logits, "hidden": hidden}
        if return_embeddings:
            out["embeddings"] = embeddings
        if return_hidden_states:
            out["hidden_states"] = hidden_states
        return out

    def output_weights(self) -> torch.Tensor:
        """Return output projection weights (vocab_size, hidden_dim)."""
        return self.fc.weight

