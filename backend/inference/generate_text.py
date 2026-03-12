"""Text generation inference for trained word-level models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from backend.data.text_gen_data import decode_ids, encode_text, tokenize
from backend.models.text_gen import TextGenConfig, TextGenerationModel


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_text_gen_model(model_dir: str) -> Dict:
    model_path = Path(model_dir)
    metadata = json.loads((model_path / "metadata.json").read_text(encoding="utf-8"))
    idx_to_word = {int(k): v for k, v in metadata["idx_to_word"].items()}
    word_to_idx = metadata["word_to_idx"]
    cfg = metadata["config"]

    model_cfg = TextGenConfig(
        vocab_size=cfg["vocab_size"],
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        rnn_type=metadata["model_type"],
    )
    model = TextGenerationModel(model_cfg)
    model.load_state_dict(torch.load(model_path / "model.pt", map_location="cpu"))
    model.eval()

    return {
        "model": model,
        "metadata": metadata,
        "word_to_idx": word_to_idx,
        "idx_to_word": idx_to_word,
    }


def generate_text(
    model_bundle: Dict,
    seed_text: str,
    length: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Dict:
    model: TextGenerationModel = model_bundle["model"]
    word_to_idx = model_bundle["word_to_idx"]
    idx_to_word = model_bundle["idx_to_word"]
    model.eval()
    device = _device()
    model = model.to(device)

    if not seed_text:
        seed_text = "alice was"

    seed_words = tokenize(seed_text)
    seed_ids = encode_text(seed_words, word_to_idx)
    if not seed_ids:
        seed_ids = [word_to_idx.get("the", 1)]

    generated_ids = list(seed_ids)
    hidden = None
    explain_steps: List[dict] = []

    with torch.no_grad():
        for _ in range(length):
            input_ids = torch.tensor([generated_ids], dtype=torch.long, device=device)
            out = model(input_ids, hidden=hidden)
            logits = out["logits"][:, -1, :] / max(temperature, 1e-6)
            hidden = out["hidden"]

            if top_k and top_k > 0:
                values, indices = torch.topk(
                    logits, k=min(top_k, logits.shape[-1]), dim=-1
                )
                probs = torch.softmax(values, dim=-1)
                next_local = torch.multinomial(probs, num_samples=1).item()
                next_idx = int(indices[0, next_local].item())
            else:
                probs = torch.softmax(logits, dim=-1)
                next_idx = int(torch.multinomial(probs, num_samples=1).item())

            generated_ids.append(next_idx)
            explain_steps.append(
                {
                    "context_ids": list(generated_ids[:-1]),
                    "predicted_id": next_idx,
                    "predicted_token": idx_to_word.get(next_idx, "<UNK>"),
                }
            )

    return {
        "generated_text": decode_ids(generated_ids, idx_to_word),
        "seed_text": seed_text,
        "generated_only": decode_ids(generated_ids[len(seed_ids) :], idx_to_word),
        "generated_ids": generated_ids,
        "steps": explain_steps,
    }
