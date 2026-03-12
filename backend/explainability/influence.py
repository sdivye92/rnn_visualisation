"""Gradient and causal contribution explainability methods."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def _to_device(model: torch.nn.Module, input_ids: List[int]) -> torch.Tensor:
    device = next(model.parameters()).device
    return torch.tensor([input_ids], dtype=torch.long, device=device)


def _target_logit(logits: torch.Tensor, target_index: int, task: str) -> torch.Tensor:
    if task == "text_generation":
        return logits[0, -1, target_index]
    return logits[0, target_index]


def _forward_model(
    model: torch.nn.Module,
    batch: torch.Tensor,
    task: str,
    length: int | None,
    return_embeddings: bool = False,
    return_hidden_states: bool = False,
) -> Dict:
    if task == "movie_classification":
        lengths = None if length is None else torch.tensor([length], device=batch.device)
        return model(
            batch,
            lengths=lengths,
            return_embeddings=return_embeddings,
            return_hidden_states=return_hidden_states,
        )
    return model(
        batch,
        return_embeddings=return_embeddings,
        return_hidden_states=return_hidden_states,
    )


def gradient_influence(
    model: torch.nn.Module,
    input_ids: List[int],
    target_index: int,
    task: str,
    length: int | None = None,
) -> List[float]:
    model.zero_grad(set_to_none=True)
    batch = _to_device(model, input_ids)
    out = _forward_model(model, batch, task, length, return_embeddings=True)
    score = _target_logit(out["logits"], target_index, task)

    grads = torch.autograd.grad(score, out["embeddings"], retain_graph=False)[0][0]
    return torch.norm(grads, dim=-1).detach().cpu().tolist()


def flow_heatmap(scores: List[float]) -> List[float]:
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return []
    normalized = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return normalized.tolist()


def hidden_state_delta_projection(
    model: torch.nn.Module,
    input_ids: List[int],
    target_index: int,
    task: str,
    length: int | None = None,
) -> List[float]:
    with torch.no_grad():
        batch = _to_device(model, input_ids)
        out = _forward_model(model, batch, task, length, return_hidden_states=True)
        hidden_states = out["hidden_states"][0]  # (seq, hidden_dim)
        weights = model.output_weights()[target_index]  # (hidden_dim,)

        prev = torch.zeros_like(hidden_states[0])
        scores = []
        for t in range(hidden_states.shape[0]):
            delta_h = hidden_states[t] - prev
            contribution = torch.dot(delta_h, weights)
            scores.append(float(contribution.item()))
            prev = hidden_states[t]
    return scores


def cumulative_confidence_curve(
    model: torch.nn.Module,
    input_ids: List[int],
    target_index: int,
    task: str,
) -> List[float]:
    values: List[float] = []
    with torch.no_grad():
        for end in range(1, len(input_ids) + 1):
            prefix = input_ids[:end]
            batch = _to_device(model, prefix)
            out = _forward_model(
                model,
                batch,
                task,
                len(prefix) if task == "movie_classification" else None,
            )
            logits = out["logits"]
            if task == "text_generation":
                probs = torch.softmax(logits[0, -1], dim=-1)
            else:
                probs = torch.softmax(logits[0], dim=-1)
            values.append(float(probs[target_index].item()))
    return values


def embedding_weighted_gradients(
    model: torch.nn.Module,
    input_ids: List[int],
    target_index: int,
    task: str,
    length: int | None = None,
) -> Dict[str, List[float]]:
    model.zero_grad(set_to_none=True)
    batch = _to_device(model, input_ids)
    out = _forward_model(model, batch, task, length, return_embeddings=True)
    score = _target_logit(out["logits"], target_index, task)
    grads = torch.autograd.grad(score, out["embeddings"], retain_graph=False)[0][0]
    emb = out["embeddings"][0]

    input_x_grad = emb * grads
    signed = torch.sum(input_x_grad, dim=-1)
    magnitude = torch.norm(input_x_grad, dim=-1)
    return {
        "signed": signed.detach().cpu().tolist(),
        "magnitude": magnitude.detach().cpu().tolist(),
    }


def explain_text_generation_step(
    model: torch.nn.Module,
    context_ids: List[int],
    predicted_id: int,
) -> Dict:
    grad = gradient_influence(model, context_ids, predicted_id, task="text_generation")
    heat = flow_heatmap(grad)
    delta = hidden_state_delta_projection(model, context_ids, predicted_id, task="text_generation")
    confidence = cumulative_confidence_curve(model, context_ids, predicted_id, task="text_generation")
    ixg = embedding_weighted_gradients(model, context_ids, predicted_id, task="text_generation")
    return {
        "gradient_influence": grad,
        "information_flow_heatmap": heat,
        "hidden_state_delta_projection": delta,
        "cumulative_confidence_curve": confidence,
        "embedding_weighted_gradients": ixg,
    }


def explain_classification(
    model: torch.nn.Module,
    input_ids: List[int],
    predicted_label: int,
    length: int,
) -> Dict:
    grad = gradient_influence(
        model, input_ids, predicted_label, task="movie_classification", length=length
    )
    heat = flow_heatmap(grad)
    delta = hidden_state_delta_projection(
        model, input_ids, predicted_label, task="movie_classification", length=length
    )
    confidence = cumulative_confidence_curve(
        model, input_ids[:length], predicted_label, task="movie_classification"
    )
    ixg = embedding_weighted_gradients(
        model, input_ids, predicted_label, task="movie_classification", length=length
    )
    return {
        "gradient_influence": grad[:length],
        "information_flow_heatmap": heat[:length],
        "hidden_state_delta_projection": delta[:length],
        "cumulative_confidence_curve": confidence,
        "embedding_weighted_gradients": {
            "signed": ixg["signed"][:length],
            "magnitude": ixg["magnitude"][:length],
        },
    }

