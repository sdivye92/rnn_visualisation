"""Training metric visualizations."""

from __future__ import annotations

from typing import Dict, Optional

import plotly.graph_objects as go
import streamlit as st


def render_training_history(
    history: Dict,
    title_prefix: str = "",
    key_prefix: Optional[str] = None,
) -> None:
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    if not epochs:
        st.info("No history available yet.")
        return

    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], mode="lines+markers", name="Train Loss"))
    loss_fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines+markers", name="Val Loss"))
    loss_fig.update_layout(
        title=f"{title_prefix} Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend_title="Split",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    loss_key = None if key_prefix is None else f"{key_prefix}_loss"
    st.plotly_chart(loss_fig, width="stretch", key=loss_key)

    acc_fig = go.Figure()
    acc_fig.add_trace(
        go.Scatter(x=epochs, y=history["train_accuracy"], mode="lines+markers", name="Train Accuracy")
    )
    acc_fig.add_trace(go.Scatter(x=epochs, y=history["val_accuracy"], mode="lines+markers", name="Val Accuracy"))
    acc_fig.update_layout(
        title=f"{title_prefix} Accuracy",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        yaxis_range=[0, 1],
        legend_title="Split",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    acc_key = None if key_prefix is None else f"{key_prefix}_acc"
    st.plotly_chart(acc_fig, width="stretch", key=acc_key)

