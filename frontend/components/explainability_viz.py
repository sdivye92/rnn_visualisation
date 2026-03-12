"""Reusable explainability visualizations."""

from __future__ import annotations

from typing import Dict, List

import plotly.graph_objects as go
import streamlit as st


def _bar_chart(tokens: List[str], values: List[float], title: str, color: str = "#1f77b4") -> None:
    fig = go.Figure(go.Bar(x=tokens, y=values, marker_color=color))
    fig.update_layout(title=title, xaxis_title="Token", yaxis_title="Score", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, width="stretch")


def _diverging_chart(tokens: List[str], values: List[float], title: str) -> None:
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]
    fig = go.Figure(go.Bar(x=tokens, y=values, marker_color=colors))
    fig.update_layout(
        title=title,
        xaxis_title="Token",
        yaxis_title="Signed Contribution",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, width="stretch")


def _heatmap(tokens: List[str], values: List[float], title: str) -> None:
    fig = go.Figure(
        data=go.Heatmap(
            z=[values],
            x=tokens,
            y=["Influence"],
            colorscale="Viridis",
            colorbar={"title": "Influence"},
        )
    )
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, width="stretch")


def _line(tokens: List[str], values: List[float], title: str) -> None:
    fig = go.Figure(go.Scatter(x=tokens, y=values, mode="lines+markers", fill="tozeroy"))
    fig.update_layout(
        title=title,
        xaxis_title="Prefix Token",
        yaxis_title="Confidence",
        yaxis_range=[0, 1],
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, width="stretch")


def render_explainability_charts(tokens: List[str], explain: Dict) -> None:
    if not tokens:
        st.warning("No tokens available for explainability.")
        return

    st.markdown("#### Gradient Influence")
    _bar_chart(tokens, explain["gradient_influence"], "Gradient Influence")
    _heatmap(tokens, explain["information_flow_heatmap"], "Information Flow Heatmap")

    st.markdown("#### Token Contribution (Custom Methods)")
    _diverging_chart(tokens, explain["hidden_state_delta_projection"], "Hidden State Delta Projection")
    _line(tokens, explain["cumulative_confidence_curve"], "Cumulative Confidence Curve")

    ixg = explain["embedding_weighted_gradients"]
    _bar_chart(tokens, ixg["magnitude"], "Input x Gradient (Magnitude)", "#9467bd")
    _diverging_chart(tokens, ixg["signed"], "Input x Gradient (Signed)")

