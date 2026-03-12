"""Tab 4: Movie review inference + explainability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import streamlit as st

from backend.data.movie_review_data import MovieDataConfig, load_imdb_split_samples
from backend.explainability.influence import explain_classification
from backend.inference.classify_review import classify_review, load_classifier_model
from frontend.components.explainability_viz import render_explainability_charts


def _list_saved_classifier_models() -> List[Path]:
    root = Path("saved_models")
    if not root.exists():
        return []
    model_dirs = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        meta_file = child / "metadata.json"
        model_file = child / "model.pt"
        if not (meta_file.exists() and model_file.exists()):
            continue
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("task") == "movie_classification":
            model_dirs.append(child)
    return sorted(model_dirs, key=lambda p: p.name, reverse=True)


@st.cache_data(show_spinner=False)
def _cached_test_samples(max_seq_len: int, limit: int = 300):
    return load_imdb_split_samples(MovieDataConfig(max_seq_len=max_seq_len), limit=limit)


def render_infer_classifier_tab() -> None:
    st.subheader("Inference: Movie Review Classification")
    models = _list_saved_classifier_models()
    if not models:
        st.warning("No trained movie classification models found in `saved_models/`.")
        return

    chosen = st.selectbox("Select trained model", options=models, format_func=lambda p: p.name)
    bundle = load_classifier_model(str(chosen))
    max_seq_len = int(bundle["metadata"]["config"]["max_seq_len"])

    mode = st.radio("Input mode", ["Sample from test set", "Custom review text"], horizontal=True)
    review_text = ""

    if mode == "Sample from test set":
        samples = _cached_test_samples(max_seq_len=max_seq_len, limit=300)
        sample_idx = st.selectbox(
            "Pick test sample",
            options=list(range(len(samples))),
            format_func=lambda i: f"Sample {i} (label={samples[i]['label']})",
        )
        review_text = samples[sample_idx]["text"]
        st.text_area("Selected review", value=review_text, height=180, disabled=True)
    else:
        review_text = st.text_area("Enter your own review", value="", height=180)

    if st.button("Run Inference", type="primary") and review_text.strip():
        result = classify_review(bundle, review_text)
        explain = explain_classification(
            bundle["model"].to("cpu"),
            input_ids=result["input_ids"],
            predicted_label=result["label_idx"],
            length=result["length"],
        )
        st.session_state["cls_result"] = result
        st.session_state["cls_explain"] = explain

    if "cls_result" not in st.session_state:
        return

    result = st.session_state["cls_result"]
    explain = st.session_state["cls_explain"]

    st.markdown("### Prediction")
    st.success(f"Predicted sentiment: **{result['label']}**")
    st.write(f"Confidence: `{result['confidence']:.4f}`")
    st.json(result["probabilities"])

    tokens = result["tokens"]
    st.markdown("### Explainability")
    render_explainability_charts(tokens, explain)
