"""Tab 2: Text generation inference + explainability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import streamlit as st

from backend.explainability.influence import explain_text_generation_step
from backend.inference.generate_text import generate_text, load_text_gen_model
from frontend.components.explainability_viz import render_explainability_charts


def _list_saved_text_models() -> List[Path]:
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
        if meta.get("task") == "text_generation":
            model_dirs.append(child)
    return sorted(model_dirs, key=lambda p: p.name, reverse=True)


def _ids_to_tokens(ids: List[int], idx_to_word: dict) -> List[str]:
    return [idx_to_word.get(int(i), "?") for i in ids]


def render_infer_text_gen_tab() -> None:
    st.subheader("Inference: Text Generation")
    models = _list_saved_text_models()
    if not models:
        st.warning("No trained text-generation models found in `saved_models/`.")
        return

    chosen = st.selectbox("Select trained model", options=models, format_func=lambda p: p.name)
    seed_text = st.text_input("Seed phrase", value="Alice was beginning")
    gen_len = st.slider("Generated length (words)", min_value=5, max_value=200, value=40)
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.8, step=0.1)
    top_k = st.slider("Top-K sampling (0 = disabled)", min_value=0, max_value=50, value=10)

    if st.button("Generate Text", type="primary"):
        bundle = load_text_gen_model(str(chosen))
        output = generate_text(
            bundle, seed_text=seed_text, length=gen_len,
            temperature=temperature, top_k=top_k,
        )
        st.session_state["tg_output"] = output
        st.session_state["tg_bundle_dir"] = str(chosen)

    if "tg_output" not in st.session_state:
        return

    output = st.session_state["tg_output"]
    st.markdown("### Generated Text")
    st.code(output["generated_text"])

    if not output["steps"]:
        st.info("No generation steps available.")
        return

    step_idx = st.slider(
        "Inspect explainability for generated word step",
        min_value=1,
        max_value=len(output["steps"]),
        value=min(5, len(output["steps"])),
        key="tg_step_slider",
    )
    step = output["steps"][step_idx - 1]

    context_ids = step["context_ids"]
    max_ctx = 50
    if len(context_ids) > max_ctx:
        context_ids = context_ids[-max_ctx:]

    bundle = load_text_gen_model(st.session_state["tg_bundle_dir"])

    explain = explain_text_generation_step(
        bundle["model"].to("cpu"),
        context_ids=context_ids,
        predicted_id=step["predicted_id"],
    )
    tokens = _ids_to_tokens(context_ids, bundle["idx_to_word"])

    st.markdown(
        f"#### Explaining step {step_idx}: predicted word `{step['predicted_token']}` "
        f"from context of {len(tokens)} words"
    )
    render_explainability_charts(tokens, explain)
