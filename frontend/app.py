"""Main Streamlit app entrypoint for RNN/LSTM training + inference."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure absolute imports (frontend.*, backend.*) work when running:
# `streamlit run frontend/app.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.tabs import (
    render_infer_classifier_tab,
    render_infer_text_gen_tab,
    render_train_classifier_tab,
    render_train_text_gen_tab,
)


def main() -> None:
    st.set_page_config(page_title="RNN/LSTM Trainer UI", layout="wide")
    st.title("RNN/LSTM Training and Inference UI")
    st.caption(
        "Tasks: (1) Alice text generation, (2) IMDB movie review classification, "
        "with training metrics + explainability."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1) Training text gen",
            "2) Inference text gen",
            "3) Training movie review classification",
            "4) Inference",
        ]
    )

    with tab1:
        render_train_text_gen_tab()
    with tab2:
        render_infer_text_gen_tab()
    with tab3:
        render_train_classifier_tab()
    with tab4:
        render_infer_classifier_tab()


if __name__ == "__main__":
    main()

