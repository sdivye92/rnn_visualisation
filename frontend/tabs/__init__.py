"""Tab renderers for the Streamlit app."""

from .infer_classifier import render_infer_classifier_tab
from .infer_text_gen import render_infer_text_gen_tab
from .train_classifier import render_train_classifier_tab
from .train_text_gen import render_train_text_gen_tab

__all__ = [
    "render_train_text_gen_tab",
    "render_infer_text_gen_tab",
    "render_train_classifier_tab",
    "render_infer_classifier_tab",
]

