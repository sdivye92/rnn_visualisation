"""Tab 1: Train text generation models."""

from __future__ import annotations

import streamlit as st

from backend.training.trainer_text_gen import train_text_generator
from frontend.components.training_viz import render_training_history


def render_train_text_gen_tab() -> None:
    st.subheader("Train Text Generation (Alice in Wonderland)")
    st.caption("Choose RNN/LSTM and train a word-level next-token model.")

    with st.form("text_train_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            model_type = st.selectbox("Model Type", ["RNN", "LSTM"], index=1)
            epochs = st.number_input("Epochs", min_value=1, max_value=500, value=5)
            learning_rate = st.number_input("Learning Rate", min_value=1e-5, max_value=1.0, value=0.001)
        with col2:
            embedding_dim = st.number_input("Embedding Dim", min_value=8, max_value=1024, value=128)
            hidden_dim = st.number_input("Hidden Dim", min_value=8, max_value=1024, value=256)
            num_layers = st.number_input("Num Layers", min_value=1, max_value=8, value=1)
        with col3:
            dropout = st.number_input("Dropout", min_value=0.0, max_value=0.9, value=0.1)
            seq_len = st.number_input("Sequence Length (words)", min_value=5, max_value=200, value=30)
            batch_size = st.number_input("Batch Size", min_value=4, max_value=1024, value=64)
        fast_mode = st.checkbox("Fast mode (fewer batches per epoch)", value=True)
        max_train_batches = st.number_input(
            "Max train batches/epoch (0 = full)",
            min_value=0,
            max_value=5000,
            value=80 if fast_mode else 0,
        )
        max_val_batches = st.number_input(
            "Max val batches/epoch (0 = full)",
            min_value=0,
            max_value=5000,
            value=20 if fast_mode else 0,
        )

        model_name = st.text_input("Model Save Name (optional)", value="")
        submit = st.form_submit_button("Start Training")

    train_placeholder = st.empty()
    chart_placeholder = st.empty()

    if submit:
        st.session_state["text_train_history"] = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        def _callback(epoch_payload):
            st.session_state["text_train_history"] = epoch_payload["history"]
            train_placeholder.info(
                f"Epoch {epoch_payload['epoch']}: "
                f"train_loss={epoch_payload['train_loss']:.4f}, "
                f"val_loss={epoch_payload['val_loss']:.4f}, "
                f"train_acc={epoch_payload['train_accuracy']:.4f}, "
                f"val_acc={epoch_payload['val_accuracy']:.4f}"
            )
            with chart_placeholder.container():
                render_training_history(
                    st.session_state["text_train_history"],
                    "Text Generation",
                    key_prefix=f"text_live_{epoch_payload['epoch']}",
                )

        result = train_text_generator(
            data_path="alice_in_wonderland.txt",
            model_type=model_type,
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            embedding_dim=int(embedding_dim),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
            seq_len=int(seq_len),
            batch_size=int(batch_size),
            max_train_batches_per_epoch=int(max_train_batches) or None,
            max_val_batches_per_epoch=int(max_val_batches) or None,
            model_name=model_name or None,
            progress_callback=_callback,
        )

        st.success(f"Training complete. Model saved at: `{result['model_dir']}`")
        with chart_placeholder.container():
            render_training_history(result["history"], "Text Generation", key_prefix="text_result")

    if "text_train_history" in st.session_state and st.session_state["text_train_history"]["train_loss"]:
        st.markdown("### Persisted Metrics")
        render_training_history(
            st.session_state["text_train_history"],
            "Text Generation",
            key_prefix="text_persisted",
        )

