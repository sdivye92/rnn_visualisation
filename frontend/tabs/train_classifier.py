"""Tab 3: Train movie review classifier."""

from __future__ import annotations

import streamlit as st

from backend.training.trainer_classifier import train_movie_classifier
from frontend.components.training_viz import render_training_history


def render_train_classifier_tab() -> None:
    st.subheader("Train Movie Review Classification")
    st.caption("IMDB dataset with strict 70-20-10 split (train-val-test).")

    with st.form("movie_train_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            model_type = st.selectbox("Model Type", ["RNN", "LSTM"], index=1)
            epochs = st.number_input("Epochs", min_value=1, max_value=100, value=3)
            learning_rate = st.number_input("Learning Rate", min_value=1e-5, max_value=1.0, value=0.001)
        with col2:
            embedding_dim = st.number_input("Embedding Dim", min_value=8, max_value=512, value=128)
            hidden_dim = st.number_input("Hidden Dim", min_value=8, max_value=1024, value=128)
            num_layers = st.number_input("Num Layers", min_value=1, max_value=6, value=1)
        with col3:
            dropout = st.number_input("Dropout", min_value=0.0, max_value=0.9, value=0.1)
            batch_size = st.number_input("Batch Size", min_value=8, max_value=1024, value=64)
            max_seq_len = st.number_input("Max Sequence Length", min_value=32, max_value=1024, value=256)

        max_vocab_size = st.number_input("Max Vocabulary Size", min_value=1000, max_value=100000, value=25000)
        fast_mode = st.checkbox("Fast mode (subset + fewer batches)", value=True)
        train_limit = st.number_input(
            "Train samples limit (0 = full)",
            min_value=0,
            max_value=50000,
            value=8000 if fast_mode else 0,
        )
        val_limit = st.number_input(
            "Val samples limit (0 = full)",
            min_value=0,
            max_value=50000,
            value=2000 if fast_mode else 0,
        )
        test_limit = st.number_input(
            "Test samples limit (0 = full)",
            min_value=0,
            max_value=50000,
            value=1000 if fast_mode else 0,
        )
        max_train_batches = st.number_input(
            "Max train batches/epoch (0 = full)",
            min_value=0,
            max_value=5000,
            value=100 if fast_mode else 0,
        )
        max_val_batches = st.number_input(
            "Max val batches/epoch (0 = full)",
            min_value=0,
            max_value=5000,
            value=30 if fast_mode else 0,
        )
        model_name = st.text_input("Model Save Name (optional)", value="")
        submit = st.form_submit_button("Start Training")

    train_placeholder = st.empty()
    chart_placeholder = st.empty()

    if submit:
        st.session_state["movie_train_history"] = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        def _callback(epoch_payload):
            st.session_state["movie_train_history"] = epoch_payload["history"]
            train_placeholder.info(
                f"Epoch {epoch_payload['epoch']}: "
                f"train_loss={epoch_payload['train_loss']:.4f}, "
                f"val_loss={epoch_payload['val_loss']:.4f}, "
                f"train_acc={epoch_payload['train_accuracy']:.4f}, "
                f"val_acc={epoch_payload['val_accuracy']:.4f}"
            )
            with chart_placeholder.container():
                render_training_history(
                    st.session_state["movie_train_history"],
                    "Movie Review",
                    key_prefix=f"movie_live_{epoch_payload['epoch']}",
                )

        result = train_movie_classifier(
            model_type=model_type,
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            embedding_dim=int(embedding_dim),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
            batch_size=int(batch_size),
            max_vocab_size=int(max_vocab_size),
            max_seq_len=int(max_seq_len),
            train_limit=int(train_limit),
            val_limit=int(val_limit),
            test_limit=int(test_limit),
            max_train_batches_per_epoch=int(max_train_batches) or None,
            max_val_batches_per_epoch=int(max_val_batches) or None,
            model_name=model_name or None,
            progress_callback=_callback,
        )
        st.success(f"Training complete. Model saved at: `{result['model_dir']}`")
        st.caption(f"Split sizes: {result['metadata']['split_sizes']}")
        with chart_placeholder.container():
            render_training_history(result["history"], "Movie Review", key_prefix="movie_result")

    if "movie_train_history" in st.session_state and st.session_state["movie_train_history"]["train_loss"]:
        st.markdown("### Persisted Metrics")
        render_training_history(
            st.session_state["movie_train_history"],
            "Movie Review",
            key_prefix="movie_persisted",
        )

