"""Training entry points for both tasks."""

from .trainer_classifier import train_movie_classifier
from .trainer_text_gen import train_text_generator

__all__ = ["train_text_generator", "train_movie_classifier"]

