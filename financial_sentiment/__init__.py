"""Financial sentiment classification package."""

from .core import CANONICAL_LABELS, predict, probabilities_by_label, resolve_label_map
from .service import SentimentService

__all__ = [
    "CANONICAL_LABELS",
    "SentimentService",
    "predict",
    "probabilities_by_label",
    "resolve_label_map",
]
