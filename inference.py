"""Shared inference utilities for the Streamlit applications."""

from __future__ import annotations

from pathlib import Path
from typing import Any

CANONICAL_LABELS = ("negative", "neutral", "positive")


def resolve_label_map(model: Any) -> dict[int, str]:
    """Return a normalized label map from a Hugging Face model config.

    Older custom checkpoints often contain generic labels such as ``LABEL_0``.
    For those three-class checkpoints, this project uses its documented training
    order: negative, neutral, positive.
    """

    raw_map = getattr(getattr(model, "config", None), "id2label", {}) or {}
    normalized = {
        int(index): str(label).strip().lower()
        for index, label in raw_map.items()
    }

    if set(normalized.values()) == set(CANONICAL_LABELS):
        return normalized
    if len(normalized) == 3 or not normalized:
        return dict(enumerate(CANONICAL_LABELS))

    raise ValueError(
        "The model must expose negative, neutral, and positive sentiment labels."
    )


def probabilities_by_label(
    probabilities: list[float], label_map: dict[int, str]
) -> dict[str, float]:
    """Associate probabilities with labels and return canonical display order."""

    if len(probabilities) != len(label_map):
        raise ValueError("Probability count does not match the model label count.")

    mapped = {
        label_map[index]: float(probability)
        for index, probability in enumerate(probabilities)
    }
    missing = set(CANONICAL_LABELS) - mapped.keys()
    if missing:
        raise ValueError(f"Missing sentiment labels: {', '.join(sorted(missing))}")
    return {label: mapped[label] for label in CANONICAL_LABELS}


def predict(text: str, tokenizer: Any, model: Any, device: Any) -> dict[str, Any]:
    """Run one text through a sequence-classification model."""

    import torch

    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Text cannot be empty.")

    encoded = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

    with torch.inference_mode():
        logits = model(**encoded).logits
        values = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()

    probabilities = probabilities_by_label(values, resolve_label_map(model))
    predicted_label = max(probabilities, key=probabilities.get)
    return {"predicted_label": predicted_label, "probabilities": probabilities}


def require_local_model(path: Path) -> Path:
    """Validate that a local Hugging Face model directory is available."""

    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"Local model not found at {path}. Run the training notebook first."
        )
    return path
