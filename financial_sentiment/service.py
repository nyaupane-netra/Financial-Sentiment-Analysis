"""Lifecycle boundary for loading and calling the custom model."""

from __future__ import annotations

from threading import Lock
from typing import Any

from .config import Settings
from .core import predict, require_local_model


class SentimentService:
    """Lazily load one checkpoint and reuse it safely across requests."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_environment()
        self._bundle: tuple[Any, Any, Any] | None = None
        self._load_lock = Lock()

    @property
    def is_ready(self) -> bool:
        return (
            self.settings.model_path.is_dir()
            and (self.settings.model_path / "config.json").is_file()
        )

    @property
    def is_loaded(self) -> bool:
        return self._bundle is not None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._bundle is not None:
            return self._bundle

        with self._load_lock:
            if self._bundle is None:
                import torch
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )

                model_path = require_local_model(self.settings.model_path)
                device = torch.device(self.settings.device)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, local_files_only=True
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, local_files_only=True
                ).to(device)
                model.eval()
                self._bundle = tokenizer, model, device
        return self._bundle

    def predict(self, text: str) -> dict[str, Any]:
        tokenizer, model, device = self._load()
        return predict(text, tokenizer, model, device)
