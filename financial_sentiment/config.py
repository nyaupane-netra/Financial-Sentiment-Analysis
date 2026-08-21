"""Environment-driven application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    model_path: Path
    device: str = "cpu"

    @classmethod
    def from_environment(cls) -> Settings:
        return cls(
            model_path=Path(
                os.getenv(
                    "FIN_SENTIMENT_MODEL_PATH",
                    PROJECT_ROOT / "custom_financial_bert",
                )
            ).expanduser(),
            device=os.getenv("FIN_SENTIMENT_DEVICE", "cpu"),
        )
