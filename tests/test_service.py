from pathlib import Path

from financial_sentiment.config import Settings
from financial_sentiment.service import SentimentService


def test_service_reports_missing_model(tmp_path: Path):
    service = SentimentService(Settings(model_path=tmp_path / "missing"))
    assert service.is_ready is False
    assert service.is_loaded is False
