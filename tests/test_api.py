from fastapi.testclient import TestClient

from financial_sentiment.api import create_app


class FakePredictor:
    is_ready = True
    is_loaded = True

    def predict(self, text):
        return {
            "predicted_label": "positive",
            "probabilities": {"negative": 0.05, "neutral": 0.15, "positive": 0.8},
        }


def test_health_reports_model_state():
    response = TestClient(create_app(FakePredictor())).get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_available": True,
        "model_loaded": True,
    }


def test_prediction_contract():
    response = TestClient(create_app(FakePredictor())).post(
        "/v1/predictions", json={"text": "Revenue exceeded expectations."}
    )
    assert response.status_code == 200
    assert response.json()["predicted_label"] == "positive"
    assert response.json()["probabilities"]["positive"] == 0.8


def test_prediction_rejects_empty_text():
    response = TestClient(create_app(FakePredictor())).post(
        "/v1/predictions", json={"text": ""}
    )
    assert response.status_code == 422


def test_prediction_returns_503_when_checkpoint_is_missing():
    class MissingModel(FakePredictor):
        is_ready = False
        is_loaded = False

        def predict(self, text):
            raise FileNotFoundError("checkpoint unavailable")

    response = TestClient(create_app(MissingModel())).post(
        "/v1/predictions", json={"text": "Revenue was stable."}
    )
    assert response.status_code == 503
