"""Versioned HTTP API for custom-model inference."""

from __future__ import annotations

from typing import Annotated, Any, Protocol

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from .core import MAX_TEXT_LENGTH
from .service import SentimentService


class Predictor(Protocol):
    @property
    def is_ready(self) -> bool: ...

    @property
    def is_loaded(self) -> bool: ...

    def predict(self, text: str) -> dict[str, Any]: ...


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)


class PredictionResponse(BaseModel):
    predicted_label: str
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_available: bool
    model_loaded: bool


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor


def create_app(predictor: Predictor | None = None) -> FastAPI:
    application = FastAPI(
        title="Financial Sentiment API",
        version="1.0.0",
        description="Three-class financial sentiment inference.",
    )
    application.state.predictor = predictor or SentimentService()

    @application.get("/health", response_model=HealthResponse, tags=["operations"])
    def health(
        service: Annotated[Predictor, Depends(get_predictor)],
    ) -> HealthResponse:
        return HealthResponse(
            status="ok" if service.is_ready else "degraded",
            model_available=service.is_ready,
            model_loaded=service.is_loaded,
        )

    @application.post(
        "/v1/predictions",
        response_model=PredictionResponse,
        tags=["predictions"],
    )
    def create_prediction(
        payload: PredictionRequest,
        service: Annotated[Predictor, Depends(get_predictor)],
    ) -> PredictionResponse:
        try:
            return PredictionResponse(**service.predict(payload.text))
        except (FileNotFoundError, OSError) as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error

    return application


app = create_app()
