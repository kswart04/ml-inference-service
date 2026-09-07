from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from inference_service.adapters.fake import FakeAdapter
from inference_service.api.errors import ServiceError, error_response
from inference_service.api.middleware import RequestBoundaryMiddleware
from inference_service.api.schemas import (
    ErrorResponse,
    ModelInfo,
    ModelsResponse,
    PredictRequest,
    PredictResponse,
)
from inference_service.core.contracts import ModelKey
from inference_service.runtime.config import Settings
from inference_service.runtime.registry import ModelRegistry


def create_app(settings: Settings | None = None) -> FastAPI:
    """M0 runs only a tiny in-memory fake; real execution requires M1's worker."""
    config = settings if settings is not None else Settings()
    adapter = FakeAdapter(max_text_characters=config.max_text_characters)
    registry = ModelRegistry([adapter])
    ready = False

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal ready
        try:
            adapter.load()
            ready = True
            yield
        finally:
            ready = False
            adapter.close()

    app = FastAPI(
        title="ML Inference Service",
        version="0.1.0",
        description="M0: deterministic fake adapter; scheduling and real models are planned.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestBoundaryMiddleware, max_body_bytes=config.max_body_bytes)

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError) -> JSONResponse:
        return error_response(request.state.request_id, exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            request.state.request_id, 422, "invalid_input", "Invalid request input."
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        response = error_response(
            request.state.request_id,
            exc.status_code,
            "http_error",
            "HTTP request could not be handled.",
        )
        if exc.headers:
            response.headers.update(exc.headers)
        return response

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        return error_response(
            request.state.request_id, 500, "internal_error", "Prediction service error."
        )

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready", responses={503: {"model": ErrorResponse}})
    async def readiness() -> dict[str, str]:
        if not ready:
            raise ServiceError(503, "unavailable", "Service is not ready.")
        return {"status": "ready"}

    @app.get("/v1/models")
    async def models() -> ModelsResponse:
        return ModelsResponse(
            models=[
                ModelInfo(
                    model_id=metadata.key.model_id,
                    model_version=metadata.key.model_version,
                    task=metadata.task,
                    input_type=metadata.input_type,
                    labels=metadata.labels,
                    device=metadata.device,
                    max_batch_size=metadata.max_batch_size,
                    available=ready,
                )
                for metadata in registry.models()
            ]
        )

    @app.post(
        "/v1/predict",
        responses={status: {"model": ErrorResponse} for status in (404, 413, 422, 500, 503)},
    )
    async def predict(body: PredictRequest, request: Request) -> PredictResponse:
        key = ModelKey(body.model_id, body.model_version)
        try:
            selected = registry.get(key)
        except KeyError:
            raise ServiceError(404, "model_not_found", "Unknown model ID or version.") from None
        if not ready:
            raise ServiceError(503, "unavailable", "Service is not ready.")
        try:
            selected.validate(body.input)
        except ValueError as exc:
            raise ServiceError(422, "invalid_input", str(exc)) from None
        # Deliberately one item in M0. Replace this with scheduler submission in M1.
        results = selected.predict_batch([body.input])
        if len(results) != 1:
            raise ServiceError(
                500, "adapter_contract_error", "Adapter returned an invalid result count."
            )
        return PredictResponse(
            request_id=request.state.request_id,
            model_id=key.model_id,
            model_version=key.model_version,
            prediction=results[0],
        )

    return app
