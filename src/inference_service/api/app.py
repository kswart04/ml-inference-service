import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse, Response

from inference_service.adapters.fake import FakeAdapter
from inference_service.adapters.protocol import ModelAdapter
from inference_service.api.errors import ServiceError, error_response
from inference_service.api.middleware import RequestBoundaryMiddleware
from inference_service.api.schemas import (
    ErrorResponse,
    ModelInfo,
    ModelsResponse,
    PredictRequest,
    PredictResponse,
)
from inference_service.core.contracts import ModelKey, Prediction
from inference_service.core.exceptions import (
    AdapterContractError,
    BatchExecutionError,
    QueueFullError,
    RequestDeadlineError,
    SchedulerUnavailableError,
)
from inference_service.core.scheduler import ModelScheduler, SchedulerConfig
from inference_service.observability.logging import configure_logging
from inference_service.observability.metrics import ServiceMetrics
from inference_service.runtime.config import Settings
from inference_service.runtime.registry import ModelRegistry


async def _await_prediction(
    scheduler: ModelScheduler, request: Request, body: PredictRequest
) -> Prediction:
    prediction = asyncio.create_task(
        scheduler.submit(request.state.request_id, body.input, started_at=request.state.started_at)
    )
    stop_disconnect_poll = asyncio.Event()

    async def disconnected() -> None:
        while not stop_disconnect_poll.is_set() and not await request.is_disconnected():
            try:
                await asyncio.wait_for(stop_disconnect_poll.wait(), timeout=0.05)
            except TimeoutError:
                pass

    disconnect = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait({prediction, disconnect}, return_when=asyncio.FIRST_COMPLETED)
        if disconnect in done and prediction not in done:
            raise asyncio.CancelledError
        return await prediction
    finally:
        stop_disconnect_poll.set()
        for task in (prediction, disconnect):
            if not task.done():
                task.cancel()
        await asyncio.gather(prediction, disconnect, return_exceptions=True)


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings if settings is not None else Settings()
    adapter: ModelAdapter
    if config.adapter == "fake":
        adapter = FakeAdapter(max_text_characters=config.max_text_characters)
    elif config.adapter == "huggingface":
        from inference_service.adapters.huggingface import HuggingFaceSentimentAdapter

        adapter = HuggingFaceSentimentAdapter(
            config.artifact_dir,
            device=config.device,
            max_text_characters=config.max_text_characters,
        )
    else:
        from inference_service.adapters.custom import CustomSentimentAdapter

        adapter = CustomSentimentAdapter(
            config.custom_artifact_dir,
            device=config.device,
            max_text_characters=config.max_text_characters,
        )
    if config.max_batch_size > adapter.metadata.max_batch_size:
        raise ValueError("Configured batch size exceeds adapter maximum.")
    registry = ModelRegistry([adapter])
    metrics = ServiceMetrics()
    logger = configure_logging()
    scheduler = ModelScheduler(
        adapter,
        SchedulerConfig(
            policy=config.scheduling_policy,
            max_batch_size=config.max_batch_size,
            collection_delay_seconds=config.max_collection_delay_ms / 1000,
            pending_capacity=config.pending_capacity,
            deadline_seconds=config.request_deadline_ms / 1000,
            shutdown_grace_seconds=config.graceful_shutdown_seconds,
            watchdog_seconds=config.worker_watchdog_seconds,
        ),
        metrics,
        logger,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await scheduler.start()
        logger.info("service_started", extra={"policy": config.scheduling_policy.value})
        try:
            yield
        finally:
            logger.info("service_draining")
            await scheduler.shutdown()
            logger.info("service_stopped")

    app = FastAPI(
        title="ML Inference Service",
        version="0.4.0",
        description="M3: shared batching for fake, Hugging Face, and locally trained models.",
        lifespan=lifespan,
    )
    app.state.scheduler = scheduler
    app.state.metrics = metrics
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
        logger.error("request_error", extra={"error_category": type(exc).__name__})
        return error_response(
            request.state.request_id, 500, "internal_error", "Prediction service error."
        )

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready", responses={503: {"model": ErrorResponse}})
    async def readiness() -> dict[str, str]:
        if not scheduler.ready:
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
                    available=scheduler.ready,
                )
                for metadata in registry.models()
            ]
        )

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/v1/predict",
        response_model=PredictResponse,
        responses={
            status: {"model": ErrorResponse} for status in (404, 413, 422, 429, 500, 503, 504)
        },
    )
    async def predict(body: PredictRequest, request: Request) -> PredictResponse | JSONResponse:
        key = ModelKey(body.model_id, body.model_version)
        try:
            selected = registry.get(key)
        except KeyError:
            raise ServiceError(404, "model_not_found", "Unknown model ID or version.") from None
        try:
            selected.validate(body.input)
        except ValueError as exc:
            raise ServiceError(422, "invalid_input", str(exc)) from None
        try:
            result = await _await_prediction(scheduler, request, body)
        except QueueFullError:
            response = error_response(
                request.state.request_id, 429, "queue_full", "Pending queue capacity reached."
            )
            response.headers["Retry-After"] = str(config.retry_after_seconds)
            return response
        except RequestDeadlineError:
            raise ServiceError(
                504, "deadline_exceeded", "Server request deadline exceeded."
            ) from None
        except SchedulerUnavailableError:
            raise ServiceError(503, "unavailable", "Model worker is unavailable.") from None
        except AdapterContractError:
            raise ServiceError(
                500, "adapter_contract_error", "Adapter returned an invalid result count."
            ) from None
        except BatchExecutionError:
            raise ServiceError(500, "execution_error", "Model batch execution failed.") from None
        return PredictResponse(
            request_id=request.state.request_id,
            model_id=key.model_id,
            model_version=key.model_version,
            prediction=result,
        )

    return app
