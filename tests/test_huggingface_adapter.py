from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from inference_service.adapters.huggingface import (
    MANIFEST_NAME,
    MODEL_INTERNAL_VERSION,
    MODEL_MAX_LENGTH,
    MODEL_REPO_ID,
    MODEL_REVISION,
    HuggingFaceSentimentAdapter,
)
from inference_service.api.app import create_app
from inference_service.core.contracts import TextInput
from inference_service.core.scheduler import ModelScheduler, SchedulerConfig
from inference_service.observability.metrics import ServiceMetrics
from inference_service.runtime.config import SchedulingPolicy, Settings

pytestmark = pytest.mark.model
ARTIFACT_DIR = Path("artifacts/huggingface-sst2")


@pytest.fixture(scope="module")
def prepared_dir() -> Path:
    manifest = ARTIFACT_DIR / MANIFEST_NAME
    if not manifest.is_file():
        pytest.skip(
            "Hugging Face artifacts are missing; run "
            "`uv run --extra hf python scripts/prepare_huggingface_model.py`."
        )
    return ARTIFACT_DIR


@pytest.fixture(scope="module")
def adapter(prepared_dir: Path) -> Iterator[HuggingFaceSentimentAdapter]:
    instance = HuggingFaceSentimentAdapter(prepared_dir)
    instance.load()
    yield instance
    instance.close()


def test_metadata_and_preprocessing_contract(adapter: HuggingFaceSentimentAdapter) -> None:
    assert adapter.metadata.key.model_id == "huggingface-sentiment"
    assert adapter.metadata.key.model_version == MODEL_INTERNAL_VERSION
    assert adapter.metadata.labels == ("negative", "positive")
    assert adapter.metadata.device == "cpu"
    assert MODEL_REPO_ID == "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    assert len(MODEL_REVISION) == 40
    assert MODEL_MAX_LENGTH == 256


def test_t12_single_and_batched_predictions_agree(adapter: HuggingFaceSentimentAdapter) -> None:
    texts = [
        "Wonderful.",
        "This movie was painfully slow, poorly written, and much too long.",
        "The performances were excellent even though the premise was simple.",
    ]
    items = [TextInput(text=text) for text in texts]
    singles = [adapter.predict_batch([item])[0] for item in items]
    batched = adapter.predict_batch(items)
    for single, combined in zip(singles, batched, strict=True):
        assert single.label == combined.label
        assert single.scores.negative == pytest.approx(combined.scores.negative, abs=1e-6)
        assert single.scores.positive == pytest.approx(combined.scores.positive, abs=1e-6)


def test_adapter_uses_one_forward_pass_for_a_batch(
    adapter: HuggingFaceSentimentAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    model = adapter._model  # noqa: SLF001 - instrument the real forward boundary.
    original = model.forward

    def recording_forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", recording_forward)
    results = adapter.predict_batch(
        [TextInput(text="I loved it."), TextInput(text="I hated it."), TextInput(text="Fine.")]
    )
    assert len(results) == 3
    assert calls == 1
    assert adapter.last_timings.preprocessing_seconds >= 0
    assert adapter.last_timings.forward_seconds > 0
    assert adapter.last_timings.postprocessing_seconds >= 0


def test_fresh_adapter_reloads_offline_and_reproduces_prediction(
    prepared_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    item = TextInput(text="A smart and genuinely moving film.")
    predictions = []
    for _ in range(2):
        instance = HuggingFaceSentimentAdapter(prepared_dir)
        instance.load()
        predictions.append(instance.predict_batch([item])[0])
        instance.close()
    assert predictions[0].label == predictions[1].label
    assert predictions[0].scores.negative == pytest.approx(predictions[1].scores.negative, abs=1e-8)
    assert predictions[0].scores.positive == pytest.approx(predictions[1].scores.positive, abs=1e-8)


async def test_real_adapter_uses_shared_scheduler_batch(prepared_dir: Path) -> None:
    adapter = HuggingFaceSentimentAdapter(prepared_dir)
    scheduler = ModelScheduler(
        adapter,
        SchedulerConfig(
            policy=SchedulingPolicy.TIMED,
            max_batch_size=3,
            collection_delay_seconds=1,
            pending_capacity=8,
            deadline_seconds=5,
            shutdown_grace_seconds=2,
            watchdog_seconds=10,
        ),
        ServiceMetrics(),
        logging.getLogger("test.huggingface.scheduler"),
    )
    await scheduler.start()
    calls = 0
    model = adapter._model  # noqa: SLF001 - instrument the real forward boundary.
    original = model.forward

    def recording_forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    model.forward = recording_forward
    loop = asyncio.get_running_loop()
    results = await asyncio.gather(
        *(
            scheduler.submit(f"real-{index}", TextInput(text=text), started_at=loop.time())
            for index, text in enumerate(("Excellent.", "Awful.", "I loved this."))
        )
    )
    assert [result.label for result in results] == ["positive", "negative", "positive"]
    assert calls == 1
    await scheduler.shutdown()


async def test_independent_http_requests_share_real_forward_pass(
    prepared_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    app = create_app(
        Settings(
            adapter="huggingface",
            artifact_dir=prepared_dir,
            scheduling_policy=SchedulingPolicy.TIMED,
            max_batch_size=3,
            max_collection_delay_ms=1_000,
            request_deadline_ms=5_000,
        )
    )
    async with app.router.lifespan_context(app):
        adapter = app.state.scheduler.adapter
        calls = 0
        original = adapter._model.forward

        def recording_forward(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        adapter._model.forward = recording_forward
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            responses = await asyncio.gather(
                *(
                    client.post(
                        "/v1/predict",
                        json={
                            "model_id": "huggingface-sentiment",
                            "model_version": MODEL_INTERNAL_VERSION,
                            "input": {"text": text},
                        },
                    )
                    for text in ("Excellent.", "Awful.", "I loved this.")
                )
            )
            assert [response.status_code for response in responses] == [200, 200, 200]
            assert [response.json()["prediction"]["label"] for response in responses] == [
                "positive",
                "negative",
                "positive",
            ]
            metrics = (await client.get("/metrics")).text
            assert 'inference_batch_size_sum{model_id="huggingface-sentiment"' in metrics
            assert " 3.0" in metrics
        assert calls == 1
