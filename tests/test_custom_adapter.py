from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from inference_service.adapters.custom import CustomSentimentAdapter
from inference_service.api.app import create_app
from inference_service.core.contracts import TextInput
from inference_service.runtime.config import Settings

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def custom_dir() -> Path:
    path = Path("artifacts/custom-sentiment").resolve()
    if not (path / "manifest.json").is_file():
        pytest.skip(
            "Custom artifacts missing; run training.data and training.train with --extra custom"
        )
    return path


@pytest.fixture(scope="module")
def custom(custom_dir: Path) -> Iterator[CustomSentimentAdapter]:
    adapter = CustomSentimentAdapter(custom_dir)
    adapter.load()
    yield adapter
    adapter.close()


def test_t12_custom_single_batch_parity(custom: CustomSentimentAdapter) -> None:
    items = [
        TextInput(text=t)
        for t in ("Great!", "A disappointing and very boring movie.", "!!!", "excellent " * 300)
    ]
    singles = [custom.predict_batch([item])[0] for item in items]
    batch = custom.predict_batch(items)
    for single, batched in zip(singles, batch, strict=True):
        assert single.label == batched.label
        assert single.scores.positive == pytest.approx(batched.scores.positive, abs=1e-6)
        assert single.scores.negative == pytest.approx(batched.scores.negative, abs=1e-6)


def test_t16_fresh_process_reload(
    custom: CustomSentimentAdapter, custom_dir: Path, tmp_path: Path
) -> None:
    texts = ["The food was excellent.", "I regret buying this.", "!!!"]
    expected = custom.predict_batch([TextInput(text=t) for t in texts])
    code = """
import json, socket, sys
def blocked(*args, **kwargs):
    raise AssertionError('Network access forbidden during reload')
socket.socket.connect = blocked
from inference_service.adapters.custom import CustomSentimentAdapter
from inference_service.core.contracts import TextInput
from pathlib import Path
adapter = CustomSentimentAdapter(Path(sys.argv[1]))
adapter.load()
print(json.dumps([p.model_dump() for p in adapter.predict_batch(
    [TextInput(text=t) for t in json.loads(sys.argv[2])])]))
adapter.close()
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(custom_dir), json.dumps(texts)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    actual = json.loads(process.stdout)
    for wanted, result in zip(expected, actual, strict=True):
        assert wanted.label == result["label"]
        assert wanted.scores.positive == pytest.approx(result["scores"]["positive"], abs=1e-6)
        assert wanted.scores.negative == pytest.approx(result["scores"]["negative"], abs=1e-6)


def test_corrupt_weights_fail_startup(custom_dir: Path, tmp_path: Path) -> None:
    path = tmp_path / "model"
    shutil.copytree(custom_dir, path)
    (path / "weights.safetensors").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="integrity.*weights"):
        CustomSentimentAdapter(path).load()


def test_pooling_ignores_padding() -> None:
    import torch

    from inference_service.adapters.custom_artifacts import CustomConfig
    from inference_service.adapters.custom_network import SentimentNetwork

    model = SentimentNetwork(5, CustomConfig())
    with torch.no_grad():
        # A nonzero padding embedding detects reliance on padding_idx alone.
        model.embedding.weight[0].fill_(1000)
        assert torch.allclose(
            model(torch.tensor([[2, 3]])), model(torch.tensor([[2, 3, 0, 0]])), atol=1e-6
        )


async def test_custom_http_requests_share_one_forward(custom_dir: Path) -> None:
    app = create_app(
        Settings(
            adapter="custom",
            custom_artifact_dir=custom_dir,
            max_batch_size=3,
            max_collection_delay_ms=1000,
        )
    )
    texts = ["Excellent food.", "This was disappointing.", "I liked the acting."]
    async with app.router.lifespan_context(app):
        adapter = app.state.scheduler.adapter
        expected = adapter.predict_batch([TextInput(text=t) for t in texts])
        calls = 0
        original = adapter._model.forward

        def forward(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        adapter._model.forward = forward
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            responses = await asyncio.gather(
                *[
                    client.post(
                        "/v1/predict",
                        json={
                            "model_id": "custom-sentiment",
                            "model_version": adapter.metadata.key.model_version,
                            "input": {"text": t},
                        },
                    )
                    for t in texts
                ]
            )
            assert [r.status_code for r in responses] == [200] * 3
            for response, prediction in zip(responses, expected, strict=True):
                actual = response.json()["prediction"]
                assert actual["label"] == prediction.label
                assert actual["scores"]["positive"] == pytest.approx(
                    prediction.scores.positive, abs=1e-6
                )
            assert (await client.get("/health/ready")).status_code == 200
        assert calls == 1
