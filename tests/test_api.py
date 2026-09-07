from collections.abc import AsyncIterator, Sequence
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient

from inference_service.adapters.fake import FakeAdapter
from inference_service.api.app import create_app
from inference_service.core.contracts import Prediction, TextInput
from inference_service.runtime.config import Settings


def payload(text: object = "The acting was excellent.") -> dict[str, object]:
    return {"model_id": "fake-sentiment", "model_version": "v1", "input": {"text": text}}


def test_prediction_and_server_request_id(client: TestClient) -> None:
    response = client.post("/v1/predict", json=payload(), headers={"X-Request-ID": "client-value"})
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "fake-sentiment"
    assert data["model_version"] == "v1"
    assert data["prediction"]["label"] == "positive"
    assert data["prediction"]["scores"]["positive"] == 0.8
    assert str(UUID(data["request_id"])) == response.headers["x-request-id"]
    other = client.post("/v1/predict", json=payload())
    assert other.json()["request_id"] != data["request_id"]


@pytest.mark.parametrize("text", ["", " \n\t", 123, None, [], {}])
def test_invalid_text(client: TestClient, text: object) -> None:
    response = client.post("/v1/predict", json=payload(text))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"
    assert response.json()["request_id"] == response.headers["x-request-id"]


@pytest.mark.parametrize("field", ["model_id", "model_version", "input"])
def test_required_fields(client: TestClient, field: str) -> None:
    body = payload()
    del body[field]
    assert client.post("/v1/predict", json=body).status_code == 422


def test_unknown_fields_and_caller_selected_paths_rejected(client: TestClient) -> None:
    body = payload()
    body["model_path"] = "/some/path"
    assert client.post("/v1/predict", json=body).status_code == 422
    body = payload()
    body["input"] = {"text": "good", "extra": True}
    assert client.post("/v1/predict", json=body).status_code == 422


@pytest.mark.parametrize("field", ["model_id", "model_version"])
def test_unknown_model_identity(client: TestClient, field: str) -> None:
    body = payload()
    body[field] = "not-configured"
    response = client.post("/v1/predict", json=body)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_health_and_model_metadata(client: TestClient) -> None:
    assert client.get("/health/live").json() == {"status": "alive"}
    assert client.get("/health/ready").json() == {"status": "ready"}
    models = client.get("/v1/models").json()["models"]
    assert len(models) == 1
    assert models[0]["available"] is True
    assert models[0]["model_version"] == "v1"
    assert models[0]["labels"] == ["negative", "positive"]


async def test_readiness_tracks_lifespan() -> None:
    app = create_app(Settings())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/health/live")).status_code == 200
        assert (await client.get("/health/ready")).status_code == 503
        assert (await client.post("/v1/predict", json=payload())).status_code == 503
        async with app.router.lifespan_context(app):
            assert (await client.get("/health/ready")).status_code == 200
        assert (await client.get("/health/ready")).status_code == 503
        assert (await client.get("/v1/models")).json()["models"][0]["available"] is False


def test_character_limit_prevents_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    executed: list[TextInput] = []
    original = FakeAdapter.predict_batch

    def recording(self: FakeAdapter, items: Sequence[TextInput]) -> list[Prediction]:
        executed.extend(items)
        return original(self, items)

    monkeypatch.setattr(FakeAdapter, "predict_batch", recording)
    with TestClient(create_app(Settings(max_text_characters=4))) as client:
        assert client.post("/v1/predict", json=payload("good")).status_code == 200
        assert client.post("/v1/predict", json=payload("good!")).status_code == 422
    assert [item.text for item in executed] == ["good"]


def test_body_limit_precedes_json_parsing() -> None:
    with TestClient(create_app(Settings(max_body_bytes=10))) as client:
        # Exactly at the cap reaches JSON validation; beyond it fails before parsing.
        assert client.post("/v1/predict", content=b"x" * 10).status_code == 422
        response = client.post("/v1/predict", content=b"x" * 11)
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "body_too_large"


async def test_body_limit_counts_chunked_utf8_bytes() -> None:
    async def chunks() -> AsyncIterator[bytes]:
        yield "éé".encode()
        yield "éé".encode()

    app = create_app(Settings(max_body_bytes=7))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/predict", content=chunks())
        assert response.status_code == 413
        assert response.json()["request_id"] == response.headers["x-request-id"]


def test_invalid_json_and_routing_errors_are_safe(client: TestClient) -> None:
    response = client.post(
        "/v1/predict", content=b"private-input{", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert "private-input" not in response.text
    missing = client.get("/missing")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "http_error"
    wrong_method = client.get("/v1/predict")
    assert wrong_method.status_code == 405
    assert wrong_method.headers["allow"] == "POST"


def test_execution_failure_does_not_expose_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self: FakeAdapter, items: Sequence[TextInput]) -> list[Prediction]:
        raise RuntimeError("secret model path and private input")

    monkeypatch.setattr(FakeAdapter, "predict_batch", fail)
    with TestClient(create_app(Settings()), raise_server_exceptions=False) as client:
        response = client.post("/v1/predict", json=payload())
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert response.json()["request_id"] == response.headers["x-request-id"]
    assert "secret" not in response.text
    assert "Traceback" not in response.text


def test_wrong_result_count_fails_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    def empty(self: FakeAdapter, items: Sequence[TextInput]) -> list[Prediction]:
        return []

    monkeypatch.setattr(FakeAdapter, "predict_batch", empty)
    with TestClient(create_app(Settings())) as client:
        response = client.post("/v1/predict", json=payload())
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "adapter_contract_error"
