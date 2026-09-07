from dataclasses import FrozenInstanceError, replace

import pytest
from pydantic import ValidationError

from inference_service.adapters.fake import FakeAdapter
from inference_service.adapters.protocol import ModelAdapter
from inference_service.core.contracts import ModelKey, ModelMetadata, TextInput
from inference_service.runtime.config import Settings
from inference_service.runtime.registry import ModelRegistry


def test_fake_results_are_deterministic_and_ordered() -> None:
    adapter: ModelAdapter = FakeAdapter()
    adapter.load()
    items = [
        TextInput(text=text) for text in ("excellent", "terrible", "unrecognized", "great bad")
    ]
    results = adapter.predict_batch(items)
    assert [item.label for item in results] == ["positive", "negative", "positive", "positive"]
    assert [item.scores.positive for item in results] == [0.8, 0.2, 0.5, 0.5]
    assert results == [adapter.predict_batch([item])[0] for item in items]
    assert results == adapter.predict_batch(items)
    assert all(item.scores.negative + item.scores.positive == pytest.approx(1) for item in results)
    adapter.close()
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.predict_batch(items)


def test_fake_batch_size_contract() -> None:
    adapter = FakeAdapter()
    adapter.load()
    with pytest.raises(ValueError, match="batch size"):
        adapter.predict_batch([])
    with pytest.raises(ValueError, match="batch size"):
        adapter.predict_batch([TextInput(text="good")] * 9)


def test_registry_uses_immutable_explicit_versions() -> None:
    class SecondVersion(FakeAdapter):
        @property
        def metadata(self) -> ModelMetadata:
            return replace(super().metadata, key=ModelKey("fake-sentiment", "v2"))

    first, second = FakeAdapter(), SecondVersion()
    registry = ModelRegistry([first, second])
    assert registry.get(ModelKey("fake-sentiment", "v1")) is first
    assert registry.get(ModelKey("fake-sentiment", "v2")) is second
    item = TextInput(text="hello")
    assert first.batch_key(item) != second.batch_key(item)
    with pytest.raises(KeyError):
        registry.get(ModelKey("fake-sentiment", "latest"))
    with pytest.raises(FrozenInstanceError):
        first.metadata.key.model_version = "changed"  # type: ignore[misc]  # Test runtime immutability.


def test_registry_rejects_ambiguous_or_empty_configuration() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        ModelRegistry([FakeAdapter(), FakeAdapter()])
    with pytest.raises(ValueError, match="At least one"):
        ModelRegistry([])


@pytest.mark.parametrize("field", ["max_body_bytes", "max_text_characters"])
@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_invalid_settings_fail_startup(
    monkeypatch: pytest.MonkeyPatch, field: str, value: str
) -> None:
    monkeypatch.setenv(f"INFERENCE_{field.upper()}", value)
    with pytest.raises(ValidationError):
        Settings()


def test_environment_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_MAX_TEXT_CHARACTERS", "40")
    assert Settings().max_text_characters == 40
    monkeypatch.setenv("INFERENCE_ADAPTER", "untrusted-hub-id")
    with pytest.raises(ValidationError):
        Settings()
