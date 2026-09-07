from collections.abc import Sequence
from types import MappingProxyType

from inference_service.adapters.protocol import ModelAdapter
from inference_service.core.contracts import ModelKey, ModelMetadata


class ModelRegistry:
    """Fixed at construction; request callers cannot select arbitrary artifacts."""

    def __init__(self, adapters: Sequence[ModelAdapter]) -> None:
        entries: dict[ModelKey, ModelAdapter] = {}
        for adapter in adapters:
            key = adapter.metadata.key
            if key in entries:
                raise ValueError(f"Duplicate model identity: {key}")
            entries[key] = adapter
        if not entries:
            raise ValueError("At least one model is required.")
        self._adapters = MappingProxyType(entries)

    def get(self, key: ModelKey) -> ModelAdapter:
        return self._adapters[key]

    def models(self) -> tuple[ModelMetadata, ...]:
        return tuple(adapter.metadata for adapter in self._adapters.values())
